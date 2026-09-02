#include "carbon/device.hpp"
#include "carbon/cross_section.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/rng.hpp"
#include "carbon/transport.hpp"

#ifdef CARBON_HAS_SYCL

#include <cmath>

#include <exception>
#include <stdexcept>
#include <string>

namespace carbon {
namespace {

const char* backend_name(sycl::backend backend) {
    switch (backend) {
        case sycl::backend::opencl:
            return "opencl";
        case sycl::backend::ext_oneapi_level_zero:
            return "level_zero";
        case sycl::backend::ext_oneapi_cuda:
            return "cuda";
        case sycl::backend::ext_oneapi_hip:
            return "hip";
        case sycl::backend::ext_oneapi_native_cpu:
            return "native_cpu";
        default:
            return "unknown";
    }
}

sycl::property_list make_queue_properties() {
    return sycl::property_list{sycl::property::queue::enable_profiling{},
                               sycl::property::queue::in_order{}};
}

auto make_async_handler() {
    return [](sycl::exception_list exceptions) {
        for (const auto& exception : exceptions) {
            std::rethrow_exception(exception);
        }
    };
}

// Prefer a GPU whose SYCL backend matches `wanted`. Returns negative score
// when the device is not usable for that preference.
int backend_gpu_score(const sycl::device& device, sycl::backend wanted) {
    if (!device.is_gpu()) {
        return -1;
    }
    if (device.get_backend() != wanted) {
        return -1;
    }
    // Prefer devices that can run the accurate double-atomic scorer.
    int score = 1000;
    if (device.has(sycl::aspect::fp64)) {
        score += 100;
    }
    if (device.has(sycl::aspect::atomic64)) {
        score += 100;
    }
    return score;
}

sycl::queue make_backend_gpu_queue(sycl::backend wanted, const std::string& label) {
    try {
        return sycl::queue{
            [=](const sycl::device& device) { return backend_gpu_score(device, wanted); },
            make_async_handler(), make_queue_properties()};
    } catch (const sycl::exception& ex) {
        throw std::runtime_error("No SYCL " + label +
                                 " GPU found (set ONEAPI_DEVICE_SELECTOR or install the "
                                 "matching runtime). Underlying error: " +
                                 std::string(ex.what()));
    }
}

}  // namespace

sycl::queue make_sycl_queue(const std::string& device_name) {
    const auto async_handler = make_async_handler();
    const auto properties = make_queue_properties();

    if (device_name == "gpu") {
        // Uses the default GPU selector. Pin the vendor at runtime with
        // ONEAPI_DEVICE_SELECTOR=level_zero:0 (Intel Arc) or cuda:gpu (NVIDIA).
        return sycl::queue{sycl::gpu_selector_v, async_handler, properties};
    }
    if (device_name == "cpu") {
        // Prefer a true CPU SYCL device; fall back to GPU when host-only OpenCL
        // is unavailable (common on GPU workstations without CPU OpenCL RT).
        try {
            return sycl::queue{sycl::cpu_selector_v, async_handler, properties};
        } catch (const sycl::exception&) {
            return sycl::queue{sycl::gpu_selector_v, async_handler, properties};
        }
    }
    if (device_name == "default") {
        return sycl::queue{sycl::default_selector_v, async_handler, properties};
    }
    // Explicit vendor/backends so multi-GPU hosts can choose without env vars.
    if (device_name == "cuda" || device_name == "nvidia") {
        return make_backend_gpu_queue(sycl::backend::ext_oneapi_cuda, "CUDA/NVIDIA");
    }
    if (device_name == "level_zero" || device_name == "intel" || device_name == "arc") {
        return make_backend_gpu_queue(sycl::backend::ext_oneapi_level_zero,
                                      "Level Zero/Intel");
    }
    if (device_name == "opencl") {
        return make_backend_gpu_queue(sycl::backend::opencl, "OpenCL");
    }
    throw std::invalid_argument(
        "Unknown SYCL device selector: " + device_name +
        " (expected serial path uses host; SYCL: gpu|cpu|default|cuda|nvidia|"
        "level_zero|intel|arc|opencl)");
}

std::string describe_sycl_device(const std::string& device_name) {
    auto queue = make_sycl_queue(device_name);
    const auto& device = queue.get_device();
    std::string description = device.get_info<sycl::info::device::name>() + " | " +
                              device.get_info<sycl::info::device::vendor>() + " | backend " +
                              backend_name(device.get_backend()) + " | driver " +
                              device.get_info<sycl::info::device::driver_version>();
    description += " | fp64=";
    description += device.has(sycl::aspect::fp64) ? "yes" : "no";
    description += " atomic64=";
    description += device.has(sycl::aspect::atomic64) ? "yes" : "no";
    return description;
}

std::vector<float> test_schneider_device_lookup_batch(
    const std::vector<float>& host_table,
    std::uint32_t section_count,
    std::uint32_t energy_nodes,
    float e_min,
    float inv_dE,
    const std::vector<std::uint32_t>& query_sections,
    const std::vector<float>& query_energies,
    const std::vector<float>& query_densities,
    const std::string& device_preference) {
    if (query_sections.size() != query_energies.size() ||
        query_sections.size() != query_densities.size()) {
        throw std::invalid_argument("Query buffer dimension mismatch");
    }
    const std::size_t n_queries = query_sections.size();
    if (n_queries == 0) {
        return {};
    }
    const std::size_t table_elements = static_cast<std::size_t>(section_count) * energy_nodes;
    if (host_table.size() != table_elements) {
        throw std::invalid_argument("Host table size mismatch");
    }
    if (section_count != 25) {
        throw std::invalid_argument("section_count must be 25");
    }
    if (energy_nodes < 2) {
        throw std::invalid_argument("energy_nodes must be >= 2");
    }

    auto queue = make_sycl_queue(device_preference);

    float* table_device = sycl::malloc_device<float>(table_elements, queue);
    std::uint32_t* sections_device = sycl::malloc_device<std::uint32_t>(n_queries, queue);
    float* energies_device = sycl::malloc_device<float>(n_queries, queue);
    float* densities_device = sycl::malloc_device<float>(n_queries, queue);
    float* results_device = sycl::malloc_device<float>(n_queries, queue);

    if (!table_device || !sections_device || !energies_device || !densities_device || !results_device) {
        if (table_device) sycl::free(table_device, queue);
        if (sections_device) sycl::free(sections_device, queue);
        if (energies_device) sycl::free(energies_device, queue);
        if (densities_device) sycl::free(densities_device, queue);
        if (results_device) sycl::free(results_device, queue);
        throw std::runtime_error("Failed to allocate device memory for Schneider lookup batch");
    }

    queue.copy(host_table.data(), table_device, table_elements);
    queue.copy(query_sections.data(), sections_device, n_queries);
    queue.copy(query_energies.data(), energies_device, n_queries);
    queue.copy(query_densities.data(), densities_device, n_queries).wait_and_throw();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{n_queries}, [=](sycl::id<1> idx) {
            const auto i = idx[0];
            results_device[i] = schneider_primary_macroscopic_xs(
                table_device, section_count, energy_nodes, e_min, inv_dE,
                sections_device[i], energies_device[i], densities_device[i]);
        });
    }).wait_and_throw();

    std::vector<float> results(n_queries);
    queue.copy(results_device, results.data(), n_queries).wait_and_throw();

    sycl::free(table_device, queue);
    sycl::free(sections_device, queue);
    sycl::free(energies_device, queue);
    sycl::free(densities_device, queue);
    sycl::free(results_device, queue);

    return results;
}

Step11TestResult run_step11_piecewise_hazard_gpu_test(
    const CtGrid& grid,
    const std::vector<float>& host_schneider_table,
    std::uint32_t section_count,
    std::uint32_t energy_nodes,
    float e_min,
    float inv_dE,
    float energy_mevu,
    std::uint32_t num_histories,
    float ray_origin_x,
    float ray_origin_y,
    float ray_origin_z,
    float ray_dir_x,
    float ray_dir_y,
    float ray_dir_z,
    float max_track_length_mm,
    float bin_width_mm,
    std::uint32_t num_bins,
    const std::string& device_preference) {
    if (num_histories == 0) {
        return {};
    }
    const std::size_t table_elements = static_cast<std::size_t>(section_count) * energy_nodes;
    if (host_schneider_table.size() != table_elements) {
        throw std::invalid_argument("Host table size mismatch");
    }
    const std::size_t voxel_count = static_cast<std::size_t>(grid.nx) * grid.ny * grid.nz;
    if (grid.density_g_per_cm3.size() != voxel_count || grid.material_id.size() != voxel_count) {
        throw std::invalid_argument("Grid dimensions mismatch voxel count");
    }

    auto queue = make_sycl_queue(device_preference);

    float* table_dev = sycl::malloc_device<float>(table_elements, queue);
    float* density_dev = sycl::malloc_device<float>(voxel_count, queue);
    std::uint8_t* material_dev = sycl::malloc_device<std::uint8_t>(voxel_count, queue);

    std::uint64_t* surv_count_dev = sycl::malloc_device<std::uint64_t>(1, queue);
    std::uint64_t* zero_prog_dev = sycl::malloc_device<std::uint64_t>(1, queue);
    std::uint64_t* changed_mat_dev = sycl::malloc_device<std::uint64_t>(1, queue);
    std::uint64_t* face_cross_dev = sycl::malloc_device<std::uint64_t>(1, queue);
    std::uint64_t* bins_dev = sycl::malloc_device<std::uint64_t>(num_bins > 0 ? num_bins : 1, queue);

    queue.copy(host_schneider_table.data(), table_dev, table_elements);
    queue.copy(grid.density_g_per_cm3.data(), density_dev, voxel_count);
    queue.copy(grid.material_id.data(), material_dev, voxel_count);

    queue.memset(surv_count_dev, 0, sizeof(std::uint64_t));
    queue.memset(zero_prog_dev, 0, sizeof(std::uint64_t));
    queue.memset(changed_mat_dev, 0, sizeof(std::uint64_t));
    queue.memset(face_cross_dev, 0, sizeof(std::uint64_t));
    if (num_bins > 0) {
        queue.memset(bins_dev, 0, num_bins * sizeof(std::uint64_t));
    }
    queue.wait_and_throw();

    const float org_x = grid.origin_x_mm;
    const float org_y = grid.origin_y_mm;
    const float org_z = grid.origin_z_mm;
    const float sp_x = grid.spacing_x_mm;
    const float sp_y = grid.spacing_y_mm;
    const float sp_z = grid.spacing_z_mm;
    const std::uint32_t nx = grid.nx;
    const std::uint32_t ny = grid.ny;
    const std::uint32_t nz = grid.nz;

    const float dir_len = sycl::sqrt(ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y + ray_dir_z * ray_dir_z);
    const float dx = dir_len > 1.0e-6F ? ray_dir_x / dir_len : 0.0F;
    const float dy = dir_len > 1.0e-6F ? ray_dir_y / dir_len : 0.0F;
    const float dz = dir_len > 1.0e-6F ? ray_dir_z / dir_len : 1.0F;

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{num_histories}, [=](sycl::id<1> idx) {
            const auto history_id = idx[0];
            float pos_x = ray_origin_x;
            float pos_y = ray_origin_y;
            float pos_z = ray_origin_z;
            float path_traversed = 0.0F;

            float tau_remaining = 0.0F;
            bool tau_active = false;
            std::uint32_t rng_step = 0;
            bool survived = true;
            constexpr std::uint32_t max_steps = 10000;
            std::uint32_t step_count = 0;

            while (path_traversed < max_track_length_mm && step_count < max_steps) {
                step_count++;
                float dens = 0.0F;
                std::uint8_t mat = 0;
                if (!ct_sample(pos_x, pos_y, pos_z, org_x, org_y, org_z, sp_x, sp_y, sp_z,
                               nx, ny, nz, density_dev, material_dev, dens, mat)) {
                    break;
                }

                const float mass_rate = schneider_primary_mass_xs(
                    table_dev, section_count, energy_nodes, e_min, inv_dE,
                    static_cast<std::uint32_t>(mat), energy_mevu);
                const float macro_xs = dens * mass_rate;

                const float max_substep = max_track_length_mm - path_traversed;
                const auto clamp_res = clamp_step_to_ct_faces_exact(
                    max_substep, pos_x, pos_y, pos_z, dx, dy, dz,
                    org_x, org_y, org_z, sp_x, sp_y, sp_z, nx, ny, nz);
                const float t_face = clamp_res.step_mm;

                if (!(t_face > 0.0F)) {
                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        zp(*zero_prog_dev);
                    zp.fetch_add(1U);
                    pos_x += 1.0e-4F * dx;
                    pos_y += 1.0e-4F * dy;
                    pos_z += 1.0e-4F * dz;
                    path_traversed += 1.0e-4F;
                    continue;
                }

                const float test_mid = 0.5F * t_face;
                float dens_mid = 0.0F;
                std::uint8_t mat_mid = 0;
                if (ct_sample(pos_x + test_mid * dx, pos_y + test_mid * dy, pos_z + test_mid * dz,
                              org_x, org_y, org_z, sp_x, sp_y, sp_z, nx, ny, nz,
                              density_dev, material_dev, dens_mid, mat_mid, dx, dy, dz)) {
                    if (mat_mid != mat || sycl::fabs(dens_mid - dens) > 1.0e-4F) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            cm(*changed_mat_dev);
                        cm.fetch_add(1U);
                    }
                }

                float u = 1.0F;
                if (!tau_active) {
                    u = rng::uniform01(1234567ULL, history_id, rng_step++, 8);
                }
                float step_used = t_face;
                const bool collision = consume_schneider_optical_depth_segment(
                    tau_remaining, tau_active, step_used, macro_xs, u);

                if (collision) {
                    pos_x += step_used * dx;
                    pos_y += step_used * dy;
                    pos_z += step_used * dz;
                    path_traversed += step_used;
                    survived = false;

                    if (num_bins > 0 && bin_width_mm > 0.0F) {
                        const auto bin_idx = static_cast<std::size_t>(path_traversed / bin_width_mm);
                        if (bin_idx < num_bins) {
                            sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                b_ref(bins_dev[bin_idx]);
                            b_ref.fetch_add(1U);
                        }
                    }
                    break;
                } else {
                    pos_x += t_face * dx;
                    pos_y += t_face * dy;
                    pos_z += t_face * dz;
                    path_traversed += t_face;

                    if (clamp_res.hit_face) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            fc(*face_cross_dev);
                        fc.fetch_add(1U);

                        if ((clamp_res.axis_mask & 1) != 0 && sycl::fabs(dx) > 1.0e-6F) {
                            const float fx = (pos_x - org_x) / sp_x;
                            const int face_x = static_cast<int>(sycl::round(fx));
                            const float b_x = org_x + static_cast<float>(face_x) * sp_x;
                            pos_x = sycl::nextafter(b_x, dx > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                        if ((clamp_res.axis_mask & 2) != 0 && sycl::fabs(dy) > 1.0e-6F) {
                            const float fy = (pos_y - org_y) / sp_y;
                            const int face_y = static_cast<int>(sycl::round(fy));
                            const float b_y = org_y + static_cast<float>(face_y) * sp_y;
                            pos_y = sycl::nextafter(b_y, dy > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                        if ((clamp_res.axis_mask & 4) != 0 && sycl::fabs(dz) > 1.0e-6F) {
                            const float fz = (pos_z - org_z) / sp_z;
                            const int face_z = static_cast<int>(sycl::round(fz));
                            const float b_z = org_z + static_cast<float>(face_z) * sp_z;
                            pos_z = sycl::nextafter(b_z, dz > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                    }
                }
            }

            if (survived) {
                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    s_ref(*surv_count_dev);
                s_ref.fetch_add(1U);
            }
        });
    }).wait_and_throw();

    Step11TestResult result{};
    std::uint64_t h_surv = 0;
    queue.copy(surv_count_dev, &h_surv, 1);
    queue.copy(zero_prog_dev, &result.zero_progress_count, 1);
    queue.copy(changed_mat_dev, &result.changed_material_step_span_count, 1);
    queue.copy(face_cross_dev, &result.face_cross_count, 1);
    if (num_bins > 0) {
        result.interaction_binned_counts.resize(num_bins);
        queue.copy(bins_dev, result.interaction_binned_counts.data(), num_bins);
    }
    queue.wait_and_throw();

    result.survival_fraction = static_cast<double>(h_surv) / static_cast<double>(num_histories);
    result.survival_fraction_std_err = std::sqrt(
        (result.survival_fraction * (1.0 - result.survival_fraction)) / static_cast<double>(num_histories));

    sycl::free(table_dev, queue);
    sycl::free(density_dev, queue);
    sycl::free(material_dev, queue);
    sycl::free(surv_count_dev, queue);
    sycl::free(zero_prog_dev, queue);
    sycl::free(changed_mat_dev, queue);
    sycl::free(face_cross_dev, queue);
    sycl::free(bins_dev, queue);

    return result;
}

}  // namespace carbon

#endif

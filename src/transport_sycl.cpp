#include "carbon/cross_section.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/device.hpp"
#include "carbon/electron_transport.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/particle.hpp"
#include "carbon/rng.hpp"
#include "carbon/slab_phantom.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/straggling.hpp"
#include "carbon/transport.hpp"

#ifdef CARBON_HAS_SYCL

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace carbon {

#include "detail/sycl_dose_atomic.inc"
#include "detail/sycl_profile.inc"
#include "detail/sycl_transport_context_impl.inc"
#include "detail/sycl_transport_context_methods.inc"

namespace {

#include "detail/sycl_device_math.inc"
#include "detail/sycl_score_device.inc"

float cuda_clock_warmup(sycl::queue& queue) {
    auto* dummy = sycl::malloc_device<float>(1024, queue);
    if (dummy == nullptr) {
        return 0.0F;
    }
    queue.memset(dummy, 0, 1024 * sizeof(float)).wait_and_throw();
    const auto start = std::chrono::steady_clock::now();
    auto event = queue.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{1024}, sycl::range<1>{128}},
        [=](sycl::nd_item<1> item) {
            const auto lane = item.get_global_linear_id();
            if (lane < 1024) {
                dummy[lane] = static_cast<float>(lane) * 1.001F;
            }
        });
    event.wait_and_throw();
    sycl::free(dummy, queue);
    return static_cast<float>(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

}  // namespace

TransportResult transport_sycl(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const CrossSectionTable& cross_section,
                               const std::string& device_name,
                               SyclTransportContext* context) {
    config.validate();
    const auto start = std::chrono::steady_clock::now();
    const auto& table_energies = stopping_power.energies();
    if (table_energies.size() < 2 || !is_uniform_grid(table_energies)) {
        throw std::invalid_argument("Stopping power table must have a uniform energy grid");
    }

    const auto resolved_device_name =
        device_name.empty() ? std::string("gpu") : device_name;

    sycl::queue local_queue = (context != nullptr)
                                  ? context->impl_->queue
                                  : make_sycl_queue(resolved_device_name);
    auto& queue = local_queue;
    const auto device = queue.get_device();
    const auto backend = queue.get_backend();
    const bool is_cuda_backend = backend == sycl::backend::ext_oneapi_cuda;

    const auto free_device = [&](auto* pointer) {
        if (pointer != nullptr) {
            sycl::free(pointer, queue);
        }
    };

    const auto number_of_histories = config.number_of_histories;
    const auto number_of_bins = config.number_of_bins();
    const auto number_of_voxels = config.number_of_voxels();
    const auto table_size = stopping_power.values().size();
    const auto cross_section_table_size = cross_section.values().size();

    const bool reuse_immutable_buffers = context != nullptr;
    if (context != nullptr) {
        context->impl_->ensure_initialized(stopping_power, cross_section);
    }

    float* table_device = context != nullptr ? context->impl_->table_device
                                            : sycl::malloc_device<float>(table_size, queue);
    float* energy_grid_device =
        context != nullptr ? context->impl_->energy_grid_device
                           : (config.enable_csda_range_energy_loss
                                  ? sycl::malloc_device<float>(table_size, queue)
                                  : nullptr);
    float* cumulative_range_device =
        context != nullptr ? context->impl_->cumulative_range_device
                           : (config.enable_csda_range_energy_loss
                                  ? sycl::malloc_device<float>(table_size, queue)
                                  : nullptr);
    float* cross_section_device =
        context != nullptr ? context->impl_->cross_section_device
                           : sycl::malloc_device<float>(cross_section_table_size, queue);

    const auto free_immutable_device = [&](auto* pointer) {
        if (!reuse_immutable_buffers && pointer != nullptr) {
            sycl::free(pointer, queue);
        }
    };

    // Slab layers
    const auto enable_layered_phantom = config.enable_layered_phantom;
    const auto slab_layer_count =
        enable_layered_phantom ? config.slab_layers.size() : std::size_t{0};
    const auto use_material_tables =
        enable_layered_phantom && !config.slab_stopping_power_files.empty();
    const auto material_table_count = use_material_tables ? slab_layer_count : std::size_t{0};

    std::vector<float> slab_z_ends_host(slab_layer_count);
    std::vector<float> slab_densities_host(slab_layer_count);
    std::vector<float> slab_radiation_lengths_host(slab_layer_count);
    for (std::size_t i = 0; i < slab_layer_count; ++i) {
        slab_z_ends_host[i] = static_cast<float>(config.slab_layers[i].z_end_mm);
        slab_densities_host[i] = static_cast<float>(config.slab_layers[i].density_g_per_cm3);
        slab_radiation_lengths_host[i] =
            i < config.slab_radiation_lengths_g_per_cm2.size()
                ? static_cast<float>(config.slab_radiation_lengths_g_per_cm2[i])
                : static_cast<float>(water_radiation_length_g_per_cm2);
    }
    float* slab_z_ends_device = slab_layer_count > 0
                                   ? sycl::malloc_device<float>(slab_layer_count, queue)
                                   : nullptr;
    float* slab_densities_device = slab_layer_count > 0
                                       ? sycl::malloc_device<float>(slab_layer_count, queue)
                                       : nullptr;
    float* slab_radiation_lengths_device =
        slab_layer_count > 0
            ? sycl::malloc_device<float>(slab_layer_count, queue)
            : nullptr;
    if (slab_layer_count > 0) {
        queue.copy(slab_z_ends_host.data(), slab_z_ends_device, slab_layer_count);
        queue.copy(slab_densities_host.data(), slab_densities_device, slab_layer_count);
        queue.copy(slab_radiation_lengths_host.data(), slab_radiation_lengths_device,
                   slab_layer_count).wait_and_throw();
    }

    std::vector<float> material_sp_host(material_table_count * table_size);
    std::vector<float> material_xs_host(material_table_count * cross_section_table_size);
    if (use_material_tables) {
        for (std::size_t layer = 0; layer < slab_layer_count; ++layer) {
            const auto sp = StoppingPowerTable::from_csv(config.slab_stopping_power_files[layer]);
            const auto xs = CrossSectionTable::from_csv(config.slab_cross_section_files[layer]);
            for (std::size_t i = 0; i < table_size; ++i) {
                material_sp_host[layer * table_size + i] = static_cast<float>(sp.values()[i]);
            }
            for (std::size_t i = 0; i < cross_section_table_size; ++i) {
                material_xs_host[layer * cross_section_table_size + i] =
                    static_cast<float>(xs.values()[i]);
            }
        }
    }
    float* material_sp_device = material_table_count > 0
                                   ? sycl::malloc_device<float>(material_sp_host.size(), queue)
                                   : nullptr;
    float* material_xs_device = material_table_count > 0
                                   ? sycl::malloc_device<float>(material_xs_host.size(), queue)
                                   : nullptr;
    if (material_table_count > 0) {
        queue.copy(material_sp_host.data(), material_sp_device, material_sp_host.size());
        queue.copy(material_xs_host.data(), material_xs_device, material_xs_host.size())
            .wait_and_throw();
    }

    // Hetero insert
    const auto enable_hetero_insert = config.enable_hetero_insert;
    const auto insert_x_min = static_cast<float>(config.hetero_insert.x_min_mm);
    const auto insert_x_max = static_cast<float>(config.hetero_insert.x_max_mm);
    const auto insert_y_min = static_cast<float>(config.hetero_insert.y_min_mm);
    const auto insert_y_max = static_cast<float>(config.hetero_insert.y_max_mm);
    const auto insert_z_min = static_cast<float>(config.hetero_insert.z_min_mm);
    const auto insert_z_max = static_cast<float>(config.hetero_insert.z_max_mm);
    const auto insert_density_g_per_cm3 =
        static_cast<float>(config.hetero_insert.density_g_per_cm3);
    const auto insert_radiation_length_g_per_cm2 =
        static_cast<float>(config.insert_radiation_length_g_per_cm2);
    const auto use_insert_material_tables =
        enable_hetero_insert && !config.insert_stopping_power_file.empty();

    std::vector<float> insert_sp_host(use_insert_material_tables ? table_size : 0);
    std::vector<float> insert_xs_host(use_insert_material_tables ? cross_section_table_size : 0);
    if (use_insert_material_tables) {
        const auto sp = StoppingPowerTable::from_csv(config.insert_stopping_power_file);
        const auto xs = CrossSectionTable::from_csv(config.insert_cross_section_file);
        for (std::size_t i = 0; i < table_size; ++i) {
            insert_sp_host[i] = static_cast<float>(sp.values()[i]);
        }
        for (std::size_t i = 0; i < cross_section_table_size; ++i) {
            insert_xs_host[i] = static_cast<float>(xs.values()[i]);
        }
    }
    float* insert_sp_device = use_insert_material_tables
                                  ? sycl::malloc_device<float>(table_size, queue)
                                  : nullptr;
    float* insert_xs_device = use_insert_material_tables
                                  ? sycl::malloc_device<float>(cross_section_table_size, queue)
                                  : nullptr;
    if (use_insert_material_tables) {
        queue.copy(insert_sp_host.data(), insert_sp_device, table_size);
        queue.copy(insert_xs_host.data(), insert_xs_device, cross_section_table_size)
            .wait_and_throw();
    }

    // CT grid
    const auto enable_ct_grid = config.enable_ct_grid;
    float ct_origin_x = 0.0F, ct_origin_y = 0.0F, ct_origin_z = 0.0F;
    float ct_spacing_x = 0.0F, ct_spacing_y = 0.0F, ct_spacing_z = 0.0F;
    std::uint32_t ct_nx = 0, ct_ny = 0, ct_nz = 0;
    float* ct_density_device = nullptr;
    std::uint8_t* ct_material_device = nullptr;
    float* ct_mass_sp_factor_lut_device = nullptr;
    float* ct_mass_sp_za_rel_device = nullptr;
    float* ct_sp_device = nullptr;
    float* ct_xs_device = nullptr;
    float* ct_ref_density_device = nullptr;
    std::uint32_t ct_n_mass_factors = 0;
    std::uint32_t ct_density_spr_n_rho = 0;
    float ct_mass_spr_log_rho_min = 0.0F;
    float ct_mass_spr_inv_dlog = 0.0F;
    bool use_ct_mass_sp = false;
    bool use_ct_density_mass_spr = false;
    bool use_ct_material_sp = false;
    bool use_ct_material_xs = false;
    bool ct_material_ids_are_schneider_sections = false;
    const auto ct_skip_homogeneous_face_clamp = config.ct_skip_homogeneous_face_clamp;

    if (enable_ct_grid) {
        const auto grid = CtGrid::load(
            config.ct_grid_file, config.ct_schneider_file, config.ct_dicom_origin_mode);
        ct_origin_x = grid.origin_x_mm;
        ct_origin_y = grid.origin_y_mm;
        ct_origin_z = grid.origin_z_mm;
        ct_spacing_x = grid.spacing_x_mm;
        ct_spacing_y = grid.spacing_y_mm;
        ct_spacing_z = grid.spacing_z_mm;
        ct_nx = grid.nx;
        ct_ny = grid.ny;
        ct_nz = grid.nz;
        ct_material_ids_are_schneider_sections =
            grid.file_version >= CtGrid::version_v2;

        const auto voxel_count = static_cast<std::size_t>(ct_nx) * ct_ny * ct_nz;
        ct_density_device = sycl::malloc_device<float>(voxel_count, queue);
        ct_material_device = sycl::malloc_device<std::uint8_t>(voxel_count, queue);
        queue.copy(grid.density_g_per_cm3.data(), ct_density_device, voxel_count);
        queue.copy(grid.material_id.data(), ct_material_device, voxel_count).wait_and_throw();

        use_ct_mass_sp = ct_material_ids_are_schneider_sections;
        use_ct_material_sp = !config.ct_water_stopping_power_file.empty() ||
                             !config.ct_bone_stopping_power_file.empty();
        use_ct_material_xs = !config.ct_bone_cross_section_file.empty() ||
                             !config.ct_schneider_cross_section_file.empty();

        if (use_ct_mass_sp) {
            ct_n_mass_factors = static_cast<std::uint32_t>(grid.mass_sp_za_rel.size());
            std::vector<float> mass_factor_lut(
                static_cast<std::size_t>(ct_n_mass_factors) * table_size);
            const auto sp_scale = static_cast<float>(config.ct_stopping_power_scale);

            const auto try_density_spr = [&]() -> bool {
                if (!config.ct_use_density_mass_spr) {
                    return false;
                }
                const auto resolve = [](const std::filesystem::path& user_path,
                                        const std::filesystem::path& fallback) {
                    return !user_path.empty() && std::filesystem::exists(user_path)
                               ? user_path
                               : (std::filesystem::exists(fallback) ? fallback
                                                                    : std::filesystem::path{});
                };
                const auto air_path = resolve(config.ct_air_stopping_power_file,
                                             "data/stopping_power_air_geant4_11_3_2.csv");
                const auto lung_path = resolve(config.ct_lung_stopping_power_file,
                                              "data/stopping_power_lung_geant4_11_3_2.csv");
                const auto bone_path = resolve(config.ct_bone_stopping_power_file,
                                              "data/stopping_power_bone_geant4_11_3_2.csv");
                if (air_path.empty() || lung_path.empty() || bone_path.empty()) {
                    return false;
                }
                const auto air_table = StoppingPowerTable::from_csv(air_path);
                const auto lung_table = StoppingPowerTable::from_csv(lung_path);
                const auto bone_table = StoppingPowerTable::from_csv(bone_path);
                const auto density_lut = build_density_mass_spr_lut(
                    stopping_power, air_table, lung_table, bone_table, sp_scale);
                if (density_lut.n_rho < 2 ||
                    density_lut.factors.size() !=
                        static_cast<std::size_t>(density_lut.n_rho) * table_size) {
                    return false;
                }
                mass_factor_lut = density_lut.factors;
                ct_density_spr_n_rho = density_lut.n_rho;
                ct_mass_spr_log_rho_min = density_lut.log_rho_min;
                ct_mass_spr_inv_dlog = density_lut.inv_dlog;
                use_ct_density_mass_spr = true;
                return true;
            };

            if (!try_density_spr()) {
                for (std::uint32_t sec = 0; sec < ct_n_mass_factors; ++sec) {
                    const auto za = grid.mass_sp_za_rel[sec];
                    const auto I_eV = grid.mass_sp_I_eV[sec];
                    const auto base = static_cast<std::size_t>(sec) * table_size;
                    for (std::size_t i = 0; i < table_size; ++i) {
                        mass_factor_lut[base + i] =
                            sp_scale * ct_mass_sp_energy_factor(
                                           za, I_eV,
                                           static_cast<float>(stopping_power.energies()[i]));
                    }
                }
            }
            const auto lut_bytes = mass_factor_lut.size();
            ct_mass_sp_factor_lut_device = sycl::malloc_device<float>(lut_bytes, queue);
            ct_mass_sp_za_rel_device = sycl::malloc_device<float>(ct_n_mass_factors, queue);
            queue.copy(mass_factor_lut.data(), ct_mass_sp_factor_lut_device, lut_bytes);
            queue.copy(grid.mass_sp_za_rel.data(), ct_mass_sp_za_rel_device, ct_n_mass_factors)
                .wait_and_throw();
        }
    }

    // Spot batching / TPS source
    const auto primary_spot_count = config.primary_spot_batch.size();
    PrimarySpotBatchEntry* primary_spots_device = nullptr;
    if (primary_spot_count > 0) {
        primary_spots_device =
            sycl::malloc_device<PrimarySpotBatchEntry>(primary_spot_count, queue);
        queue.copy(config.primary_spot_batch.data(), primary_spots_device, primary_spot_count)
            .wait_and_throw();
    }

    // Scorers & Result buffers
    const auto enable_voxel_scoring = config.enable_voxel_scoring;
    const auto enable_let_scoring = config.enable_let_scoring;
    const auto voxel_scorer_clamps_transport = config.voxel_scorer_clamps_transport;
    const auto voxel_bins_x = config.voxel_bins_x;
    const auto voxel_bins_y = config.voxel_bins_y;
    const auto voxel_bins_z = config.voxel_bins_z;
    const auto voxel_size_x_mm = static_cast<float>(config.voxel_size_x_mm);
    const auto voxel_size_y_mm = static_cast<float>(config.voxel_size_y_mm);
    const auto voxel_size_z_mm = static_cast<float>(config.voxel_size_z_mm);
    const auto voxel_plane_size = voxel_bins_x * voxel_bins_y;
    float voxel_min_x_mm =
        -0.5F * static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
    float voxel_max_x_mm =
        voxel_min_x_mm + static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
    float voxel_min_y_mm =
        -0.5F * static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    float voxel_max_y_mm =
        voxel_min_y_mm + static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    if (enable_ct_grid &&
        std::abs(ct_spacing_x - voxel_size_x_mm) < 1.0e-5F &&
        std::abs(ct_spacing_y - voxel_size_y_mm) < 1.0e-5F) {
        voxel_min_x_mm = ct_origin_x;
        voxel_min_y_mm = ct_origin_y;
        voxel_max_x_mm =
            ct_origin_x + static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
        voxel_max_y_mm =
            ct_origin_y + static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    }

    auto* dose_device = sycl::malloc_device<DoseAtomicT>(number_of_bins, queue);
    auto* voxel_dose_device = enable_voxel_scoring
                                  ? sycl::malloc_device<DoseAtomicT>(number_of_voxels, queue)
                                  : nullptr;
    auto* let_moments_device = enable_let_scoring
                                   ? sycl::malloc_device<LetAtomicT>(4 * number_of_bins, queue)
                                   : nullptr;
    auto* voxel_let_moments_device =
        enable_let_scoring && enable_voxel_scoring
            ? sycl::malloc_device<LetAtomicT>(4 * number_of_voxels, queue)
            : nullptr;

    auto* deposited_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* escaped_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* steps_device = sycl::malloc_device<std::uint32_t>(number_of_histories, queue);

    if (dose_device == nullptr || deposited_device == nullptr ||
        escaped_device == nullptr || steps_device == nullptr ||
        (enable_voxel_scoring && voxel_dose_device == nullptr) ||
        (enable_let_scoring && let_moments_device == nullptr)) {
        throw std::bad_alloc();
    }

    if (!reuse_immutable_buffers) {
        std::vector<float> table_host(table_size);
        std::transform(stopping_power.values().begin(), stopping_power.values().end(),
                       table_host.begin(),
                       [](double value) { return static_cast<float>(value); });
        queue.copy(table_host.data(), table_device, table_size);
        if (config.enable_csda_range_energy_loss) {
            std::vector<float> energy_grid_host(table_size);
            std::transform(stopping_power.energies().begin(), stopping_power.energies().end(),
                           energy_grid_host.begin(),
                           [](double value) { return static_cast<float>(value); });
            queue.copy(energy_grid_host.data(), energy_grid_device, table_size);
            std::vector<float> cumulative_range_host(table_size);
            std::transform(stopping_power.cumulative_ranges_mm().begin(),
                           stopping_power.cumulative_ranges_mm().end(),
                           cumulative_range_host.begin(),
                           [](double value) { return static_cast<float>(value); });
            queue.copy(cumulative_range_host.data(), cumulative_range_device, table_size);
        }
        std::vector<float> cross_section_host(cross_section_table_size);
        std::transform(cross_section.values().begin(), cross_section.values().end(),
                       cross_section_host.begin(),
                       [](double value) { return static_cast<float>(value); });
        queue.copy(cross_section_host.data(), cross_section_device, cross_section_table_size);
        queue.wait_and_throw();
    }

    queue.memset(dose_device, 0, number_of_bins * sizeof(DoseAtomicT));
    if (enable_voxel_scoring) {
        queue.memset(voxel_dose_device, 0, number_of_voxels * sizeof(DoseAtomicT));
    }
    if (enable_let_scoring) {
        queue.memset(let_moments_device, 0, 4 * number_of_bins * sizeof(LetAtomicT));
        if (voxel_let_moments_device != nullptr) {
            queue.memset(voxel_let_moments_device, 0, 4 * number_of_voxels * sizeof(LetAtomicT));
        }
    }

    const std::size_t local_size = is_cuda_backend ? 128U : (device.is_gpu() ? 256U : 128U);
    std::size_t history_chunk = config.history_chunk_size;
    if (history_chunk == 0) {
        history_chunk = is_cuda_backend ? (number_of_histories > 1000000 ? 16384 : 4096)
                                        : (device.is_gpu() ? 8192 : number_of_histories);
    }
    history_chunk = std::max<std::size_t>(1, history_chunk);

    const auto initial_energy_MeV = static_cast<float>(config.initial_total_energy_MeV());
    const auto beam_energy_spread = static_cast<float>(config.beam_energy_spread);
    const auto phantom_length_mm = static_cast<float>(config.phantom_length_mm);
    const auto depth_bin_width_mm = static_cast<float>(config.depth_bin_width_mm);
    const auto maximum_step_mm = static_cast<float>(config.maximum_step_mm);
    const auto maximum_relative_energy_loss =
        static_cast<float>(config.maximum_relative_energy_loss);
    const auto energy_cutoff_MeV = static_cast<float>(config.energy_cutoff_MeV);

    if (is_cuda_backend && !config.enable_minibeam) {
        cuda_clock_warmup(queue);
    }

    const auto enable_energy_straggling = config.enable_energy_straggling;
    const auto enable_step_stable_straggling =
        enable_energy_straggling && config.enable_step_stable_straggling;
    const auto straggling_sampling_length_mm =
        static_cast<float>(config.straggling_sampling_length_mm);
    const auto straggling_scale = static_cast<float>(config.straggling_scale);
    const auto straggling_sampler = config.straggling_sampler_id();

    std::array<float, max_straggling_scale_points> straggling_scale_energies{};
    std::array<float, max_straggling_scale_points> straggling_scale_values{};
    const auto straggling_scale_point_count =
        config.straggling_scale_energies_MeVu.size();
    for (std::size_t index = 0; index < straggling_scale_point_count; ++index) {
        straggling_scale_energies[index] =
            static_cast<float>(config.straggling_scale_energies_MeVu[index]);
        straggling_scale_values[index] =
            static_cast<float>(config.straggling_scale_values[index]);
    }

    const auto multiple_scattering_scale =
        static_cast<float>(config.multiple_scattering_scale);
    const auto water_density_g_per_cm3 = static_cast<float>(config.water_density_g_per_cm3);
    const auto enable_multiple_scattering = config.enable_multiple_scattering;
    const auto enable_ct_material_mcs = config.enable_ct_material_mcs;
    const auto enable_tps_source = config.uses_fixed_patient_coordinates();
    const auto random_seed = config.random_seed;
    const auto enable_flat_source = config.enable_flat_source;
    const auto flat_source_half_width_x_mm =
        static_cast<float>(config.flat_source_half_width_x_mm);
    const auto flat_source_half_width_y_mm =
        static_cast<float>(config.flat_source_half_width_y_mm);
    const auto enable_emittance_source = config.enable_emittance_source;
    const auto emittance_sigma_x_mm = static_cast<float>(config.emittance_sigma_x_mm);
    const auto emittance_sigma_y_mm = static_cast<float>(config.emittance_sigma_y_mm);
    const auto emittance_sigma_x_prime = static_cast<float>(config.emittance_sigma_x_prime);
    const auto emittance_sigma_y_prime = static_cast<float>(config.emittance_sigma_y_prime);
    const auto source_origin_x_mm = static_cast<float>(config.source_origin_x_mm);
    const auto source_origin_y_mm = static_cast<float>(config.source_origin_y_mm);
    const auto source_origin_z_mm = static_cast<float>(config.source_origin_z_mm);
    const auto beam_ux_x = static_cast<float>(config.beam_ux_x);
    const auto beam_ux_y = static_cast<float>(config.beam_ux_y);
    const auto beam_ux_z = static_cast<float>(config.beam_ux_z);
    const auto beam_uy_x = static_cast<float>(config.beam_uy_x);
    const auto beam_uy_y = static_cast<float>(config.beam_uy_y);
    const auto beam_uy_z = static_cast<float>(config.beam_uy_z);
    const auto beam_uz_x = static_cast<float>(config.beam_uz_x);
    const auto beam_uz_y = static_cast<float>(config.beam_uz_y);
    const auto beam_uz_z = static_cast<float>(config.beam_uz_z);
    const auto emittance_correlation_x = static_cast<float>(config.emittance_correlation_x);
    const auto emittance_correlation_y = static_cast<float>(config.emittance_correlation_y);
    const auto inverse_mass_number = 1.0f / static_cast<float>(config.primary_mass_number);
    const auto primary_mass_number = config.primary_mass_number;
    const auto enable_csda_range_energy_loss = config.enable_csda_range_energy_loss;
    const auto primary_atomic_number = config.primary_atomic_number;
    const auto primary_rest_mass_MeV =
        static_cast<float>(config.resolved_primary_rest_mass_MeV());
    const auto minimum_table_energy = static_cast<float>(stopping_power.energies().front());
    const auto inverse_table_step =
        1.0f / static_cast<float>(stopping_power.energies()[1] - stopping_power.energies()[0]);

    double primary_kernel_seconds = 0.0;

    for (std::size_t hist_offset = 0; hist_offset < number_of_histories;
         hist_offset += history_chunk) {
        const auto chunk_count =
            std::min(history_chunk, number_of_histories - hist_offset);
        const auto chunk_global =
            ((chunk_count + local_size - 1) / local_size) * local_size;
        auto kernel_event = queue.parallel_for(
            sycl::nd_range<1>{sycl::range<1>{chunk_global}, sycl::range<1>{local_size}},
            [=](sycl::nd_item<1> item) {
                const auto lane = item.get_global_linear_id();
                if (lane >= chunk_count) {
                    return;
                }
                const auto global_history = hist_offset + lane;

                const PrimarySpotBatchEntry* spot = nullptr;
                std::uint64_t rng_history = global_history;
                if (primary_spot_count > 0) {
                    std::size_t lower = 0;
                    std::size_t upper = primary_spot_count;
                    while (lower + 1 < upper) {
                        const auto middle = lower + (upper - lower) / 2;
                        if (global_history < primary_spots_device[middle].history_begin) {
                            upper = middle;
                        } else {
                            lower = middle;
                        }
                    }
                    spot = primary_spots_device + lower;
                    rng_history = global_history - spot->history_begin;
                }
                const auto spot_seed = spot != nullptr ? spot->random_seed : random_seed;
                const auto spot_initial_energy_MeV =
                    spot != nullptr ? spot->floats[0] : initial_energy_MeV;
                const auto spot_energy_spread =
                    spot != nullptr ? spot->floats[1] : beam_energy_spread;

                auto energy_MeV = spot_initial_energy_MeV;
                if (spot_energy_spread > 0.0F) {
                    const auto u0 = sycl::fmax(
                        rng::uniform01(spot_seed, rng_history, 0, 40), 1.0e-12F);
                    const auto u1 = rng::uniform01(spot_seed, rng_history, 0, 41);
                    constexpr float two_pi = 6.2831853071795864769F;
                    const auto gauss =
                        sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                    energy_MeV =
                        spot_initial_energy_MeV * (1.0F + spot_energy_spread * gauss);
                    if (energy_MeV < energy_cutoff_MeV) {
                        energy_MeV = energy_cutoff_MeV;
                    }
                }

                auto local_x_mm = 0.0F;
                auto local_y_mm = 0.0F;
                auto local_dx = 0.0F;
                auto local_dy = 0.0F;
                auto local_dz = 1.0F;
                if (enable_flat_source) {
                    local_x_mm = flat_source_half_width_x_mm *
                                 (2.0F * rng::uniform01(spot_seed, rng_history, 0, 34) - 1.0F);
                    local_y_mm = flat_source_half_width_y_mm *
                                 (2.0F * rng::uniform01(spot_seed, rng_history, 0, 35) - 1.0F);
                } else if (enable_emittance_source) {
                    const auto u0 = sycl::fmax(
                        rng::uniform01(spot_seed, rng_history, 0, 30), 1.0e-12F);
                    const auto u1 = rng::uniform01(spot_seed, rng_history, 0, 31);
                    const auto u2 = sycl::fmax(
                        rng::uniform01(spot_seed, rng_history, 0, 32), 1.0e-12F);
                    const auto u3 = rng::uniform01(spot_seed, rng_history, 0, 33);
                    constexpr float two_pi = 6.2831853071795864769F;
                    const auto g0 = sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                    const auto g1 = sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::sin(two_pi * u1);
                    const auto g2 = sycl::sqrt(-2.0F * sycl::log(u2)) * sycl::cos(two_pi * u3);
                    const auto g3 = sycl::sqrt(-2.0F * sycl::log(u2)) * sycl::sin(two_pi * u3);
                    const auto sigma_x = spot != nullptr ? spot->floats[2] : emittance_sigma_x_mm;
                    const auto sigma_y = spot != nullptr ? spot->floats[3] : emittance_sigma_y_mm;
                    const auto sigma_x_prime =
                        spot != nullptr ? spot->floats[4] : emittance_sigma_x_prime;
                    const auto sigma_y_prime =
                        spot != nullptr ? spot->floats[5] : emittance_sigma_y_prime;
                    local_x_mm = sigma_x * g0;
                    local_y_mm = sigma_y * g2;
                    const auto rho_x = sycl::clamp(
                        spot != nullptr ? spot->floats[6] : emittance_correlation_x,
                        -0.9999F, 0.9999F);
                    const auto rho_y = sycl::clamp(
                        spot != nullptr ? spot->floats[7] : emittance_correlation_y,
                        -0.9999F, 0.9999F);
                    const auto x_prime =
                        sigma_x_prime * (rho_x * g0 + sycl::sqrt(1.0F - rho_x * rho_x) * g1);
                    const auto y_prime =
                        sigma_y_prime * (rho_y * g2 + sycl::sqrt(1.0F - rho_y * rho_y) * g3);
                    const auto inv_norm =
                        sycl::rsqrt(1.0F + x_prime * x_prime + y_prime * y_prime);
                    local_dx = x_prime * inv_norm;
                    local_dy = y_prime * inv_norm;
                    local_dz = inv_norm;
                }

                const auto origin_x = spot != nullptr ? spot->floats[8] : source_origin_x_mm;
                const auto origin_y = spot != nullptr ? spot->floats[9] : source_origin_y_mm;
                const auto origin_z = spot != nullptr ? spot->floats[10] : source_origin_z_mm;
                const auto ux_x = spot != nullptr ? spot->floats[11] : beam_ux_x;
                const auto ux_y = spot != nullptr ? spot->floats[12] : beam_ux_y;
                const auto ux_z = spot != nullptr ? spot->floats[13] : beam_ux_z;
                const auto uy_x = spot != nullptr ? spot->floats[14] : beam_uy_x;
                const auto uy_y = spot != nullptr ? spot->floats[15] : beam_uy_y;
                const auto uy_z = spot != nullptr ? spot->floats[16] : beam_uy_z;
                const auto uz_x = spot != nullptr ? spot->floats[17] : beam_uz_x;
                const auto uz_y = spot != nullptr ? spot->floats[18] : beam_uz_y;
                const auto uz_z = spot != nullptr ? spot->floats[19] : beam_uz_z;

                auto position_x_mm = origin_x + ux_x * local_x_mm + uy_x * local_y_mm;
                auto position_y_mm = origin_y + ux_y * local_x_mm + uy_y * local_y_mm;
                auto position_z_mm = origin_z + ux_z * local_x_mm + uy_z * local_y_mm;
                auto direction_x = ux_x * local_dx + uy_x * local_dy + uz_x * local_dz;
                auto direction_y = ux_y * local_dx + uy_x * local_dy + uz_y * local_dz;
                auto direction_z = ux_z * local_dx + uy_z * local_dy + uz_z * local_dz;
                {
                    const auto inv_n = sycl::rsqrt(sycl::fmax(
                        1.0e-20F, direction_x * direction_x + direction_y * direction_y +
                                      direction_z * direction_z));
                    direction_x *= inv_n;
                    direction_y *= inv_n;
                    direction_z *= inv_n;
                }

                if (enable_tps_source) {
                    auto t_enter = 0.0F;
                    auto t_exit = 1.0e30F;
                    auto hit = true;
                    auto intersect_slab = [&](const float position, const float direction,
                                              const float lower, const float upper) {
                        if (sycl::fabs(direction) < 1.0e-8F) {
                            if (position < lower || position >= upper) {
                                hit = false;
                            }
                            return;
                        }
                        auto first = (lower - position) / direction;
                        auto second = (upper - position) / direction;
                        if (first > second) {
                            const auto temporary = first;
                            first = second;
                            second = temporary;
                        }
                        t_enter = sycl::fmax(t_enter, first);
                        t_exit = sycl::fmin(t_exit, second);
                        if (t_exit < t_enter) {
                            hit = false;
                        }
                    };
                    if (enable_voxel_scoring) {
                        intersect_slab(position_x_mm, direction_x, voxel_min_x_mm, voxel_max_x_mm);
                        intersect_slab(position_y_mm, direction_y, voxel_min_y_mm, voxel_max_y_mm);
                    }
                    intersect_slab(position_z_mm, direction_z, 0.0F, phantom_length_mm);
                    if (hit && t_exit >= t_enter) {
                        const auto entry = t_enter + 1.0e-4F;
                        position_x_mm += entry * direction_x;
                        position_y_mm += entry * direction_y;
                        position_z_mm += entry * direction_z;
                    }
                } else {
                    if (sycl::fabs(direction_z) > 1.0e-8F) {
                        const auto t_plane = -position_z_mm / direction_z;
                        position_x_mm += t_plane * direction_x;
                        position_y_mm += t_plane * direction_y;
                        position_z_mm = 0.0F;
                    }
                }

                auto history_deposited_MeV = 0.0F;
                std::uint32_t steps = 0;
                StepStableStragglingState<float> stable_straggling;
                stable_straggling.initialize(straggling_sampling_length_mm);

                double pending_primary_depth_MeV = 0.0;
                double pending_let_numerator = 0.0;
                double pending_let_denominator = 0.0;
                double pending_voxel_let_numerator = 0.0;
                double pending_voxel_let_denominator = 0.0;
                int pending_primary_bin = 0;
                double pending_primary_voxel_MeV = 0.0;
                std::size_t pending_primary_voxel = 0;
                auto last_primary_stopping_power_MeV_per_mm = 0.0F;
                auto last_primary_density_g_per_cm3 = 0.0F;

                constexpr std::uint32_t max_primary_steps = 2000000U;
                while (energy_MeV > energy_cutoff_MeV && steps < max_primary_steps) {
                    const auto escaped_z =
                        position_z_mm < 0.0F || position_z_mm >= phantom_length_mm;
                    const auto escaped_xy =
                        enable_voxel_scoring &&
                        (position_x_mm < voxel_min_x_mm || position_x_mm >= voxel_max_x_mm ||
                         position_y_mm < voxel_min_y_mm || position_y_mm >= voxel_max_y_mm);
                    if (escaped_z || escaped_xy) {
                        break;
                    }

                    const auto absolute_direction_x = sycl::fabs(direction_x);
                    const auto absolute_direction_y = sycl::fabs(direction_y);
                    const auto absolute_direction_z = sycl::fabs(direction_z);
                    auto bin = direction_z < 0.0F
                                   ? static_cast<int>(
                                         sycl::ceil(position_z_mm / depth_bin_width_mm)) - 1
                                   : static_cast<int>(
                                         sycl::floor(position_z_mm / depth_bin_width_mm));
                    bin = sycl::max(0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));

                    auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                    auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                    if (enable_voxel_scoring) {
                        const auto x_coordinate =
                            (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                        const auto y_coordinate =
                            (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                        voxel_x = direction_x < 0.0F
                                      ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                      : static_cast<int>(sycl::floor(x_coordinate));
                        voxel_y = direction_y < 0.0F
                                      ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                      : static_cast<int>(sycl::floor(y_coordinate));
                        voxel_x = sycl::max(
                            0, sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                        voxel_y = sycl::max(
                            0, sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                    }
                    const auto voxel_index = static_cast<std::size_t>(bin) * voxel_plane_size +
                                             static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                             static_cast<std::size_t>(voxel_x);

                    const auto energy_MeVu = energy_MeV * inverse_mass_number;
                    auto floating_index = (energy_MeVu - minimum_table_energy) * inverse_table_step;
                    auto index = static_cast<int>(sycl::floor(floating_index));
                    index = sycl::max(0, sycl::min(index, static_cast<int>(table_size) - 2));
                    const auto fraction =
                        sycl::clamp(floating_index - static_cast<float>(index), 0.0F, 1.0F);

                    const auto in_insert =
                        enable_hetero_insert &&
                        inside_hetero_insert(position_x_mm, position_y_mm, position_z_mm,
                                             insert_x_min, insert_x_max, insert_y_min,
                                             insert_y_max, insert_z_min, insert_z_max);
                    auto local_density_g_per_cm3 = slab_density_g_per_cm3(
                        position_z_mm, slab_z_ends_device, slab_densities_device,
                        slab_layer_count, water_density_g_per_cm3);
                    if (in_insert) {
                        local_density_g_per_cm3 = insert_density_g_per_cm3;
                    }
                    std::uint8_t ct_material = 2;
                    auto in_ct = false;
                    if (enable_ct_grid) {
                        float ct_rho = water_density_g_per_cm3;
                        in_ct = ct_sample(position_x_mm, position_y_mm, position_z_mm,
                                          ct_origin_x, ct_origin_y, ct_origin_z, ct_spacing_x,
                                          ct_spacing_y, ct_spacing_z, ct_nx, ct_ny, ct_nz,
                                          ct_density_device, ct_material_device, ct_rho,
                                          ct_material);
                        if (in_ct) {
                            local_density_g_per_cm3 = ct_rho;
                        }
                    }

                    const auto layer_for_material =
                        slab_layer_count > 0
                            ? slab_layer_index(position_z_mm, slab_z_ends_device, slab_layer_count)
                            : 0U;
                    float stopping_power_MeV_per_mm = 0.0F;
                    if (enable_ct_grid) {
                        const auto water_sp =
                            table_device[index] +
                            fraction * (table_device[index + 1] - table_device[index]);
                        if (use_ct_mass_sp && in_ct && ct_mass_sp_factor_lut_device != nullptr &&
                            ct_n_mass_factors > 0) {
                            const auto mass_factor = ct_lookup_mass_sp_factor(
                                ct_mass_sp_factor_lut_device,
                                use_ct_density_mass_spr ? ct_density_spr_n_rho : ct_n_mass_factors,
                                table_size, use_ct_density_mass_spr,
                                ct_mass_spr_log_rho_min, ct_mass_spr_inv_dlog,
                                static_cast<std::uint32_t>(ct_material), local_density_g_per_cm3,
                                static_cast<std::size_t>(index), fraction,
                                [](float x) { return sycl::log(x); });
                            stopping_power_MeV_per_mm = ct_mass_scaled_stopping_power(
                                water_sp, local_density_g_per_cm3, mass_factor);
                        } else {
                            stopping_power_MeV_per_mm =
                                water_sp * sycl::fmax(local_density_g_per_cm3, 1.0e-6F);
                        }
                    } else if (in_insert && use_insert_material_tables) {
                        stopping_power_MeV_per_mm =
                            insert_sp_device[static_cast<std::size_t>(index)] +
                            fraction *
                                (insert_sp_device[static_cast<std::size_t>(index) + 1] -
                                 insert_sp_device[static_cast<std::size_t>(index)]);
                    } else if (material_table_count > 0) {
                        const auto base = static_cast<std::size_t>(layer_for_material) * table_size;
                        stopping_power_MeV_per_mm =
                            material_sp_device[base + static_cast<std::size_t>(index)] +
                            fraction *
                                (material_sp_device[base + static_cast<std::size_t>(index) + 1] -
                                 material_sp_device[base + static_cast<std::size_t>(index)]);
                    } else {
                        const auto table_stopping_power_MeV_per_mm =
                            table_device[index] +
                            fraction * (table_device[index + 1] - table_device[index]);
                        const auto scale_density =
                            (slab_layer_count > 0 || in_insert) ? local_density_g_per_cm3 : 1.0F;
                        stopping_power_MeV_per_mm =
                            (slab_layer_count > 0 || in_insert)
                                ? table_stopping_power_MeV_per_mm * scale_density
                                : table_stopping_power_MeV_per_mm;
                    }

                    auto step_mm = sycl::fmin(
                        maximum_step_mm,
                        maximum_relative_energy_loss * energy_MeV /
                            sycl::fmax(stopping_power_MeV_per_mm, 1.0e-6F));
                    if (absolute_direction_z >= 1.0e-6F) {
                        const auto boundary_z_mm =
                            direction_z < 0.0F
                                ? static_cast<float>(bin) * depth_bin_width_mm
                                : static_cast<float>(bin + 1) * depth_bin_width_mm;
                        const auto dz_step = (boundary_z_mm - position_z_mm) / direction_z;
                        if (dz_step > 1.0e-5F) {
                            step_mm = sycl::fmin(step_mm, dz_step);
                        }
                    }
                    if (slab_layer_count > 0) {
                        const auto slab_step = distance_to_slab_interface_mm(
                            position_z_mm, direction_z, slab_z_ends_device, slab_layer_count,
                            phantom_length_mm);
                        if (slab_step > 1.0e-5F) {
                            step_mm = sycl::fmin(step_mm, slab_step);
                        }
                    }
                    if (enable_hetero_insert) {
                        const auto insert_step = distance_to_insert_interface_mm(
                            position_x_mm, position_y_mm, position_z_mm, direction_x,
                            direction_y, direction_z, insert_x_min, insert_x_max, insert_y_min,
                            insert_y_max, insert_z_min, insert_z_max, phantom_length_mm);
                        if (insert_step > 1.0e-5F) {
                            step_mm = sycl::fmin(step_mm, insert_step);
                        }
                    }
                    if (enable_ct_grid && in_ct) {
                        step_mm = clamp_step_to_ct_faces_near_z_if_needed(
                            step_mm, position_x_mm, position_y_mm, position_z_mm,
                            direction_x, direction_y, direction_z, ct_origin_x,
                            ct_origin_y, ct_origin_z, ct_spacing_x, ct_spacing_y,
                            ct_spacing_z, ct_nx, ct_ny, ct_nz, ct_density_device,
                            ct_material_device, local_density_g_per_cm3, ct_material,
                            ct_skip_homogeneous_face_clamp, nullptr);
                    }
                    if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                        absolute_direction_x >= 1.0e-6F) {
                        const auto boundary_x_mm =
                            voxel_min_x_mm +
                            static_cast<float>(voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                                voxel_size_x_mm;
                        const auto dx_step = (boundary_x_mm - position_x_mm) / direction_x;
                        if (dx_step > 1.0e-5F) {
                            step_mm = sycl::fmin(step_mm, dx_step);
                        }
                    }
                    if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                        absolute_direction_y >= 1.0e-6F) {
                        const auto boundary_y_mm =
                            voxel_min_y_mm +
                            static_cast<float>(voxel_y + (direction_y > 0.0F ? 1 : 0)) *
                                voxel_size_y_mm;
                        const auto dy_step = (boundary_y_mm - position_y_mm) / direction_y;
                        if (dy_step > 1.0e-5F) {
                            step_mm = sycl::fmin(step_mm, dy_step);
                        }
                    }

                    if (enable_energy_straggling && enable_step_stable_straggling) {
                        stable_straggling.prepare_step(
                            step_mm, phantom_length_mm - position_z_mm);
                    }
                    step_mm = sycl::fmax(step_mm, 1.0e-5F);

                    float mean_loss_MeV = 0.0F;
                    if (enable_csda_range_energy_loss && energy_grid_device != nullptr &&
                        cumulative_range_device != nullptr) {
                        const auto end_energy_MeVu = csda_energy_after_distance_device(
                            energy_grid_device, table_device, cumulative_range_device,
                            table_size, energy_MeVu, step_mm, primary_mass_number);
                        mean_loss_MeV = sycl::clamp(
                            energy_MeV -
                                static_cast<float>(primary_mass_number) * end_energy_MeVu,
                            0.0F, energy_MeV);
                    } else if (material_table_count == 0 && !in_insert &&
                               (!use_ct_mass_sp || !in_ct)) {
                        const auto mid_energy_MeV = sycl::fmax(
                            minimum_table_energy * static_cast<float>(primary_mass_number),
                            energy_MeV - 0.5F * stopping_power_MeV_per_mm * step_mm);
                        const auto mid_energy_MeVu = mid_energy_MeV * inverse_mass_number;
                        const auto mid_floating_index =
                            (mid_energy_MeVu - minimum_table_energy) * inverse_table_step;
                        auto mid_index =
                            static_cast<int>(sycl::floor(mid_floating_index));
                        mid_index = sycl::max(
                            0, sycl::min(mid_index, static_cast<int>(table_size) - 2));
                        const auto mid_fraction = sycl::clamp(
                            mid_floating_index - static_cast<float>(mid_index),
                            0.0F, 1.0F);
                        const auto mid_sp =
                            (table_device[mid_index] +
                             mid_fraction * (table_device[mid_index + 1] -
                                             table_device[mid_index])) *
                            sycl::fmax(local_density_g_per_cm3, 1.0e-6F);
                        mean_loss_MeV = mid_sp * step_mm;
                    } else {
                        mean_loss_MeV = stopping_power_MeV_per_mm * step_mm;
                    }
                    auto deposited_MeV = sycl::fmin(mean_loss_MeV, energy_MeV);

                    if (enable_energy_straggling) {
                        const auto local_scale = interpolate_straggling_scale(
                            energy_MeVu, straggling_scale_energies, straggling_scale_values,
                            straggling_scale_point_count, straggling_scale);
                        const auto effective_charge =
                            ion_effective_charge_device(primary_atomic_number, energy_MeVu);
                        const auto variance_MeV2 =
                            condensed_total_loss_variance_with_mass_MeV2_device(
                                energy_MeV, primary_rest_mass_MeV, effective_charge, step_mm,
                                local_density_g_per_cm3);
                        if (enable_step_stable_straggling) {
                            if (!stable_straggling.block_active) {
                                const auto u0 = sycl::fmax(
                                    rng::uniform01(spot_seed, rng_history,
                                                   stable_straggling.block_index, 0),
                                    1.0e-12F);
                                const auto u1 = rng::uniform01(
                                    spot_seed, rng_history, stable_straggling.block_index, 1);
                                const auto u2 = rng::uniform01(
                                    spot_seed, rng_history, stable_straggling.block_index, 2);
                                constexpr float two_pi = 6.2831853071795864769F;
                                const auto gauss =
                                    sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                                stable_straggling.begin_block(
                                    mean_loss_MeV, variance_MeV2, step_mm, local_scale, gauss,
                                    energy_MeV, u2, straggling_sampler);
                            }
                            deposited_MeV = stable_straggling.consume_loss(step_mm, energy_MeV);
                        } else {
                            const auto u0 = sycl::fmax(
                                rng::uniform01(spot_seed, rng_history, steps, 0), 1.0e-12F);
                            const auto u1 = rng::uniform01(spot_seed, rng_history, steps, 1);
                            const auto u2 = rng::uniform01(spot_seed, rng_history, steps, 2);
                            constexpr float two_pi = 6.2831853071795864769F;
                            const auto gauss =
                                sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                            const auto sigma_MeV =
                                local_scale * sycl::sqrt(sycl::fmax(0.0F, variance_MeV2));
                            deposited_MeV = sample_condensed_energy_loss(
                                mean_loss_MeV, sigma_MeV, gauss, u2, energy_MeV, straggling_sampler);
                        }
                    }

                    if (bin != pending_primary_bin) {
                        if (pending_primary_depth_MeV > 0.0) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_dose(dose_device[pending_primary_bin]);
                            atomic_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_depth_MeV));
                            pending_primary_depth_MeV = 0.0;
                        }
                        if (enable_let_scoring) {
                            flush_letd_moments_device(
                                let_moments_device, number_of_bins,
                                static_cast<std::size_t>(pending_primary_bin), nullptr, 0, 0,
                                pending_let_numerator, pending_let_denominator, true);
                            pending_let_numerator = 0.0;
                            pending_let_denominator = 0.0;
                        }
                        pending_primary_bin = bin;
                    }
                    if (enable_voxel_scoring && voxel_index != pending_primary_voxel) {
                        if (pending_primary_voxel_MeV > 0.0) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_voxel_dose(voxel_dose_device[pending_primary_voxel]);
                            atomic_voxel_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                            pending_primary_voxel_MeV = 0.0;
                        }
                        if (enable_let_scoring && voxel_let_moments_device != nullptr) {
                            flush_letd_moments_device(
                                voxel_let_moments_device, number_of_voxels,
                                pending_primary_voxel, nullptr, 0, 0,
                                pending_voxel_let_numerator, pending_voxel_let_denominator, true);
                            pending_voxel_let_numerator = 0.0;
                            pending_voxel_let_denominator = 0.0;
                        }
                        pending_primary_voxel = voxel_index;
                    }

                    pending_primary_depth_MeV += deposited_MeV;
                    if (enable_voxel_scoring) {
                        pending_primary_voxel_MeV += deposited_MeV;
                    }
                    history_deposited_MeV += deposited_MeV;
                    last_primary_stopping_power_MeV_per_mm = stopping_power_MeV_per_mm;
                    last_primary_density_g_per_cm3 = local_density_g_per_cm3;

                    if (enable_let_scoring) {
                        const auto step_numerator =
                            static_cast<double>(deposited_MeV) *
                            static_cast<double>(stopping_power_MeV_per_mm) /
                            static_cast<double>(sycl::fmax(local_density_g_per_cm3, 1.0e-6F));
                        const auto step_denominator = static_cast<double>(deposited_MeV);
                        pending_let_numerator += step_numerator;
                        pending_let_denominator += step_denominator;
                        if (voxel_let_moments_device != nullptr) {
                            pending_voxel_let_numerator += step_numerator;
                            pending_voxel_let_denominator += step_denominator;
                        }
                    }

                    if (enable_multiple_scattering) {
                        auto radiation_length_g_per_cm2 =
                            static_cast<float>(water_radiation_length_g_per_cm2);
                        if (enable_ct_grid && in_ct && enable_ct_material_mcs) {
                            radiation_length_g_per_cm2 =
                                ct_material_radiation_length_g_per_cm2(
                                    ct_material_class(ct_material, ct_material_ids_are_schneider_sections));
                        } else if (in_insert) {
                            radiation_length_g_per_cm2 = insert_radiation_length_g_per_cm2;
                        } else if (slab_layer_count > 0) {
                            radiation_length_g_per_cm2 =
                                slab_radiation_lengths_device[layer_for_material];
                        }
                        const auto theta0 = highland_projected_rms_angle_device(
                            energy_MeV, primary_atomic_number, primary_mass_number, step_mm,
                            local_density_g_per_cm3, radiation_length_g_per_cm2) *
                            multiple_scattering_scale;
                        const auto u0 = sycl::fmax(
                            rng::uniform01(spot_seed, rng_history, steps, 3), 1.0e-12F);
                        const auto u1 = rng::uniform01(spot_seed, rng_history, steps, 4);
                        const auto u2 = sycl::fmax(
                            rng::uniform01(spot_seed, rng_history, steps, 5), 1.0e-12F);
                        const auto u3 = rng::uniform01(spot_seed, rng_history, steps, 6);
                        constexpr float two_pi = 6.2831853071795864769F;
                        const auto theta_x =
                            theta0 * sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                        const auto theta_y =
                            theta0 * sycl::sqrt(-2.0F * sycl::log(u2)) * sycl::cos(two_pi * u3);
                        const auto transverse_magnitude =
                            sycl::sqrt(theta_x * theta_x + theta_y * theta_y);
                        const auto local_direction_z =
                            sycl::cos(sycl::fmin(transverse_magnitude, 1.5707963F));
                        const auto transverse_scale =
                            transverse_magnitude > 0.0F
                                ? sycl::sin(sycl::fmin(transverse_magnitude, 1.5707963F)) /
                                      transverse_magnitude
                                : 1.0F;
                        const auto rotated = rotate_local_direction(
                            theta_x * transverse_scale, theta_y * transverse_scale,
                            local_direction_z,
                            Direction3F{direction_x, direction_y, direction_z});
                        direction_x = rotated.x;
                        direction_y = rotated.y;
                        direction_z = rotated.z;
                    }

                    position_x_mm += direction_x * step_mm;
                    position_y_mm += direction_y * step_mm;
                    position_z_mm += direction_z * step_mm;

                    if (absolute_direction_z >= 1.0e-6F) {
                        const auto boundary_z_mm =
                            direction_z < 0.0F
                                ? static_cast<float>(bin) * depth_bin_width_mm
                                : static_cast<float>(bin + 1) * depth_bin_width_mm;
                        if (sycl::fabs(boundary_z_mm - position_z_mm) <= 1.0e-5F) {
                            position_z_mm = sycl::nextafter(
                                boundary_z_mm,
                                direction_z > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                    }
                    if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                        absolute_direction_x >= 1.0e-6F) {
                        const auto boundary_x_mm =
                            voxel_min_x_mm +
                            static_cast<float>(voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                                voxel_size_x_mm;
                        if (sycl::fabs(boundary_x_mm - position_x_mm) <= 1.0e-5F) {
                            position_x_mm = sycl::nextafter(
                                boundary_x_mm,
                                direction_x > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                    }
                    if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                        absolute_direction_y >= 1.0e-6F) {
                        const auto boundary_y_mm =
                            voxel_min_y_mm +
                            static_cast<float>(voxel_y + (direction_y > 0.0F ? 1 : 0)) *
                                voxel_size_y_mm;
                        if (sycl::fabs(boundary_y_mm - position_y_mm) <= 1.0e-5F) {
                            position_y_mm = sycl::nextafter(
                                boundary_y_mm,
                                direction_y > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                    }

                    energy_MeV -= deposited_MeV;
                    ++steps;
                }

                if (energy_MeV > 0.0F && energy_MeV <= energy_cutoff_MeV &&
                    position_z_mm >= 0.0F && position_z_mm < phantom_length_mm) {
                    const auto cutoff_energy_MeV = energy_MeV;
                    pending_primary_depth_MeV += cutoff_energy_MeV;
                    if (enable_voxel_scoring) {
                        pending_primary_voxel_MeV += cutoff_energy_MeV;
                    }
                    history_deposited_MeV += cutoff_energy_MeV;
                    if (enable_let_scoring) {
                        const auto cutoff_numerator =
                            static_cast<double>(cutoff_energy_MeV) *
                            static_cast<double>(last_primary_stopping_power_MeV_per_mm) /
                            static_cast<double>(sycl::fmax(last_primary_density_g_per_cm3, 1.0e-6F));
                        const auto cutoff_denominator = static_cast<double>(cutoff_energy_MeV);
                        pending_let_numerator += cutoff_numerator;
                        pending_let_denominator += cutoff_denominator;
                        if (voxel_let_moments_device != nullptr) {
                            pending_voxel_let_numerator += cutoff_numerator;
                            pending_voxel_let_denominator += cutoff_denominator;
                        }
                    }
                    energy_MeV = 0.0F;
                }

                if (pending_primary_depth_MeV > 0.0) {
                    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_dose(dose_device[pending_primary_bin]);
                    atomic_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_depth_MeV));
                }
                if (enable_let_scoring) {
                    flush_letd_moments_device(
                        let_moments_device, number_of_bins,
                        static_cast<std::size_t>(pending_primary_bin), nullptr, 0, 0,
                        pending_let_numerator, pending_let_denominator, true);
                }
                if (enable_voxel_scoring) {
                    if (pending_primary_voxel_MeV > 0.0) {
                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_voxel_dose(voxel_dose_device[pending_primary_voxel]);
                        atomic_voxel_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                    }
                    if (enable_let_scoring && voxel_let_moments_device != nullptr) {
                        flush_letd_moments_device(
                            voxel_let_moments_device, number_of_voxels,
                            pending_primary_voxel, nullptr, 0, 0,
                            pending_voxel_let_numerator, pending_voxel_let_denominator, true);
                    }
                }

                deposited_device[global_history] = history_deposited_MeV;
                escaped_device[global_history] = energy_MeV;
                steps_device[global_history] = steps;
            });
        kernel_event.wait_and_throw();
        primary_kernel_seconds += event_duration_seconds(kernel_event);
    }

    std::vector<DoseAtomicT> dose_device_host(number_of_bins);
    queue.copy(dose_device, dose_device_host.data(), number_of_bins).wait_and_throw();
    std::vector<double> dose_host(number_of_bins);
    std::transform(dose_device_host.begin(), dose_device_host.end(), dose_host.begin(),
                   [](DoseAtomicT val) { return static_cast<double>(val); });

    std::vector<double> voxel_dose_host;
    if (enable_voxel_scoring) {
        std::vector<DoseAtomicT> voxel_dose_device_host(number_of_voxels);
        queue.copy(voxel_dose_device, voxel_dose_device_host.data(), number_of_voxels)
            .wait_and_throw();
        voxel_dose_host.resize(number_of_voxels);
        std::transform(voxel_dose_device_host.begin(), voxel_dose_device_host.end(),
                       voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::vector<LetAtomicT> let_moments_host;
    if (enable_let_scoring) {
        let_moments_host.resize(4 * number_of_bins);
        queue.copy(let_moments_device, let_moments_host.data(), 4 * number_of_bins)
            .wait_and_throw();
    }

    std::vector<LetAtomicT> voxel_let_moments_host;
    if (enable_let_scoring && voxel_let_moments_device != nullptr) {
        voxel_let_moments_host.resize(4 * number_of_voxels);
        queue.copy(voxel_let_moments_device, voxel_let_moments_host.data(), 4 * number_of_voxels)
            .wait_and_throw();
    }

    std::vector<float> deposited_host(number_of_histories);
    std::vector<float> escaped_host(number_of_histories);
    std::vector<std::uint32_t> steps_host(number_of_histories);
    queue.copy(deposited_device, deposited_host.data(), number_of_histories);
    queue.copy(escaped_device, escaped_host.data(), number_of_histories);
    queue.copy(steps_device, steps_host.data(), number_of_histories).wait_and_throw();

    // Free buffers
    free_immutable_device(table_device);
    free_immutable_device(energy_grid_device);
    free_immutable_device(cumulative_range_device);
    free_immutable_device(cross_section_device);
    free_device(dose_device);
    free_device(voxel_dose_device);
    free_device(let_moments_device);
    free_device(voxel_let_moments_device);
    free_device(deposited_device);
    free_device(escaped_device);
    free_device(steps_device);
    free_device(primary_spots_device);
    free_device(slab_z_ends_device);
    free_device(slab_densities_device);
    free_device(slab_radiation_lengths_device);
    free_device(material_sp_device);
    free_device(material_xs_device);
    free_device(insert_sp_device);
    free_device(insert_xs_device);
    free_device(ct_density_device);
    free_device(ct_material_device);
    free_device(ct_mass_sp_factor_lut_device);
    free_device(ct_mass_sp_za_rel_device);
    free_device(ct_sp_device);
    free_device(ct_xs_device);
    free_device(ct_ref_density_device);

    TransportResult result;
    result.backend = "sycl-" + resolved_device_name +
                     (config.enable_energy_straggling ? "+straggling" : "");
    if (config.uses_moment_matched_straggling()) {
        result.backend += "+moment-matched-straggling";
    }
    if (config.enable_step_stable_straggling) {
        result.backend += "+step-stable-primary-straggling";
    }
    const auto batch_has_energy_spread = std::any_of(
        config.primary_spot_batch.begin(), config.primary_spot_batch.end(),
        [](const auto& spot) { return spot.floats[1] > 0.0F; });
    if (config.beam_energy_spread > 0.0 || batch_has_energy_spread) {
        result.backend += "+espread";
    }
    if (!config.primary_spot_batch.empty()) {
        result.backend += "+spot-batch";
    }
    if (config.enable_tps_source) {
        result.backend += "+tps-source";
    }
    if (enable_multiple_scattering) {
        result.backend += "+multiple-scattering";
    }
    if (enable_ct_grid && enable_ct_material_mcs) {
        result.backend += "+ct-material-mcs";
    }
    if (enable_layered_phantom) {
        result.backend += use_material_tables ? "+layered-material" : "+layered-slab";
    }
    if (enable_hetero_insert) {
        result.backend += use_insert_material_tables ? "+hetero-insert-material" : "+hetero-insert";
    }
    if (enable_ct_grid) {
        if (use_ct_mass_sp) {
            result.backend += use_ct_density_mass_spr ? "+ct-grid-density-spr+ct-dda"
                                                     : "+ct-grid-mass-sp-lut+ct-dda";
        } else if (use_ct_material_sp) {
            result.backend += "+ct-grid-material+ct-dda";
        } else {
            result.backend += "+ct-grid+ct-dda";
        }
    }
    if constexpr (k_dose_atomic_fp32) {
        result.backend += "+fp32-dose";
    } else {
        result.backend += "+fp64-dose";
    }
    if (enable_voxel_scoring) {
        result.backend += "+voxel-scoring";
        if (!voxel_scorer_clamps_transport) {
            result.backend += "+scorer-decoupled";
        }
    }
    if (enable_let_scoring) {
        result.backend += "+letd-scoring";
    }

    result.primary_deposited_energy_MeV = dose_host;
    result.deposited_energy_MeV = std::move(dose_host);
    result.voxel_deposited_energy_MeV = std::move(voxel_dose_host);

    if (enable_let_scoring) {
        const auto extract_let_moment = [&](const std::size_t moment) {
            const auto begin = let_moments_host.begin() +
                               static_cast<std::ptrdiff_t>(moment * number_of_bins);
            return std::vector<double>(begin, begin + static_cast<std::ptrdiff_t>(number_of_bins));
        };
        result.primary_letd_numerator = extract_let_moment(0);
        result.primary_letd_denominator = extract_let_moment(1);
        result.all_hadron_letd_numerator = extract_let_moment(2);
        result.all_hadron_letd_denominator = extract_let_moment(3);
    }
    if (enable_let_scoring && voxel_let_moments_device != nullptr) {
        const auto extract_voxel_let_moment = [&](const std::size_t moment) {
            const auto begin = voxel_let_moments_host.begin() +
                               static_cast<std::ptrdiff_t>(moment * number_of_voxels);
            return std::vector<double>(begin, begin + static_cast<std::ptrdiff_t>(number_of_voxels));
        };
        result.primary_voxel_letd_numerator = extract_voxel_let_moment(0);
        result.primary_voxel_letd_denominator = extract_voxel_let_moment(1);
        result.all_hadron_voxel_letd_numerator = extract_voxel_let_moment(2);
        result.all_hadron_voxel_letd_denominator = extract_voxel_let_moment(3);
    }

    if (config.primary_spot_batch.empty()) {
        result.initial_energy_MeV =
            config.initial_total_energy_MeV() * static_cast<double>(number_of_histories);
    } else {
        result.initial_energy_MeV = 0.0;
        for (const auto& spot : config.primary_spot_batch) {
            result.initial_energy_MeV +=
                static_cast<double>(spot.floats[0]) *
                static_cast<double>(spot.history_end - spot.history_begin);
        }
    }
    result.total_deposited_energy_MeV =
        std::accumulate(deposited_host.begin(), deposited_host.end(), 0.0);
    result.escaped_energy_MeV =
        std::accumulate(escaped_host.begin(), escaped_host.end(), 0.0);
    result.total_steps = std::accumulate(steps_host.begin(), steps_host.end(), std::uint64_t{0});

    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.primary_kernel_seconds = primary_kernel_seconds;
    return result;
}

}  // namespace carbon

#endif

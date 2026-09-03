// tools/run_step21_level3_gpu.cpp
#include <sycl/sycl.hpp>
#include "carbon/ct_grid.hpp"
#include "carbon/inelastic_package_v3.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/schneider_stopping_table.hpp"
#include "carbon/schneider_target_sampler.hpp"
#include "carbon/secondary_rate_table.hpp"
#include "carbon/rng.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Direction3F {
    float x{0.0F};
    float y{0.0F};
    float z{1.0F};
};

inline Direction3F rotate_local_direction(const float local_x,
                                          const float local_y,
                                          const float local_z,
                                          const Direction3F parent_direction) noexcept {
    if (!sycl::isfinite(local_x) || !sycl::isfinite(local_y)) {
        return Direction3F{0.0F, 0.0F,
                           sycl::clamp(local_z, -1.0F, 1.0F) *
                               (parent_direction.z < 0.0F ? -1.0F : 1.0F)};
    }
    const auto parent_norm = sycl::sqrt(parent_direction.x * parent_direction.x +
                                        parent_direction.y * parent_direction.y +
                                        parent_direction.z * parent_direction.z);
    const auto inverse_parent_norm = parent_norm > 0.0F ? 1.0F / parent_norm : 1.0F;
    const Direction3F w{parent_direction.x * inverse_parent_norm,
                        parent_direction.y * inverse_parent_norm,
                        parent_norm > 0.0F ? parent_direction.z * inverse_parent_norm : 1.0F};
    const Direction3F reference =
        sycl::fabs(w.x) < 0.9F ? Direction3F{1.0F, 0.0F, 0.0F}
                               : Direction3F{0.0F, 1.0F, 0.0F};
    const auto projection = reference.x * w.x + reference.y * w.y + reference.z * w.z;
    Direction3F u{reference.x - projection * w.x,
                  reference.y - projection * w.y,
                  reference.z - projection * w.z};
    const auto inverse_u_norm =
        1.0F / sycl::sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
    u = Direction3F{u.x * inverse_u_norm, u.y * inverse_u_norm, u.z * inverse_u_norm};
    const Direction3F v{w.y * u.z - w.z * u.y,
                        w.z * u.x - w.x * u.z,
                        w.x * u.y - w.y * u.x};
    Direction3F output{local_x * u.x + local_y * v.x + local_z * w.x,
                       local_x * u.y + local_y * v.y + local_z * w.y,
                       local_x * u.z + local_y * v.z + local_z * w.z};
    const auto norm = sycl::sqrt(output.x * output.x + output.y * output.y + output.z * output.z);
    return norm > 0.0F ? Direction3F{output.x / norm, output.y / norm, output.z / norm} : parent_direction;
}

inline float compute_highland_theta0(float E_mev, int Z, int A, float path_mm, float rho, float X0) {
    if (E_mev <= 0.0F || path_mm <= 0.0F || rho <= 0.0F || X0 <= 0.0F) return 0.0F;
    float E_per_u = E_mev / static_cast<float>(A);
    float E_tot = E_per_u + 931.494F;
    float p_per_u = sycl::sqrt(E_per_u * (E_per_u + 2.0F * 931.494F));
    float beta = p_per_u / E_tot;
    float p_tot = static_cast<float>(A) * p_per_u;
    float rad_len = rho * (path_mm / 10.0F) / X0;
    if (beta <= 0.0F || p_tot <= 0.0F || rad_len <= 0.0F) return 0.0F;
    float charge = static_cast<float>(Z);
    float log_arg = rad_len * charge * charge / (beta * beta);
    float corr = sycl::fmax(0.0F, 1.0F + 0.038F * sycl::log(sycl::fmax(1.0e-12F, log_arg)));
    return 13.6F * charge / (beta * p_tot) * sycl::sqrt(rad_len) * corr;
}

int main(int argc, char* argv[]) {
    try {
        const std::filesystem::path repo_dir = "/mnt/sdb/wuwei/MAIGO";
        const std::filesystem::path base_dir = "/mnt/sda/wuwei/step21_level3";
        const std::filesystem::path cctg_path = repo_dir / "data/schneider/heterogeneous_level3.cctg";
        const std::filesystem::path stopping_path = repo_dir / "data/schneider/schneider_stopping_v1.bin";
        const std::filesystem::path primary_rates_path = repo_dir / "data/schneider/schneider_inelastic_rates_v1.bin";
        const std::filesystem::path c12_cinel_path = repo_dir / "data/schneider/cinel03_c12_targets.bin";
        const std::filesystem::path sec_rates_path = repo_dir / "data/schneider/secondary_inelastic_rates_v1.bin";
        const std::filesystem::path sec_cinel_path = repo_dir / "data/schneider/cinel03_secondary_targets.bin";

        std::cout << "Loading CT Grid, Stopping Power, Samplers and Packages...\n";
        const auto ct_grid = carbon::CtGrid::load(cctg_path.string(), "", "");
        const auto stopping_table = carbon::SchneiderStoppingTable::from_binary(stopping_path);
        const auto flat_stopping = stopping_table.to_flat_mass_stopping_float();

        const auto rate_table = carbon::SchneiderRateTable::from_binary(primary_rates_path);
        const carbon::SchneiderTargetSampler target_sampler(rate_table);

        const auto c12_package = carbon::InelasticPackageV3Table::from_binary(c12_cinel_path);
        const auto c12_dev_tables = c12_package.make_device_tables();

        const auto sec_package = carbon::InelasticPackageV3Table::from_binary(sec_cinel_path);
        const auto sec_dev_tables = sec_package.make_device_tables();

        const auto sec_rate_table = carbon::SecondaryRateTable::from_binary(sec_rates_path);

        sycl::queue queue{sycl::default_selector_v, sycl::property::queue::in_order{}};
        std::cout << "Running on SYCL Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

        // Upload grid densities and materials
        const size_t num_voxels = ct_grid.nx * ct_grid.ny * ct_grid.nz;
        auto* dev_densities = sycl::malloc_device<float>(num_voxels, queue);
        auto* dev_materials = sycl::malloc_device<uint8_t>(num_voxels, queue);
        queue.copy(ct_grid.density_g_per_cm3.data(), dev_densities, num_voxels);
        queue.copy(ct_grid.material_id.data(), dev_materials, num_voxels);

        // Upload Stopping Table
        const size_t stopping_size = flat_stopping.size();
        auto* dev_stopping = sycl::malloc_device<float>(stopping_size, queue);
        queue.copy(flat_stopping.data(), dev_stopping, stopping_size);

        // Upload Radiation Lengths for Schneider sections
        std::vector<float> rad_lengths(25);
        for (unsigned s = 0; s < 25; ++s) {
            rad_lengths[s] = static_cast<float>(carbon::schneider_section_radiation_length_g_per_cm2(s));
        }
        auto* dev_rad_lengths = sycl::malloc_device<float>(25, queue);
        queue.copy(rad_lengths.data(), dev_rad_lengths, 25);

        // Upload Sampler CDF and Total Rates
        const auto sampler_dev = target_sampler.device_table();
        float* dev_cdf = sycl::malloc_device<float>(target_sampler.cdf_table().size(), queue);
        float* dev_total_rates = sycl::malloc_device<float>(target_sampler.total_mass_rates().size(), queue);
        queue.copy(target_sampler.cdf_table().data(), dev_cdf, target_sampler.cdf_table().size()).wait_and_throw();
        queue.copy(target_sampler.total_mass_rates().data(), dev_total_rates, target_sampler.total_mass_rates().size()).wait_and_throw();

        carbon::SchneiderTargetSamplerDeviceTable gpu_sampler = sampler_dev;
        gpu_sampler.cdf_table = dev_cdf;
        gpu_sampler.total_mass_rates = dev_total_rates;

        // Upload Primary CINEL03 Package
        carbon::Cinel03EnergyNode* dev_c12_nodes = sycl::malloc_device<carbon::Cinel03EnergyNode>(c12_dev_tables.energy_nodes.size(), queue);
        std::uint32_t* dev_c12_offsets = sycl::malloc_device<std::uint32_t>(c12_dev_tables.event_offsets.size(), queue);
        std::uint32_t* dev_c12_indices = sycl::malloc_device<std::uint32_t>(c12_dev_tables.event_indices.size(), queue);
        carbon::Cinel03DeviceInteraction* dev_c12_ints = sycl::malloc_device<carbon::Cinel03DeviceInteraction>(c12_dev_tables.interactions.size(), queue);
        carbon::Cinel03DeviceProduct* dev_c12_prods = sycl::malloc_device<carbon::Cinel03DeviceProduct>(c12_dev_tables.products.size(), queue);

        queue.copy(c12_dev_tables.energy_nodes.data(), dev_c12_nodes, c12_dev_tables.energy_nodes.size()).wait_and_throw();
        queue.copy(c12_dev_tables.event_offsets.data(), dev_c12_offsets, c12_dev_tables.event_offsets.size()).wait_and_throw();
        queue.copy(c12_dev_tables.event_indices.data(), dev_c12_indices, c12_dev_tables.event_indices.size()).wait_and_throw();
        queue.copy(c12_dev_tables.interactions.data(), dev_c12_ints, c12_dev_tables.interactions.size()).wait_and_throw();
        queue.copy(c12_dev_tables.products.data(), dev_c12_prods, c12_dev_tables.products.size()).wait_and_throw();

        const std::uint32_t c12_node_count = static_cast<std::uint32_t>(c12_dev_tables.energy_nodes.size());
        const std::uint32_t c12_total_events = static_cast<std::uint32_t>(c12_dev_tables.interactions.size());

        // Device output dose buffer
        auto* dev_voxel_edep = sycl::malloc_device<float>(num_voxels, queue);

        const uint32_t nx = ct_grid.nx, ny = ct_grid.ny, nz = ct_grid.nz;
        const float dx = ct_grid.spacing_x_mm, dy = ct_grid.spacing_y_mm, dz = ct_grid.spacing_z_mm;
        const float ox = ct_grid.origin_x_mm, oy = ct_grid.origin_y_mm, oz = ct_grid.origin_z_mm;

        struct CaseDef {
            std::string id;
            double energy_mevu;
            double rot_x_deg;
            double trans_y;
            uint64_t histories;
        };

        const std::vector<CaseDef> cases = {
            {"level3_axis_aligned", 220.0, 0.0, 0.0, 50000},
            {"level3_oblique", 260.0, 15.0, -10.0, 50000}
        };

        for (const auto& c : cases) {
            std::cout << "\n=======================================================\n";
            std::cout << "Simulating on GPU: " << c.id << " (" << c.energy_mevu << " MeV/u, " << c.rot_x_deg << " deg)\n";
            std::cout << "=======================================================\n";

            queue.memset(dev_voxel_edep, 0, num_voxels * sizeof(float)).wait_and_throw();

            const uint64_t N = c.histories;
            const float e_init = static_cast<float>(c.energy_mevu);
            const double rad = c.rot_x_deg * M_PI / 180.0;
            const float u_x0 = 0.0F;
            const float u_y0 = static_cast<float>(std::sin(rad));
            const float u_z0 = static_cast<float>(std::cos(rad));
            const float source_z = -50.0F;
            const float t_entry = (0.0F - source_z) / u_z0;
            const float start_x = 0.0F + t_entry * u_x0;
            const float start_y = static_cast<float>(c.trans_y) + t_entry * u_y0;
            const float start_z = 0.0F;

            queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                const uint64_t hist = idx[0];
                const uint64_t seed = 20260903ULL;

                // Primary carbon entry state
                float px = start_x, py = start_y, pz = start_z;
                Direction3F u_dir{u_x0, u_y0, u_z0};
                float E = e_init; // MeV/u

                // Sample primary optical depth
                float u_tau = carbon::rng::uniform01(seed, hist, 0, 1);
                float tau = -sycl::log(sycl::fmax(1.0e-12F, u_tau));

                bool alive = true;
                int step_count = 0;

                while (alive && E > 0.5F && step_count < 1000) {
                    step_count++;
                    int ix = static_cast<int>(sycl::floor((px - ox) / dx));
                    int iy = static_cast<int>(sycl::floor((py - oy) / dy));
                    int iz = static_cast<int>(sycl::floor((pz - oz) / dz));

                    if (ix < 0 || ix >= static_cast<int>(nx) ||
                        iy < 0 || iy >= static_cast<int>(ny) ||
                        iz < 0 || iz >= static_cast<int>(nz)) {
                        break; // Escaped CT volume
                    }

                    const size_t v_idx = static_cast<size_t>(iz) * (nx * ny) + static_cast<size_t>(iy) * nx + ix;
                    const uint8_t sec_id = dev_materials[v_idx];
                    const float rho = dev_densities[v_idx];
                    const float X0 = dev_rad_lengths[sec_id < 25 ? sec_id : 24];

                    // Distance to next voxel boundary along direction
                    float d_x = (u_dir.x > 1.0e-6F) ? (ox + (ix + 1) * dx - px) / u_dir.x :
                                ((u_dir.x < -1.0e-6F) ? (ox + ix * dx - px) / u_dir.x : 1.0e9F);
                    float d_y = (u_dir.y > 1.0e-6F) ? (oy + (iy + 1) * dy - py) / u_dir.y :
                                ((u_dir.y < -1.0e-6F) ? (oy + iy * dy - py) / u_dir.y : 1.0e9F);
                    float d_z = (u_dir.z > 1.0e-6F) ? (oz + (iz + 1) * dz - pz) / u_dir.z :
                                ((u_dir.z < -1.0e-6F) ? (oz + iz * dz - pz) / u_dir.z : 1.0e9F);
                    float d_face = sycl::fmin(d_x, sycl::fmin(d_y, d_z));
                    if (d_face <= 0.0F) d_face = 1.0e-4F;

                    // Continuous stopping power (MeV/u per mm)
                    float s_idx_f = (E - 0.01F) * 10.0F;
                    int s_idx = sycl::clamp(static_cast<int>(s_idx_f), 0, 4300);
                    float mass_sp = dev_stopping[sec_id * 4302 + s_idx];
                    float dedx = mass_sp * rho; // MeV/mm (total for C12)

                    // Nuclear inelastic rate (1/mm)
                    float r_idx_f = (E - gpu_sampler.energy_min_MeV_per_u) * gpu_sampler.inverse_energy_step;
                    int r_idx = sycl::clamp(static_cast<int>(r_idx_f), 0, static_cast<int>(gpu_sampler.num_energies - 1));
                    float mass_rate = gpu_sampler.total_mass_rates[sec_id * gpu_sampler.num_energies + r_idx];
                    float macro_rate = mass_rate * rho; // 1/mm

                    float d_coll = (macro_rate > 1.0e-12F) ? (tau / macro_rate) : 1.0e9F;
                    float d_stop = (dedx > 1.0e-6F) ? ((E - 0.5F) * 12.0F / dedx) : 1.0e9F;

                    float ds = sycl::fmin(d_face + 1.0e-4F, sycl::fmin(d_coll, d_stop));
                    if (ds > 1.0F) ds = 1.0F; // 1mm max step clamp

                    float dE_total = sycl::fmin(E * 12.0F, dedx * ds);
                    E -= (dE_total / 12.0F);

                    // Deposit energy into current voxel
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        v_edep(dev_voxel_edep[v_idx]);
                    v_edep.fetch_add(dE_total);

                    // Advance position
                    px += ds * u_dir.x;
                    py += ds * u_dir.y;
                    pz += ds * u_dir.z;
                    tau -= ds * macro_rate;

                    // Highland MCS deflection
                    float theta0 = compute_highland_theta0(E * 12.0F, 6, 12, ds, rho, X0);
                    if (theta0 > 0.0F) {
                        float u_mcs0 = sycl::fmax(1.0e-12F, carbon::rng::uniform01(seed, hist, step_count, 4));
                        float u_mcs1 = carbon::rng::uniform01(seed, hist, step_count, 5);
                        float rad_m = theta0 * sycl::sqrt(-2.0F * sycl::log(u_mcs0));
                        float phi_m = 6.2831853F * u_mcs1;
                        float lx = rad_m * sycl::cos(phi_m);
                        float ly = rad_m * sycl::sin(phi_m);
                        float lz = sycl::sqrt(sycl::fmax(0.0F, 1.0F - lx*lx - ly*ly));
                        u_dir = rotate_local_direction(lx, ly, lz, u_dir);
                    }

                    if (tau <= 0.0F) {
                        // Primary Carbon-12 nuclear breakup!
                        alive = false;
                        float u_tgt = carbon::rng::uniform01(seed, hist, step_count, 2);
                        int target_z = carbon::sample_schneider_target_device(gpu_sampler, sec_id, E, u_tgt);

                        float u_evt = carbon::rng::uniform01(seed, hist, step_count, 3);
                        uint32_t event_id = carbon::cinel03_find_event_device(
                            dev_c12_nodes, c12_node_count, dev_c12_offsets, dev_c12_indices, c12_total_events,
                            6, 12, target_z, E, 15.0F, u_evt);

                        if (event_id != 0xFFFFFFFFU) {
                            const auto event = dev_c12_ints[event_id];
                            const uint32_t prod_offset = event.product_offset;
                            const uint32_t prod_count = event.direct_product_count;

                            // Loop over secondary products
                            for (uint32_t p = 0; p < prod_count; ++p) {
                                const auto prod = dev_c12_prods[prod_offset + p];
                                const int pz_ion = prod.z;
                                const int pa_ion = prod.a;
                                float e_sec = prod.kinetic_energy_MeV;

                                if (pz_ion <= 0 || pa_ion <= 0 || e_sec <= 0.5F) {
                                    // Neutral or below cutoff: deposit locally in current voxel
                                    v_edep.fetch_add(e_sec);
                                    continue;
                                }

                                // Secondary charged ion transport with MCS
                                float spx = px, spy = py, spz = pz;
                                Direction3F su_dir = rotate_local_direction(
                                    prod.local_direction_x, prod.local_direction_y, prod.local_direction_z, u_dir);

                                float z2_scale = static_cast<float>(pz_ion * pz_ion) / 36.0F;

                                // Track secondary through voxels until stopped or escaped
                                for (int sstep = 0; sstep < 300 && e_sec > 0.5F; ++sstep) {
                                    int six = static_cast<int>(sycl::floor((spx - ox) / dx));
                                    int siy = static_cast<int>(sycl::floor((spy - oy) / dy));
                                    int siz = static_cast<int>(sycl::floor((spz - oz) / dz));

                                    if (six < 0 || six >= static_cast<int>(nx) ||
                                        siy < 0 || siy >= static_cast<int>(ny) ||
                                        siz < 0 || siz >= static_cast<int>(nz)) {
                                        break;
                                    }

                                    const size_t sv_idx = static_cast<size_t>(siz) * (nx * ny) + static_cast<size_t>(siy) * nx + six;
                                    const uint8_t ssec_id = dev_materials[sv_idx];
                                    const float srho = dev_densities[sv_idx];
                                    const float sX0 = dev_rad_lengths[ssec_id < 25 ? ssec_id : 24];

                                    float sd_x = (su_dir.x > 1.0e-6F) ? (ox + (six + 1) * dx - spx) / su_dir.x :
                                                 ((su_dir.x < -1.0e-6F) ? (ox + six * dx - spx) / su_dir.x : 1.0e9F);
                                    float sd_y = (su_dir.y > 1.0e-6F) ? (oy + (siy + 1) * dy - spy) / su_dir.y :
                                                 ((su_dir.y < -1.0e-6F) ? (oy + siy * dy - spy) / su_dir.y : 1.0e9F);
                                    float sd_z = (su_dir.z > 1.0e-6F) ? (oz + (siz + 1) * dz - spz) / su_dir.z :
                                                 ((su_dir.z < -1.0e-6F) ? (oz + siz * dz - spz) / su_dir.z : 1.0e9F);
                                    float sd_face = sycl::fmin(sd_x, sycl::fmin(sd_y, sd_z));
                                    if (sd_face <= 0.0F) sd_face = 1.0e-4F;

                                    float e_per_u = e_sec / static_cast<float>(pa_ion);
                                    float s_idx_f2 = (e_per_u - 0.01F) * 10.0F;
                                    int s_idx2 = sycl::clamp(static_cast<int>(s_idx_f2), 0, 4300);
                                    float sp_c12 = dev_stopping[ssec_id * 4302 + s_idx2];
                                    float s_dedx = sp_c12 * z2_scale * srho; // MeV/mm

                                    float sds = sycl::fmin(sd_face + 1.0e-4F, 2.0F);
                                    float s_dE = sycl::fmin(e_sec, s_dedx * sds);
                                    e_sec -= s_dE;

                                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        sv_edep(dev_voxel_edep[sv_idx]);
                                    sv_edep.fetch_add(s_dE);

                                    spx += sds * su_dir.x;
                                    spy += sds * su_dir.y;
                                    spz += sds * su_dir.z;

                                    // Secondary MCS
                                    float s_theta0 = compute_highland_theta0(e_sec, pz_ion, pa_ion, sds, srho, sX0);
                                    if (s_theta0 > 0.0F) {
                                        float su_mcs0 = sycl::fmax(1.0e-12F, carbon::rng::uniform01(seed, hist, sstep + 200, 6));
                                        float su_mcs1 = carbon::rng::uniform01(seed, hist, sstep + 200, 7);
                                        float srad_m = s_theta0 * sycl::sqrt(-2.0F * sycl::log(su_mcs0));
                                        float sphi_m = 6.2831853F * su_mcs1;
                                        float slx = srad_m * sycl::cos(sphi_m);
                                        float sly = srad_m * sycl::sin(sphi_m);
                                        float slz = sycl::sqrt(sycl::fmax(0.0F, 1.0F - slx*slx - sly*sly));
                                        su_dir = rotate_local_direction(slx, sly, slz, su_dir);
                                    }
                                }
                            }
                        }
                    }
                }
            }).wait_and_throw();

            // Download deposited energy
            std::vector<float> host_edep(num_voxels, 0.0F);
            queue.copy(dev_voxel_edep, host_edep.data(), num_voxels).wait_and_throw();

            const double voxel_vol_cm3 = (dx * 0.1) * (dy * 0.1) * (dz * 0.1); // in cm3
            const auto gpu_dose_csv = base_dir / "gpu" / (c.id + "_dose.csv");
            std::ofstream out(gpu_dose_csv);
            out << "# MAIGO GPU Simulation: " << c.id << "\n";
            out << "# Histories: " << c.histories << "\n";
            out << "# X in 40 bins of 0.2 cm\n";
            out << "# Y in 40 bins of 0.2 cm\n";
            out << "# Z in 75 bins of 0.2 cm\n";
            out << "# DoseToMedium ( Gy ) : Sum\n";

            double total_dose_integral_Gy = 0.0;

            for (size_t iz = 0; iz < nz; ++iz) {
                for (size_t iy = 0; iy < ny; ++iy) {
                    for (size_t ix = 0; ix < nx; ++ix) {
                        const size_t idx = iz * (nx * ny) + iy * nx + ix;
                        const double edep_mev = host_edep[idx];
                        const double rho = ct_grid.density_g_per_cm3[idx];
                        const double mass_kg = rho * voxel_vol_cm3 * 1.0e-3;
                        const double dose_gy = (mass_kg > 0.0) 
                            ? (edep_mev * 1.602176634e-13 / mass_kg) : 0.0;

                        total_dose_integral_Gy += dose_gy;
                        out << ix << ", " << iy << ", " << iz << ", " << std::setprecision(8) << dose_gy << "\n";
                    }
                }
            }
            out.close();
            std::cout << "Successfully wrote: " << gpu_dose_csv << "\n";
            std::cout << "  Total Dose Integral: " << total_dose_integral_Gy << " Gy\n";
        }

        sycl::free(dev_densities, queue);
        sycl::free(dev_materials, queue);
        sycl::free(dev_stopping, queue);
        sycl::free(dev_rad_lengths, queue);
        sycl::free(dev_cdf, queue);
        sycl::free(dev_total_rates, queue);
        sycl::free(dev_c12_nodes, queue);
        sycl::free(dev_c12_offsets, queue);
        sycl::free(dev_c12_indices, queue);
        sycl::free(dev_c12_ints, queue);
        sycl::free(dev_c12_prods, queue);
        sycl::free(dev_voxel_edep, queue);

        std::cout << "\nAll Level 3 GPU simulations finished successfully!\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error in run_step21_level3_gpu: " << ex.what() << std::endl;
        return 1;
    }
}

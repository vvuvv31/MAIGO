// Literature Data-Driven Fragmentation Model (FRED Section 2.1.3 & PMC8990885)
// Exact 18-Isotope Inclusive Channel Generation, 2D PDF Sampling, Resampling, and Independent RNG.

#include "carbon/detail/fred_fragmentation_data.hpp"

inline InelasticProductSet sample_carbon_inelastic_products_device(
    const float incident_energy_MeV,
    const float pos_x_mm,
    const float pos_y_mm,
    const float pos_z_mm,
    const Direction3F parent_dir,
    const uint32_t spot_seed,
    const uint32_t rng_history,
    const uint32_t steps,
    const float* prob_proj_h,
    const float* prob_proj_o,
    const float* prob_tgt_h,
    const float* prob_tgt_o,
    const float* ion_sp_lut,
    const int ion_sp_table_size,
    const float ion_sp_min_energy,
    const float ion_sp_inv_step,
    const float energy_cutoff_MeV) noexcept {

    InelasticProductSet set{};
    const auto e_per_u = incident_energy_MeV / 12.0F;
    if (e_per_u <= 0.5F) {
        set.local_deposit_MeV = incident_energy_MeV;
        return set;
    }

    constexpr float two_pi = 6.2831853071795864769F;
    constexpr float deg_to_rad = 0.017453292519943295F;
    constexpr uint32_t kInelasticBaseDim = 10000;

    // 1. Target Selection in water (Eq. 10-11)
    const float sigma_H = (e_per_u >= 250.0F)
                              ? 250.0F
                              : (250.0F * (1.0F + 0.80F * sycl::exp(-e_per_u / 60.0F)));

    // sigma_O(E) from Kox reaction cross-section formula for 12C + 16O (Kox et al., Phys. Rev. C 35, 1678)
    constexpr float a_kox = 1.85F;
    constexpr float ap_c12 = 12.0F;
    constexpr float at_o16 = 16.0F;
    constexpr float zp_c12 = 6.0F;
    constexpr float zt_o16 = 8.0F;
    constexpr float ap13 = 2.289428F; // 12^(1/3)
    constexpr float at13 = 2.519842F; // 16^(1/3)
    const float c_e = 0.80F * (1.0F - sycl::exp(-e_per_u / 40.0F));
    const float r_vol = ap13 + at13 + a_kox * (ap13 * at13) / (ap13 + at13) - c_e;
    const float b_c = (1.44F * zp_c12 * zt_o16) / (1.3F * (ap13 + at13));
    const float e_cm = (at_o16 / (ap_c12 + at_o16)) * incident_energy_MeV;
    const float coulomb_barrier_factor = (e_cm > b_c) ? (1.0F - b_c / e_cm) : 0.0F;
    const float sigma_O = 38.013F * r_vol * r_vol * coulomb_barrier_factor;

    const float p_target_H = (2.0F * sigma_H) / (2.0F * sigma_H + sycl::fmax(sigma_O, 1.0F));
    const float u_target = rng::uniform01(spot_seed, rng_history, steps, kInelasticBaseDim + 0);
    const uint8_t target_idx = (u_target < p_target_H) ? 0 : 2; // 0 for H, 2 for O
    const float* proj_prob = (target_idx == 0) ? prob_proj_h : prob_proj_o;
    const float* tgt_prob = (target_idx == 0) ? prob_tgt_h : prob_tgt_o;

    // 2. Binding energy is unavailable kinetic energy (untracked nuclear), not local dose.
    const auto base_q_MeV = 12.0F + 0.04F * e_per_u;
    const auto max_available_MeV = sycl::fmax(0.0F, incident_energy_MeV - base_q_MeV);

    // 3. Sample projectile and target fragment sets with resampling (Section 2.1.3)
    constexpr int max_retries = 32;
    bool accepted = false;

    for (int retry = 0; retry < max_retries; ++retry) {
        InelasticProductSet candidate{};
        candidate.untracked_energy_MeV = base_q_MeV;

        int a_rem = 12;
        int z_rem = 6;
        uint8_t frag_species[12]{};
        bool is_target_frag[12]{};
        uint8_t num_frags = 0;

        // Projectile fragments: sequential Table 1 sampling over remaining (A,Z), including 12C.
        int step_cnt = 0;
        while ((a_rem > 0 || z_rem > 0) &&
               num_frags < carbon::kMaxProjectileFragments && step_cnt < 16) {
            ++step_cnt;
            const float u_frag = rng::uniform01(
                spot_seed, rng_history, steps,
                kInelasticBaseDim + static_cast<uint32_t>(retry * 200 + step_cnt));

            const int iso_idx = carbon::sample_table1_isotope(proj_prob, a_rem, z_rem, u_frag);
            const auto iso = carbon::kFredIsotopes[iso_idx];
            if (iso.a <= a_rem && iso.z <= z_rem && iso.a > 0) {
                frag_species[num_frags] = static_cast<uint8_t>(iso_idx);
                is_target_frag[num_frags] = false;
                ++num_frags;
                a_rem -= iso.a;
                z_rem -= iso.z;
            } else if (z_rem > 0 && a_rem > 0) {
                frag_species[num_frags] = 1;
                is_target_frag[num_frags] = false;
                ++num_frags;
                a_rem -= 1;
                z_rem -= 1;
            } else if (a_rem > 0) {
                frag_species[num_frags] = 0;
                is_target_frag[num_frags] = false;
                ++num_frags;
                a_rem -= 1;
            } else {
                break;
            }
        }

        if (a_rem != 0 || z_rem != 0) {
            continue;
        }

        // Target fragments from Table 1 remaining (A,Z). Light charged (p,d,t,He) are transported;
        // neutrons are untracked; leftover heavy residue is locally deposited.
        int target_a_rem = (target_idx == 0) ? 1 : 16;
        int target_z_rem = (target_idx == 0) ? 1 : 8;
        int heavy_tgt_a = 0;
        int heavy_tgt_z = 0;
        int heavy_tgt_idx = -1;
        int tgt_step = 0;
        while ((target_a_rem > 0 || target_z_rem > 0) &&
               num_frags < carbon::kMaxInelasticProducts && tgt_step < 12) {
            ++tgt_step;
            const float u_tgt = rng::uniform01(
                spot_seed, rng_history, steps,
                kInelasticBaseDim + static_cast<uint32_t>(retry * 200 + 40 + tgt_step));
            const int iso_idx = carbon::sample_table1_isotope(tgt_prob, target_a_rem, target_z_rem, u_tgt);
            const auto iso = carbon::kFredIsotopes[iso_idx];
            if (iso.a <= 0 || iso.a > target_a_rem || iso.z > target_z_rem) {
                if (target_z_rem > 0 && target_a_rem > 0) {
                    frag_species[num_frags] = 1;
                    is_target_frag[num_frags] = true;
                    ++num_frags;
                    target_a_rem -= 1;
                    target_z_rem -= 1;
                } else if (target_a_rem > 0) {
                    frag_species[num_frags] = 0;
                    is_target_frag[num_frags] = true;
                    ++num_frags;
                    target_a_rem -= 1;
                } else {
                    break;
                }
                continue;
            }
            if (iso.z >= 3 || iso.a >= 6) {
                target_a_rem -= iso.a;
                target_z_rem -= iso.z;
                heavy_tgt_a = iso.a;
                heavy_tgt_z = iso.z;
                heavy_tgt_idx = iso_idx;
                break;
            }
            frag_species[num_frags] = static_cast<uint8_t>(iso_idx);
            is_target_frag[num_frags] = true;
            ++num_frags;
            target_a_rem -= iso.a;
            target_z_rem -= iso.z;
        }

        if (heavy_tgt_idx >= 0 && num_frags < carbon::kMaxInelasticProducts) {
            frag_species[num_frags] = static_cast<uint8_t>(heavy_tgt_idx);
            is_target_frag[num_frags] = true;
            ++num_frags;
        }

        // Sample Kinematics for all fragments from Eq. (12) 2D PDF
        float running_e_sum = 0.0F;
        float running_a_sum = 0.0F;
        uint8_t charged_count = 0;


        for (uint8_t i = 0; i < num_frags; ++i) {
            const auto sp_idx = frag_species[i];
            const auto iso = carbon::kFredIsotopes[sp_idx];
            const auto params = (target_idx == 0) ? carbon::kFredParamsH[sp_idx] : carbon::kFredParamsO[sp_idx];
            const auto af = static_cast<float>(iso.a);
            const bool is_tgt = is_target_frag[i];

            // Independent RNG streams per fragment: retry * 200 + 50 + i * 10 + [0..5]
            const uint32_t base_frag_rng = kInelasticBaseDim + static_cast<uint32_t>(retry * 200 + 50 + i * 10);
            const float u_mix = rng::uniform01(spot_seed, rng_history, steps, base_frag_rng + 0);
            const float u_e0  = rng::uniform01(spot_seed, rng_history, steps, base_frag_rng + 1);
            const float u_e1  = rng::uniform01(spot_seed, rng_history, steps, base_frag_rng + 2);
            const float u_th0 = rng::uniform01(spot_seed, rng_history, steps, base_frag_rng + 3);
            const float u_th1 = rng::uniform01(spot_seed, rng_history, steps, base_frag_rng + 4);
            const float u_ph  = rng::uniform01(spot_seed, rng_history, steps, base_frag_rng + 5);

            // Analytical mixture integration weights for Eq. (12)
            const float i_gauss = 3.14159F * params.a2 * params.sigma_e * params.sigma_th;
            const float a_e = sycl::fmax(params.a_e, 0.005F);
            const float a_th = sycl::fmax(params.a_th, 0.01F);
            const float i_exp = params.a1 / (a_e * a_th);
            const float p_gauss = i_gauss / sycl::fmax(1.0e-6F, i_gauss + i_exp);

            // Eq. (12) mixture for projectile fragments; target fragments are exponential.
            const bool sample_gauss = !is_tgt && (u_mix < p_gauss);

            float frag_e_u = 0.0F;
            float theta_rad = 0.0F;

            if (sample_gauss) {
                // Box-Muller Gaussian for E95
                const auto u_log = sycl::fmax(u_e0, 1.0e-10F);
                const auto gauss_e = sycl::sqrt(-2.0F * sycl::log(u_log)) * sycl::cos(two_pi * u_e1);
                const auto e95 = sycl::clamp(params.mean_e + params.sigma_e * gauss_e, 1.0F, 180.0F);

                // Energy scaling Eq. (13)-(16)
                const auto r_ratio = (running_a_sum > 0.0F && e_per_u > 0.0F)
                                         ? (running_e_sum / running_a_sum) / e_per_u
                                         : 0.0F;
                constexpr float c_corr = 0.40F;
                const auto k_corr = c_corr * (1.0F - sycl::clamp(r_ratio, 0.0F, 1.0F));
                frag_e_u = e95 * (e_per_u / 95.0F) * (1.0F - k_corr);

                // Half-Normal Gaussian for theta95
                const auto u_th_log = sycl::fmax(u_th0, 1.0e-10F);
                const auto gauss_th = sycl::sqrt(-2.0F * sycl::log(u_th_log)) * sycl::fabs(sycl::cos(two_pi * u_th1));
                const auto theta95_deg = params.sigma_th * gauss_th;
                const auto theta95_rad = theta95_deg * deg_to_rad;

                const bool skip_angle_scale = (iso.z == 0) || (iso.z == 1 && iso.a == 1);
                if (skip_angle_scale) {
                    theta_rad = theta95_rad;
                } else {
                    theta_rad = theta95_rad * sycl::sqrt(95.0F / e_per_u);
                }
            } else {
                // Exponential (target-like) energy: Eq. (13) scale, no event-level k.
                const auto u_e_log = sycl::fmax(u_e0, 1.0e-8F);
                const auto e95 = sycl::fmin(35.0F, -sycl::log(u_e_log) / a_e);
                frag_e_u = e95 * (e_per_u / 95.0F);

                const auto u_th_log = sycl::fmax(u_th0, 1.0e-8F);
                const auto theta_deg = -sycl::log(u_th_log) / a_th;
                const bool skip_angle_scale = (iso.z == 0) || (iso.z == 1 && iso.a == 1);
                if (skip_angle_scale) {
                    theta_rad = sycl::fmin(1.57F, theta_deg * deg_to_rad);
                } else {
                    theta_rad = sycl::fmin(1.57F, theta_deg * deg_to_rad * sycl::sqrt(95.0F / e_per_u));
                }
            }

            frag_e_u = sycl::fmax(0.0F, frag_e_u);
            const float frag_e_tot = af * frag_e_u;
            if (!is_tgt) {
                running_e_sum += frag_e_tot;
                running_a_sum += af;
            }

            const auto phi = two_pi * u_ph;
            const auto sin_th = sycl::sin(theta_rad);
            const auto cos_th = sycl::cos(theta_rad);

            const auto local_x = sin_th * sycl::cos(phi);
            const auto local_y = sin_th * sycl::sin(phi);
            const auto local_z = cos_th;

            const auto rot_dir = rotate_local_direction(local_x, local_y, local_z, parent_dir);

            if (iso.z == 0) {
                candidate.untracked_energy_MeV += frag_e_tot;
            } else {
                const bool heavy_tgt = is_tgt && (iso.z >= 3 || iso.a >= 6);
                float range_mm = 1.0e3F;
                if (heavy_tgt) {
                    float sp_est = sycl::fmax(0.80F * static_cast<float>(iso.z * iso.z), 0.20F);
                    if (ion_sp_lut != nullptr && ion_sp_table_size > 1 && af > 0.0F) {
                        const int sp_species = carbon::get_charged_species_idx(iso.z, iso.a);
                        const float e_u = frag_e_tot / af;
                        const float flt_idx = (e_u - ion_sp_min_energy) * ion_sp_inv_step;
                        auto sp_idx = static_cast<int>(sycl::floor(flt_idx));
                        sp_idx = sycl::max(0, sycl::min(sp_idx, ion_sp_table_size - 2));
                        const float sp_fr =
                            sycl::clamp(flt_idx - static_cast<float>(sp_idx), 0.0F, 1.0F);
                        const float* row = ion_sp_lut + static_cast<int>(sp_species) * ion_sp_table_size;
                        sp_est = sycl::fmax(row[sp_idx] + sp_fr * (row[sp_idx + 1] - row[sp_idx]), 0.01F);
                    }
                    range_mm = frag_e_tot / sp_est;
                }
                const bool local_stop =
                    heavy_tgt && (range_mm < 0.05F || frag_e_tot < energy_cutoff_MeV);
                if (local_stop) {
                    candidate.local_deposit_MeV += frag_e_tot;
                    candidate.remnant_local_MeV += frag_e_tot;
                } else if (charged_count < carbon::kMaxInelasticProducts) {
                    auto& p = candidate.products[charged_count++];
                    p.z = iso.z;
                    p.a = iso.a;
                    p.energy_MeV = frag_e_tot;
                    p.pos_x_mm = pos_x_mm;
                    p.pos_y_mm = pos_y_mm;
                    p.pos_z_mm = pos_z_mm;
                    p.dir_x = rot_dir.x;
                    p.dir_y = rot_dir.y;
                    p.dir_z = rot_dir.z;
                    p.weight = 1.0F;
                    p.parent_history = rng_history;
                } else {
                    candidate.product_capacity_overflow = 1;
                    candidate.product_capacity_overflow_MeV += frag_e_tot;
                }
            }
        }

        // Leftover target nucleons not identified as a Table-1 isotope: low-energy local only.
        const int leftover_a = sycl::max(target_a_rem, 0);
        if (leftover_a > 0) {
            const uint32_t recoil_rng = kInelasticBaseDim + static_cast<uint32_t>(retry * 200 + 180);
            const float u_rec = sycl::fmax(
                rng::uniform01(spot_seed, rng_history, steps, recoil_rng), 1.0e-8F);
            const float leftover_e = sycl::fmin(8.0F, -sycl::log(u_rec) / 0.40F);
            candidate.local_deposit_MeV += leftover_e;
            candidate.remnant_local_MeV += leftover_e;
        }
        const int recoil_a = heavy_tgt_a + leftover_a;

        candidate.count = charged_count;

        float total_charged_MeV = 0.0F;
        for (uint8_t i = 0; i < candidate.count; ++i) {
            total_charged_MeV += candidate.products[i].energy_MeV;
        }
        const float neutron_ke_MeV = sycl::fmax(0.0F, candidate.untracked_energy_MeV - base_q_MeV);
        const float kinetic_used_MeV =
            total_charged_MeV + candidate.local_deposit_MeV + neutron_ke_MeV +
            candidate.product_capacity_overflow_MeV;
        candidate.untracked_energy_MeV += candidate.product_capacity_overflow_MeV;

        candidate.q_MeV = base_q_MeV;
        candidate.neutron_ke_MeV = neutron_ke_MeV;
        candidate.retries_used = static_cast<uint8_t>(retry + 1);
        candidate.leftover_proj_a = static_cast<int8_t>(a_rem);
        candidate.leftover_proj_z = static_cast<int8_t>(z_rem);
        candidate.leftover_tgt_a = static_cast<int8_t>(recoil_a);
        candidate.leftover_tgt_z = static_cast<int8_t>(heavy_tgt_z + sycl::max(target_z_rem, 0));
        candidate.n_emitted = num_frags;
        for (uint8_t s = 0; s < num_frags && s < carbon::kMaxInelasticProducts; ++s) {
            candidate.emitted_idx[s] = frag_species[s];
        }

        if (kinetic_used_MeV <= max_available_MeV) {
            set = candidate;
            accepted = true;
            break;
        }
    }

    if (!accepted) {
        set.count = 0;
        set.local_deposit_MeV = 0.0F;
        set.untracked_energy_MeV = incident_energy_MeV;
        set.resample_failed = 1;
        set.retries_used = static_cast<uint8_t>(max_retries);
        set.q_MeV = 0.0F;
        set.neutron_ke_MeV = 0.0F;
    }

    return set;
}


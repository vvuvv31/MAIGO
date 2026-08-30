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
    const float* ion_energy_grid,
    const float* ion_csda_a1,
    const float energy_cutoff_MeV,
    const float p_target_h,
    const carbon::FredEventLibSetDevice lib_h = {},
    const carbon::FredEventLibSetDevice lib_o = {}) noexcept {

    InelasticProductSet set{};
    const auto e_per_u = incident_energy_MeV / 12.0F;
    if (e_per_u <= 0.5F) {
        set.local_deposit_MeV = incident_energy_MeV;
        return set;
    }

    constexpr float two_pi = 6.2831853071795864769F;
    constexpr float deg_to_rad = 0.017453292519943295F;
    constexpr uint32_t kInelasticBaseDim = 10000;

    const float u_target = rng::uniform01(spot_seed, rng_history, steps, kInelasticBaseDim + 0);
    const uint8_t target_idx = (u_target < p_target_h) ? 0 : 2; // 0 for H, 2 for O
    const float* proj_prob = (target_idx == 0) ? prob_proj_h : prob_proj_o;
    const float* tgt_prob = carbon::kFredProbO.data();
    (void)prob_tgt_h;
    (void)prob_tgt_o;

    const auto& event_set = (target_idx == 0) ? lib_h : lib_o;
    carbon::FredEventLibDevice event_lib{};
    if (event_set.count > 0) {
        std::uint32_t upper = 0;
        while (upper + 1 < event_set.count &&
               e_per_u > event_set.libraries[upper].reference_energy_MeVu) ++upper;
        std::uint32_t chosen = upper;
        if (upper > 0 && e_per_u < event_set.libraries[upper].reference_energy_MeVu) {
            const auto lower = upper - 1;
            const auto e0 = event_set.libraries[lower].reference_energy_MeVu;
            const auto e1 = event_set.libraries[upper].reference_energy_MeVu;
            const auto mix = sycl::clamp((e_per_u - e0) / (e1 - e0), 0.0F, 1.0F);
            const auto u_energy_mix = rng::uniform01(spot_seed, rng_history, steps,
                                                      kInelasticBaseDim + 2);
            chosen = u_energy_mix < mix ? upper : lower;
        }
        event_lib = event_set.libraries[chosen];
    }
    if (event_lib.event_count > 0 && event_lib.fragment_count != nullptr) {
        const float u_ev = rng::uniform01(spot_seed, rng_history, steps, kInelasticBaseDim + 3);
        auto ie = static_cast<uint32_t>(u_ev * static_cast<float>(event_lib.event_count));
        if (ie >= event_lib.event_count) {
            ie = event_lib.event_count - 1;
        }
        const float scale = e_per_u / sycl::fmax(event_lib.reference_energy_MeVu, 1.0F);
        const uint8_t nfrag = event_lib.fragment_count[ie];
        InelasticProductSet from_lib{};
        float charged = 0.0F;
        uint8_t charged_count = 0;
        const uint32_t maxf = event_lib.max_fragments;
        for (uint8_t f = 0; f < nfrag && f < maxf; ++f) {
            const std::size_t slot = static_cast<std::size_t>(ie) * maxf + f;
            const int z = static_cast<int>(event_lib.z[slot]);
            const int a = static_cast<int>(event_lib.a[slot]);
            if (z < 1 || a < 1) {
                continue;
            }
            float ke = event_lib.ke_MeV[slot] * scale;
            charged += ke;
            if (charged > incident_energy_MeV + 1.0e-3F) {
                from_lib = {};
                charged_count = 0;
                break;
            }
            if (charged_count >= carbon::kMaxInelasticProducts) {
                from_lib.untracked_energy_MeV += ke;
                continue;
            }
            // TOPAS beam is local -Z; MAIGO parent +Z uses flipped z cosine.
            const float lx = event_lib.ux[slot];
            const float ly = event_lib.uy[slot];
            const float lz = -event_lib.uz[slot];
            const auto rot = rotate_local_direction(lx, ly, lz, parent_dir);
            auto& p = from_lib.products[charged_count++];
            p.z = static_cast<int16_t>(z);
            p.a = static_cast<int16_t>(a);
            p.energy_MeV = ke;
            p.pos_x_mm = pos_x_mm;
            p.pos_y_mm = pos_y_mm;
            p.pos_z_mm = pos_z_mm;
            p.dir_x = rot.x;
            p.dir_y = rot.y;
            p.dir_z = rot.z;
            p.weight = 1.0F;
            p.parent_history = rng_history;
        }
        if (charged_count > 0) {
            from_lib.count = charged_count;
            float nke = event_lib.neutron_ke_MeV[ie] * scale;
            if (charged + nke > incident_energy_MeV) {
                nke = sycl::fmax(0.0F, incident_energy_MeV - charged);
            }
            from_lib.untracked_energy_MeV += nke;
            from_lib.neutron_ke_MeV = nke;
            const float leftover = sycl::fmax(0.0F, incident_energy_MeV - charged - nke);
            from_lib.untracked_energy_MeV += leftover;
            from_lib.remnant_excitation_MeV = leftover;
            from_lib.retries_used = 1;
            return from_lib;
        }
    }

    const auto base_q_MeV = 0.0F;
    const auto max_available_MeV = incident_energy_MeV;

    // 3. Sample projectile and target fragment sets with resampling (Section 2.1.3)
    constexpr int max_retries = 128;
    bool accepted = false;

    for (int retry = 0; retry < max_retries; ++retry) {
        InelasticProductSet candidate{};
        candidate.untracked_energy_MeV = base_q_MeV;

        int a_rem = 12;
        int z_rem = 6;
        uint8_t frag_species[12]{};
        bool is_target_frag[12]{};
        uint8_t num_frags = 0;

        float wproj[18];
        carbon::fill_energy_dependent_inclusive_weights(e_per_u, proj_prob, wproj);
        float projectile_uniforms[8]{};
        for (int j = 0; j < 8; ++j) {
            projectile_uniforms[j] = rng::uniform01(
                spot_seed, rng_history, steps,
                kInelasticBaseDim +
                    static_cast<uint32_t>(retry * 200 + 1 + j));
        }
        uint8_t joint_idx[8]{};
        int leftover_n = 0;
        const int njoint = carbon::fill_projectile_constrained_channel(
            wproj, projectile_uniforms, 8, joint_idx,
            carbon::kMaxProjectileFragments, &leftover_n);
        if (njoint < 0) {
            continue;
        }
        for (int j = 0; j < njoint && num_frags < carbon::kMaxProjectileFragments; ++j) {
            frag_species[num_frags] = joint_idx[j];
            is_target_frag[num_frags] = false;
            ++num_frags;
        }
        a_rem = 0;  // leftover_n is an emitted neutron packet, not an A residual
        z_rem = 0;

        // Transport sampled target fragments and retain remaining A/Z as an
        // explicit zero-kinetic-energy remnant in paper-minimal mode.
        int target_a_rem = (target_idx == 0) ? 1 : 16;
        int target_z_rem = (target_idx == 0) ? 1 : 8;
        int heavy_tgt_idx = -1;
        int tgt_step = 0;
        if (target_idx == 0) {
            if (num_frags < carbon::kMaxInelasticProducts) {
                frag_species[num_frags] = 1;
                is_target_frag[num_frags] = true;
                ++num_frags;
                target_a_rem = 0;
                target_z_rem = 0;
            }
        }
        while (target_idx != 0 && (target_a_rem > 0 || target_z_rem > 0) &&
               num_frags < carbon::kMaxInelasticProducts && tgt_step < 12) {
            ++tgt_step;
            const float u_tgt = rng::uniform01(
                spot_seed, rng_history, steps,
                kInelasticBaseDim + static_cast<uint32_t>(retry * 200 + 40 + tgt_step));
            const int iso_idx = carbon::sample_table1_isotope(tgt_prob, target_a_rem, target_z_rem, u_tgt);
            const auto iso = carbon::kFredIsotopes[iso_idx];
            if (iso.a <= 0 || iso.a > target_a_rem || iso.z > target_z_rem) {
                break;
            }
            if (iso.z >= 3 || iso.a >= 6) {
                target_a_rem -= iso.a;
                target_z_rem -= iso.z;
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
        float projectile_e_sum = 0.0F;
        float total_product_ke_MeV = 0.0F;
        float running_a_sum = 0.0F;
        uint8_t charged_count = 0;
        bool kinematics_ok = true;

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

            // Eq. (12) component weights integrated on the same fixed domain
            // used below: E95 in [0,Emax], theta95 in [0,180 deg].
            const float a_e = sycl::fmax(params.a_e, 0.005F);
            const float a_th = sycl::fmax(params.a_th, 0.01F);
            constexpr float sqrt_half_pi = 1.2533141373155F;
            constexpr float inv_sqrt_two = 0.7071067811865F;
            const float sigma_e = sycl::fmax(params.sigma_e, 1.0e-6F);
            const float sigma_th = sycl::fmax(params.sigma_th, 1.0e-6F);
            const float gauss_e_int = sigma_e * sqrt_half_pi *
                (sycl::erf((carbon::kFredReferenceEnergyMaxMeVu - params.mean_e) *
                            inv_sqrt_two / sigma_e) +
                 sycl::erf(params.mean_e * inv_sqrt_two / sigma_e));
            const float gauss_th_int = sigma_th * sqrt_half_pi *
                sycl::erf(180.0F * inv_sqrt_two / sigma_th);
            const float i_gauss = params.a2 * gauss_e_int * gauss_th_int;
            const float i_exp = params.a1 *
                (1.0F - sycl::exp(-a_e * carbon::kFredReferenceEnergyMaxMeVu)) / a_e *
                (1.0F - sycl::exp(-a_th * 180.0F)) / a_th;
            const float p_gauss = i_gauss / sycl::fmax(1.0e-6F, i_gauss + i_exp);

            const bool sample_gauss =
                carbon::eq12_sample_gaussian(iso.z, iso.a, is_tgt, u_mix, p_gauss);

            float frag_e_u = 0.0F;
            float theta_rad = 0.0F;

            if (sample_gauss) {
                // Box-Muller Gaussian for E95
                const auto u_log = sycl::fmax(u_e0, 1.0e-10F);
                const auto gauss_e = sycl::sqrt(-2.0F * sycl::log(u_log)) * sycl::cos(two_pi * u_e1);
                const auto e95 = sycl::fmax(params.mean_e + params.sigma_e * gauss_e, 0.0F);

                if (!is_tgt) {
                    frag_e_u = carbon::sample_projectile_fragment_Eu(
                        e95, e_per_u, iso.a, projectile_e_sum, running_a_sum);
                    if (!(frag_e_u > 0.0F)) {
                        kinematics_ok = false;
                        break;
                    }
                } else {
                    frag_e_u = e95 * (e_per_u / 95.0F);
                }

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
                // Sample once from the fixed 95 MeV/u reference domain.
                const auto u_e_log = sycl::fmax(u_e0, 1.0e-8F);
                const float emax_u = carbon::kFredReferenceEnergyMaxMeVu;
                const float norm = 1.0F - sycl::exp(-a_e * emax_u);
                const float e95 = -sycl::log(1.0F - u_e_log * sycl::fmax(norm, 1.0e-8F)) / a_e;
                if (!is_tgt) {
                    frag_e_u = carbon::sample_projectile_fragment_Eu(
                        e95, e_per_u, iso.a, projectile_e_sum, running_a_sum);
                    if (!(frag_e_u > 0.0F)) {
                        kinematics_ok = false;
                        break;
                    }
                } else {
                    frag_e_u = e95 * (e_per_u / 95.0F);
                }

                const auto u_th_log = sycl::fmax(u_th0, 1.0e-8F);
                const float th_max = 180.0F;
                const float th_norm = 1.0F - sycl::exp(-a_th * th_max);
                const auto theta_deg =
                    -sycl::log(1.0F - u_th_log * sycl::fmax(th_norm, 1.0e-8F)) / a_th;
                const bool skip_angle_scale = (iso.z == 0) || (iso.z == 1 && iso.a == 1);
                if (skip_angle_scale) {
                    theta_rad = theta_deg * deg_to_rad;
                } else {
                    theta_rad = theta_deg * deg_to_rad * sycl::sqrt(95.0F / e_per_u);
                }
            }

            frag_e_u = sycl::fmax(0.0F, frag_e_u);
            const float frag_e_tot = af * frag_e_u;
            total_product_ke_MeV += frag_e_tot;
            if (!(total_product_ke_MeV <= max_available_MeV + 1.0e-3F)) {
                kinematics_ok = false;
                break;
            }
            if (!is_tgt) {
                projectile_e_sum += frag_e_tot;
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
                if (heavy_tgt && af > 0.0F) {
                    const int sp_species = carbon::get_charged_species_idx(iso.z, iso.a);
                    const float e_u = frag_e_tot / af;
                    if (ion_sp_lut != nullptr && ion_csda_a1 != nullptr && ion_energy_grid != nullptr &&
                        ion_sp_table_size > 1) {
                        const std::size_t off =
                            static_cast<std::size_t>(sp_species) * static_cast<std::size_t>(ion_sp_table_size);
                        range_mm = carbon::csda_range_mm_device(
                            ion_energy_grid, ion_sp_lut + off, ion_csda_a1 + off,
                            static_cast<std::size_t>(ion_sp_table_size), e_u, iso.a);
                    } else {
                        float sp_est = sycl::fmax(0.80F * static_cast<float>(iso.z * iso.z), 0.20F);
                        range_mm = frag_e_tot / sp_est;
                    }
                }
                const bool local_stop =
                    heavy_tgt && carbon::remnant_local_stop_from_csda(range_mm, frag_e_tot,
                                                                     energy_cutoff_MeV);
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
        if (!kinematics_ok) {
            continue;
        }

        // Leftover projectile neutrons: one lumped KE packet (no n/p particle dump).
        if (leftover_n > 0) {
            const auto nparams = (target_idx == 0) ? carbon::kFredParamsH[0] : carbon::kFredParamsO[0];
            const uint32_t n_rng = kInelasticBaseDim + static_cast<uint32_t>(retry * 200 + 35);
            const float u_n = sycl::fmax(
                rng::uniform01(spot_seed, rng_history, steps, n_rng), 1.0e-8F);
            const float a_e_n = sycl::fmax(nparams.a_e, 0.005F);
            const float emax_u = carbon::kFredReferenceEnergyMaxMeVu;
            const float nnorm = 1.0F - sycl::exp(-a_e_n * emax_u);
            const float e95_n = -sycl::log(1.0F - u_n * sycl::fmax(nnorm, 1.0e-8F)) / a_e_n;
            const float n_ke = sycl::fmax(
                0.0F, static_cast<float>(leftover_n) * e95_n * (e_per_u / 95.0F));
            total_product_ke_MeV += n_ke;
            if (!(total_product_ke_MeV <= max_available_MeV + 1.0e-3F)) {
                continue;
            }
            candidate.untracked_energy_MeV += n_ke;
        }


        candidate.count = charged_count;

        const float neutron_ke_MeV = sycl::fmax(0.0F, candidate.untracked_energy_MeV - base_q_MeV);
        candidate.untracked_energy_MeV += candidate.product_capacity_overflow_MeV;
        // Paper-minimal Q=0 mode keeps unused available energy explicit as
        // escaped/excitation energy; it is not deposited locally.
        const float excitation_MeV =
            sycl::fmax(0.0F, max_available_MeV - total_product_ke_MeV);
        candidate.untracked_energy_MeV += excitation_MeV;
        candidate.remnant_kinetic_MeV = 0.0F;
        candidate.remnant_excitation_MeV = excitation_MeV;

        candidate.q_MeV = base_q_MeV;
        candidate.neutron_ke_MeV = neutron_ke_MeV;
        candidate.retries_used = static_cast<uint8_t>(retry + 1);
        candidate.leftover_proj_a = static_cast<int8_t>(a_rem);
        candidate.leftover_proj_z = static_cast<int8_t>(z_rem);
        candidate.leftover_tgt_a = static_cast<int8_t>(sycl::max(target_a_rem, 0));
        candidate.leftover_tgt_z = static_cast<int8_t>(sycl::max(target_z_rem, 0));
        candidate.n_emitted = num_frags;
        for (uint8_t s = 0; s < num_frags && s < carbon::kMaxInelasticProducts; ++s) {
            candidate.emitted_idx[s] = frag_species[s];
        }

        // Accept only complete events whose projectile, target, neutron and
        // local-remnant kinetic energies all fit the same signed budget.
        if (total_product_ke_MeV <= max_available_MeV + 1.0e-3F) {
            set = candidate;
            accepted = true;
            break;
        }
    }


    if (!accepted) {
        set.count = 1;
        set.local_deposit_MeV = 0.0F;
        set.untracked_energy_MeV = 0.0F;
        set.model_unassigned_MeV = 0.0F;
        set.resample_failed = 1;
        set.retries_used = static_cast<uint8_t>(max_retries);
        set.q_MeV = 0.0F;
        set.neutron_ke_MeV = 0.0F;
        auto& p = set.products[0];
        p.z = 6;
        p.a = 12;
        p.energy_MeV = incident_energy_MeV;
        p.pos_x_mm = pos_x_mm;
        p.pos_y_mm = pos_y_mm;
        p.pos_z_mm = pos_z_mm;
        p.dir_x = parent_dir.x;
        p.dir_y = parent_dir.y;
        p.dir_z = parent_dir.z;
        p.weight = 1.0F;
        p.parent_history = rng_history;
    }

    return set;
}


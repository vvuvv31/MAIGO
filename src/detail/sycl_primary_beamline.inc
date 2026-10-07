// Shared source, Cu/air transport, and medium-entry handoff.
// Included inside the primary kernel; all step/RNG/ledger operations stay inline.
// kUrbanBeamlinePath describes upstream physics, independent of water/CT downstream.
                if constexpr (kPrimaryStage != 4) {
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

                // Offline history-cohort attribution. Inactive source slots have
                // zero incident energy, retain original history/RNG identity, and
                // generate no descendants. This is not a production pruning mode.
                if (copper_history_mask_device && !copper_history_mask_device[global_history])
                    energy_MeV = 0.0F;
                const auto sampled_initial_energy_MeVu = energy_MeV * inverse_mass_number;
                // Store the energy actually transported after source sampling/cutoff.
                // This does not consume RNG or change the source distribution.
                sampled_incident_device[global_history] = energy_MeV;
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
                    // Sample angular emittance in the current spot beam frame.
                    auto slope_x = x_prime;
                    auto slope_y = y_prime;

                    const auto inv_norm =
                        sycl::rsqrt(1.0F + slope_x * slope_x + slope_y * slope_y);
                    local_dx = slope_x * inv_norm;
                    local_dy = slope_y * inv_norm;
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
                const auto composed_direction = compose_tps_direction(
                    local_dx, local_dy, local_dz, ux_x, ux_y, ux_z,
                    uy_x, uy_y, uy_z, uz_x, uz_y, uz_z);
                auto direction_x = composed_direction.x;
                auto direction_y = composed_direction.y;
                auto direction_z = composed_direction.z;

                {
                    const auto inv_n = sycl::rsqrt(sycl::fmax(
                        1.0e-20F, direction_x * direction_x + direction_y * direction_y +
                                      direction_z * direction_z));
                    direction_x *= inv_n;
                    direction_y *= inv_n;
                    direction_z *= inv_n;
                }

                float beamline_air_loss_MeV = 0.0F;
                const auto minibeam_block_entrance_z =
                    minibeam_block_center_z - 0.5F * minibeam_block_thickness;
                const auto minibeam_block_exit_z =
                    minibeam_block_center_z + 0.5F * minibeam_block_thickness;
                UrbanV2TrackState air_urban_state{};
                std::uint64_t air_urban_step = 0;
                DiscreteDeltaClockCandidate<float> air_ion_clock;
                float air_ion_threshold=std::numeric_limits<float>::max(),air_ion_rate=0;

                bool air_urban_boundary = true;
                auto advance_air = [&](const float boundary_path, const UrbanV2SafetyCtx& safety, const float ceiling) {
                    UrbanV2Material material{};
                    const auto reference = air_urban_device.materials[0];
                    material.loss_context = air_urban_device.at(0, air_urban_projectile, energy_MeV);
                    material.zeff = reference.zeff;
                    material.radlen_mm = reference.radiation_length_mm;
                    material.density_g_per_cm3 = reference.density;
                    material.projectile_z = primary_atomic_number;
                    material.projectile_a = primary_mass_number;
                    material.mass_mev = air_urban_device.species[air_urban_projectile].mass_mev;
                    float ion_distance=std::numeric_limits<float>::infinity();
                    if(air_ion_count) {
                        if(energy_MeV<air_ion_e_device[0]*primary_mass_number ||
                           energy_MeV>air_ion_e_device[air_ion_count-1]*primary_mass_number) {
                            urban_record_first_failure(minibeam_event_counts_device,rng_history,air_urban_step,0U,106);
                            return false;
                        }
                        if(energy_MeV<=air_ion_peak) {
                            if(energy_MeV*1.25F<air_ion_threshold)air_ion_threshold=energy_MeV;
                        } else if(energy_MeV<air_ion_threshold)air_ion_threshold=sycl::fmax(air_ion_peak,.8F*energy_MeV);
                        air_ion_rate=minibeam_linear_table(air_ion_e_device,air_ion_rate_device,air_ion_count,
                            air_ion_threshold*inverse_mass_number);
                        if(!air_ion_clock.active)
                            air_ion_clock.arm(-sycl::log(sycl::fmax(0x1p-24F,rng::msc_unit_strict(
                                spot_seed ^ 0xB3E520A174C968DFULL,rng_history,air_urban_step,125U))));
                        ion_distance=air_ion_clock.distance(air_ion_rate);
                    }
                    const auto step = urban_v2_propose_and_sample(
                        Direction3F{direction_x,direction_y,direction_z}, energy_MeV,
                        sycl::fmin(ceiling,ion_distance), boundary_path, position_x_mm,position_y_mm,position_z_mm,
                        direction_x,direction_y,direction_z,safety,air_urban_boundary,air_urban_state,
                        material,1.0F,spot_seed ^ 0xA17B09D5C628E341ULL,rng_history,air_urban_step++,0U,50);
                    const int reason = step.limit_reason < 0 ? -step.limit_reason : urban_step_field_reason(step);
                    if (reason || step.numeric_guard_tripped || air_urban_step >= 100000U) {
                        urban_record_first_failure(minibeam_event_counts_device,rng_history,
                            air_urban_step,0U,reason ? reason : 104);
                        return false;
                    }
                    if(air_ion_count) {
                        const bool proposed=step.final_true_path_mm>=ion_distance;
                        air_ion_clock.consume(step.final_true_path_mm,air_ion_rate,proposed);
                        sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,
                            sycl::memory_scope::device,sycl::access::address_space::global_space>(
                                minibeam_event_counts_device[air_ion_step_slot]).fetch_add(1U);
                        if(proposed)sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,
                            sycl::memory_scope::device,sycl::access::address_space::global_space>(
                                minibeam_event_counts_device[air_ion_proposal_slot]).fetch_add(1U);
                    }
                    position_x_mm += step.final_geom_path_mm*direction_x + step.displacement_mm.x;
                    position_y_mm += step.final_geom_path_mm*direction_y + step.displacement_mm.y;
                    position_z_mm += step.final_geom_path_mm*direction_z + step.displacement_mm.z;
                    direction_x=step.direction.x; direction_y=step.direction.y; direction_z=step.direction.z;
                    air_urban_boundary=step.boundary_crossed;
                    const auto stopping = minibeam_linear_table(minibeam_air_sp_energies_device,
                        minibeam_air_sp_values_device,minibeam_air_sp_count,energy_MeV*inverse_mass_number);
                    auto loss=sycl::fmin(energy_MeV,stopping*step.final_true_path_mm);
                    if (energy_MeV-loss <= energy_cutoff_MeV) loss=energy_MeV;
                    energy_MeV-=loss; beamline_air_loss_MeV+=loss;
                    return true;
                };
                auto advance_air_to_plane = [&](const float target_z) {
                    UrbanV2SafetyCtx safety{}; safety.kind=1;
                    safety.box_x0=-3000.0F; safety.box_x1=3000.0F;
                    safety.box_y0=-3000.0F; safety.box_y1=3000.0F;
                    safety.box_z0=position_z_mm; safety.box_z1=target_z;
                    air_urban_boundary=true;
                    while (position_z_mm < target_z-1.0e-6F && direction_z>1.0e-8F && energy_MeV>energy_cutoff_MeV) {
                        const float boundary=(target_z-position_z_mm)/direction_z;
                        safety.box_boundary_path_mm=boundary;
                        if (!advance_air(boundary,safety,1.0e6F)) return false;
                    }
                    return true;
                };
                if (minibeam_air_urban) {
                    if (!advance_air_to_plane(minibeam_block_entrance_z)) energy_MeV=0.0F;
                }
                const auto minibeam_hits_copper = enable_minibeam &&
                    minibeam_ray_hits_cylindrical_copper(
                        position_x_mm, position_y_mm, position_z_mm,
                        direction_x, direction_y, direction_z,
                        minibeam_cos, minibeam_sin, minibeam_block_radius,
                        minibeam_block_center_z,
                        minibeam_block_thickness, minibeam_slit_count,
                        minibeam_slit_width, minibeam_slit_pitch,
                        0.5F * minibeam_slit_length,
                        minibeam_slit_thickness, minibeam_slit_offset);
                auto minibeam_primary_elastic = false;
                auto minibeam_source_slit = 0;

                // Versioned touch state (history scope): flushed to the
                // phase-space record after transport. ever_in_copper and the Cu
                // true-path accumulator below are the actual-touch observables.
                auto beamline_ever_in_copper_hist = false;
                auto beamline_cu_true_path_hist_mm = 0.0F;
                // urban_v2 step diagnostics (research-only aggregates).
                std::uint32_t beamline_cu_steps_hist = 0;
                std::uint32_t beamline_disp_below_hist = 0;
                std::uint32_t beamline_disp_accept_hist = 0;
                std::uint32_t beamline_disp_reduce_hist = 0;
                std::uint32_t beamline_disp_cancel_hist = 0;
                std::uint32_t beamline_cth_one_hist = 0;
                std::uint32_t beamline_limit_user_hist = 0;
                std::uint32_t beamline_limit_msc_hist = 0;
                std::uint32_t beamline_limit_geom_hist = 0;
                std::uint32_t beamline_limit_range_hist = 0;
                float beamline_g_sum_hist_mm = 0.0F;
                float beamline_t_sum_hist_mm = 0.0F;
                float beamline_delta_sum_hist_mm = 0.0F;
                float beamline_raw2_sum_hist_mm2 = 0.0F;
                float beamline_acc2_sum_hist_mm2 = 0.0F;
                CopperHistoryDiagnostic copper_history_diag{};
                if (copper_diagnostic_device) {
                    copper_history_diag.entry_x=position_x_mm;
                    copper_history_diag.entry_y=position_y_mm;
                    copper_history_diag.entry_z=position_z_mm;
                    copper_history_diag.entry_energy=energy_MeV;
                    copper_history_diag.entry_dx=direction_x;
                    copper_history_diag.entry_dy=direction_y;
                    copper_history_diag.entry_dz=direction_z;
                    const auto u=minibeam_cos*position_x_mm+minibeam_sin*position_y_mm;
                    float distance=1.0e30F;
                    for (int slit=-minibeam_slit_count/2;slit<=minibeam_slit_count/2;++slit)
                        distance=sycl::fmin(distance,sycl::fabs(u-minibeam_slit_offset-
                            slit*minibeam_slit_pitch)-.5F*minibeam_slit_width);
                    copper_history_diag.wall_distance=distance;
                    copper_history_diag.cohort=copper_diagnostic_cohort(distance);
                }
                if (enable_minibeam && direction_z > 1.0e-8F) {
                    const auto slit_entrance_distance =
                        (minibeam_block_entrance_z - position_z_mm) /
                        direction_z;
                    const auto slit_entrance_x =
                        position_x_mm + slit_entrance_distance * direction_x;
                    const auto slit_entrance_y =
                        position_y_mm + slit_entrance_distance * direction_y;
                    const auto slit_u = minibeam_cos * slit_entrance_x +
                        minibeam_sin * slit_entrance_y - minibeam_slit_offset;
                    minibeam_source_slit = nearest_minibeam_slit(
                        slit_u, minibeam_slit_pitch);
                }

                if (minibeam_copper_transport && (minibeam_hits_copper || minibeam_air_urban) &&
                    direction_z > 1.0e-8F) {
                    const auto incident_beamline_energy = energy_MeV;
                    const auto air_loss_at_block_entry = beamline_air_loss_MeV;
                    const auto block_entrance_z = minibeam_block_entrance_z;
                    const auto block_exit_z = minibeam_block_exit_z;
                    const auto to_entrance =
                        (block_entrance_z - position_z_mm) / direction_z;
                    if (to_entrance >= 0.0F) {
                        position_x_mm += to_entrance * direction_x;
                        position_y_mm += to_entrance * direction_y;
                        position_z_mm = block_entrance_z;
                    }
                    float removed_energy = 0.0F;
                    std::uint64_t beamline_step = 0;
                    float copper_nuclear_tau_remaining = 0.0F;
                    bool copper_nuclear_tau_active = false;
                    // Urban v2 fMinimal state (Phase 2/5): persists across the
                    // steps of one Cu traversal; at_boundary marks steps that
                    // start at the block entrance or a material interface.
                    UrbanV2TrackState urban_v2_state{};
                    bool urban_v2_at_boundary = true;
                    bool urban_v2_prev_in_cu = false;
                    UrbanV2GeomCtx urban_v2_geom{};
                    urban_v2_geom.cos_a = minibeam_cos;
                    urban_v2_geom.sin_a = minibeam_sin;
                    urban_v2_geom.radius_mm = minibeam_block_radius;
                    urban_v2_geom.slit_count = minibeam_slit_count;
                    urban_v2_geom.slit_width_mm = minibeam_slit_width;
                    urban_v2_geom.slit_pitch_mm = minibeam_slit_pitch;
                    urban_v2_geom.slit_half_len_mm = 0.5F * minibeam_slit_length;
                    urban_v2_geom.slit_offset_mm = minibeam_slit_offset;
                    urban_v2_geom.block_entrance_z_mm = minibeam_block_entrance_z;
                    urban_v2_geom.block_exit_z_mm = minibeam_block_exit_z;
                    UrbanMcsContext copper_loss_context{};
                    // (touch accumulators live at history scope:
                    // beamline_ever_in_copper_hist / beamline_cu_true_path_hist_mm)
                    constexpr std::uint64_t maximum_beamline_steps = 100000;
                    while (energy_MeV > energy_cutoff_MeV &&
                           direction_z > 1.0e-8F &&
                           position_z_mm < block_exit_z - 1.0e-6F &&
                           beamline_step < maximum_beamline_steps) {
                        if (copper_diagnostic_device) ++copper_history_diag.loop_iterations;
                        // Limit the transported path, not only its axial
                        // projection. This keeps the Poisson tail truncation
                        // and every material interaction bounded for oblique
                        // slit-edge tracks.
                        const bool starts_in_copper = minibeam_point_in_copper(
                            position_x_mm+1.0e-5F*direction_x,position_y_mm+1.0e-5F*direction_y,
                            minibeam_cos,minibeam_sin,minibeam_block_radius,minibeam_slit_count,
                            minibeam_slit_width,minibeam_slit_pitch,0.5F*minibeam_slit_length,minibeam_slit_offset);
                        const auto nominal_path_step = sycl::fmin(
                            minibeam_air_urban && !starts_in_copper ? 1.0e6F : minibeam_copper_max_step,
                            (block_exit_z - position_z_mm) / direction_z);
                        const auto path_step = minibeam_air_urban
                            ? minibeam_path_to_surface_interval(
                                  position_x_mm,position_y_mm,direction_x,direction_y,
                                  minibeam_cos,minibeam_sin,minibeam_block_radius,minibeam_slit_count,
                                  minibeam_slit_width,minibeam_slit_pitch,0.5F*minibeam_slit_length,
                                  minibeam_slit_offset,nominal_path_step)
                            : minibeam_copper_exact_material_boundaries
                            ? minibeam_path_to_material_boundary(
                                  position_x_mm, position_y_mm, direction_x,
                                  direction_y, minibeam_cos, minibeam_sin,
                                  minibeam_block_radius, minibeam_slit_count,
                                  minibeam_slit_width, minibeam_slit_pitch,
                                  0.5F * minibeam_slit_length,
                                  minibeam_slit_offset, nominal_path_step)
                            : nominal_path_step;
                        const auto axial_step = path_step * direction_z;
                        const auto midpoint_x =
                            position_x_mm + 0.5F * path_step * direction_x;
                        const auto midpoint_y =
                            position_y_mm + 0.5F * path_step * direction_y;
                        const auto in_copper = minibeam_point_in_copper(
                            midpoint_x, midpoint_y, minibeam_cos, minibeam_sin,
                            minibeam_block_radius,
                            minibeam_slit_count, minibeam_slit_width,
                            minibeam_slit_pitch, 0.5F * minibeam_slit_length,
                            minibeam_slit_offset);
                        if (!in_copper) {
                            if (minibeam_air_urban) {
                                UrbanV2SafetyCtx safety{}; safety.kind=0; safety.cu=urban_v2_geom;
                                air_urban_boundary=air_urban_boundary || urban_v2_prev_in_cu;
                                if (!advance_air(path_step,safety,air_slit_max_step)) break;
                                ++beamline_step;
                                urban_v2_prev_in_cu=false;
                                continue;
                            }
                            position_x_mm += path_step * direction_x;
                            position_y_mm += path_step * direction_y;
                            position_z_mm += axial_step;
                            const auto air_loss = sycl::fmin(
                                energy_MeV,
                                minibeam_linear_table(
                                    minibeam_air_sp_energies_device,
                                    minibeam_air_sp_values_device,
                                    minibeam_air_sp_count,
                                    energy_MeV * inverse_mass_number) *
                                    path_step);
                            energy_MeV -= air_loss;
                            beamline_air_loss_MeV += air_loss;
                            ++beamline_step;
                            urban_v2_prev_in_cu = false;
                            continue;
                        }
                        if (minibeam_air_urban) { air_urban_boundary=true; beamline_ever_in_copper_hist=true; }
                        if (copper_diagnostic_device) ++copper_history_diag.copper_proposals;
                        const auto energy_u = energy_MeV * inverse_mass_number;
                        const auto stopping = minibeam_linear_table(
                            minibeam_copper_sp_energies_device,
                            minibeam_copper_sp_values_device,
                            minibeam_copper_sp_count, energy_u);
                        const auto rate_elastic_index = minibeam_elastic_nearest_bracketed(
                            minibeam_copper_elastic_device,
                            minibeam_copper_elastic_count, energy_u,
                            minibeam_copper_elastic_brackets_device,
                            minibeam_copper_elastic_bracket_bins,
                            minibeam_copper_elastic_bracket_first_energy,
                            minibeam_copper_elastic_bracket_inverse_width);
                        const auto rate_elastic = minibeam_copper_enable_elastic
                            ? minibeam_copper_elastic_device[
                                  rate_elastic_index].macroscopic_rate_per_mm
                            : 0.0F;
                        const auto rate_inelastic = minibeam_copper_enable_inelastic
                            ? minibeam_linear_table(
                                  minibeam_copper_xs_energies_device,
                                  minibeam_copper_xs_values_device,
                                  minibeam_copper_xs_count, energy_u)
                            : 0.0F;
                        const auto rate_total = rate_elastic + rate_inelastic;
                        if (rate_total > 0.0F && !copper_nuclear_tau_active) {
                            // FP32 midpoint conversion can round the largest
                            // 24-bit draw to 1.0. Keep the exponential variate
                            // strictly positive; all other draws are unchanged.
                            const auto optical_uniform = sycl::fmin(
                                0x1.fffffep-1F, sycl::fmax(
                                    1.0e-7F, rng::uniform01(
                                        spot_seed, rng_history, beamline_step, 60)));
                            copper_nuclear_tau_remaining = -sycl::log(optical_uniform);
                            copper_nuclear_tau_active = true;
                        }
                        auto collision_in_step = rate_total > 0.0F &&
                            copper_nuclear_tau_active &&
                            copper_nuclear_tau_remaining <= rate_total * path_step;
                        const auto transport_path = collision_in_step
                            ? copper_nuclear_tau_remaining / rate_total
                            : path_step;
                        const Direction3F pre_scatter_direction{
                            direction_x, direction_y, direction_z};
                        auto correlated_scattering = CorrelatedScatteringStep{
                            pre_scatter_direction, Direction3F{0.0F, 0.0F, 0.0F}};
                        if (minibeam_copper_enable_mcs &&
                            energy_MeV > energy_cutoff_MeV) {
                            {
                                // Re-entry after air also starts at a
                                // geometry boundary (reference stepStatus).
                                urban_v2_at_boundary = urban_v2_at_boundary ||
                                    !urban_v2_prev_in_cu;
                                // Phase 5: external TRUE candidate (user
                                // ceiling) -> fMinimal limit -> true->geom ->
                                // geometry truncation -> final true -> scatter.
                                // boundary_path_mm is the geometry-reachable
                                // length (exact material boundaries honored).
                                // currentRange comes from the packaged loss context.
                                if ((kUrbanBeamlinePath || global_urban)) {
                                    copper_loss_context = urban_device.at(
                                        urban_device.material(-2, minibeam_copper_density),
                                        urban_device.projectile(primary_atomic_number, primary_mass_number),
                                        energy_MeV);
                                }
                                correlated_scattering =
                                    copper_urban_v2_propose_and_sample(
                                        pre_scatter_direction, energy_MeV,
                                        primary_atomic_number, primary_mass_number,
                                        (kUrbanBeamlinePath || global_urban) && rate_total > 0.0F
                                            ? sycl::fmin(minibeam_copper_max_step,
                                                copper_nuclear_tau_remaining / rate_total)
                                            : minibeam_copper_max_step,
                                        (kUrbanBeamlinePath || global_urban) ? path_step : transport_path,
                                        position_x_mm, position_y_mm,
                                        position_z_mm, direction_x, direction_y,
                                        direction_z, urban_v2_geom,
                                        urban_v2_at_boundary, urban_v2_state,
                                        copper_loss_context,
                                        minibeam_copper_density,
                                        minibeam_copper_radiation_length,
                                        minibeam_copper_mcs_scale, spot_seed,
                                        rng_history, beamline_step, 0U, 50, primary_copper_scattering_mass_MeV);
                                urban_v2_at_boundary =
                                    correlated_scattering.boundary_crossed;
                                urban_v2_prev_in_cu = true;
                            }
                        }
                        // R4: explicit step proposal/finalization contract for
                        // Update the Urban v2 track state.
                        // Geometry consumes final g; loss/fluctuation and the
                        // reaction path below consume final t.  No post-hoc
                        // t/g rescaling of the ledger.
                        const auto urban_v2_proposal =
                            correlated_scattering.proposal_valid;
                        // Fix B5/C2.3: an invalid Urban proposal must fail the
                        // run and stop this history (flag + record; the host
                        // throws). The old code advanced the full path
                        // unscattered; a failed proposal must never continue
                        // as normal transport. Field contract reasons 1-5,8
                        // (no segment loop here: single advance, so no
                        // progress reasons; overshoot is structural).
                        {
                            const int cu_reason = correlated_scattering.limit_reason < 0
                                ? -correlated_scattering.limit_reason
                                : urban_step_field_reason(correlated_scattering);
                            if (correlated_scattering.numeric_guard_tripped &&
                                minibeam_event_counts_device != nullptr) {
                                urban_record_guard_evidence(
                                    minibeam_event_counts_device, 1ULL,
                                    energy_MeV,
                                    correlated_scattering
                                        .proposed_true_path_mm,
                                    correlated_scattering
                                        .final_geom_path_mm);
                                sycl::atomic_ref<
                                    std::uint64_t,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    minibeam_event_counts_device
                                        [minibeam_urban_numeric_guard_slot])
                                    .fetch_add(1U);
                            }
                            if (cu_reason != 0) {
                                urban_record_first_failure(
                                    minibeam_event_counts_device, rng_history,
                                    beamline_step, 0U, cu_reason);
                                break;
                            }
                        }
                        // (fatal++ is done inside urban_record_first_failure.)
                        const auto geom_advance_mm = urban_v2_proposal
                            ? correlated_scattering.final_geom_path_mm
                            : transport_path;
                        const auto true_loss_path_mm = urban_v2_proposal
                            ? correlated_scattering.final_true_path_mm
                            : transport_path;
                        if ((kUrbanBeamlinePath || global_urban) && urban_v2_proposal) {
                            // Compete on the accepted TRUE path. Geometry/MSC
                            // can shorten a nuclear candidate, never advance it.
                            collision_in_step = rate_total > 0.0F &&
                                copper_nuclear_tau_active &&
                                true_loss_path_mm >= copper_nuclear_tau_remaining / rate_total;
                        }
                        position_x_mm += geom_advance_mm * direction_x +
                            correlated_scattering.displacement_mm.x;
                        position_y_mm += geom_advance_mm * direction_y +
                            correlated_scattering.displacement_mm.y;
                        position_z_mm += geom_advance_mm * direction_z +
                            correlated_scattering.displacement_mm.z;
                        const auto predictor_loss = stopping * true_loss_path_mm;
                        const auto midpoint_energy_u = sycl::fmax(
                            0.0F, (energy_MeV - 0.5F * predictor_loss) *
                                      inverse_mass_number);
                        const auto midpoint_stopping =
                            minibeam_linear_table(
                                minibeam_copper_sp_energies_device,
                                minibeam_copper_sp_values_device,
                                minibeam_copper_sp_count, midpoint_energy_u);
                        const auto mean_loss = midpoint_stopping * true_loss_path_mm;
                        float native_restricted_mean = -1.0F;
                        if (copper_primary_native_mean) {
                            const auto context = copper_loss_context;
                            if (!context.valid()) {
                                urban_record_first_failure(minibeam_event_counts_device,
                                    rng_history, beamline_step, 0U, 8);
                                break;
                            }
                            const auto range = context.range(energy_MeV);
                            if (true_loss_path_mm >= range) {
                                // G4VEnergyLossProcess stops without fluctuating
                                // when the accepted true path reaches its range.
                                removed_energy += energy_MeV;
                                energy_MeV = 0.0F;
                                break;
                            }
                            native_restricted_mean = context.dedx(energy_MeV) * true_loss_path_mm;
                            if (native_restricted_mean > 0.02F * energy_MeV) {
                                const auto decrement = energy_MeV -
                                    context.energy(range - true_loss_path_mm);
                                if (decrement > 0.0F) native_restricted_mean = decrement;
                            }
                        }
                        auto proposed_loss = mean_loss;
                        // The native proton fluctuation below replaces this
                        // Gaussian loss (or stops the run); its counter-based
                        // draws 48/49 are not consumed by anything else.
                        if (minibeam_copper_enable_energy_straggling &&
                            !(copper_proton_native_fluctuation && primary_atomic_number == 1 &&
                              primary_mass_number == 1)) {
                            constexpr float copper_z_over_a_rel_water =
                                (29.0F / 63.546F) / 0.55509F;
                            const auto effective_charge =
                                ion_effective_charge_device(primary_atomic_number, midpoint_energy_u);
                            const auto variance =
                                condensed_total_loss_variance_MeV2_device(
                                    midpoint_energy_u, primary_mass_number, effective_charge,
                                    true_loss_path_mm, minibeam_copper_density,
                                    copper_z_over_a_rel_water);
                            const auto gaussian_u0 = sycl::fmax(
                                rng::uniform01(spot_seed, rng_history,
                                               beamline_step, 48),
                                1.0e-12F);
                            const auto gaussian_u1 = rng::uniform01(
                                spot_seed, rng_history, beamline_step, 49);
                            constexpr float two_pi = 6.2831853071795864769F;
                            const auto gaussian =
                                sycl::sqrt(-2.0F * sycl::log(gaussian_u0)) *
                                sycl::cos(two_pi * gaussian_u1);
                            proposed_loss = sycl::fmax(
                                0.0F, mean_loss + minibeam_copper_straggling_scale *
                                    sycl::sqrt(sycl::fmax(0.0F, variance)) * gaussian);
                        }
                        if (minibeam_copper_enable_energy_straggling) {
                            if (copper_proton_native_fluctuation && primary_atomic_number == 1 && primary_mass_number == 1) {
                                std::uint64_t draw_index = 0;
                                auto uniform = [&]() { return (rng::random_u32(
                                    spot_seed, rng_history, (std::uint64_t(beamline_step) << 32) | draw_index++, 124) >> 8) * 0x1p-24f; };
                                const auto draw = copper_proton_total_fluctuation(
                                    energy_MeV, primary_rest_mass_MeV, mean_loss, true_loss_path_mm, minibeam_copper_density, uniform, native_restricted_mean);
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device, sycl::access::address_space::global_space>(
                                    minibeam_event_counts_device[draw.valid ? copper_proton_primary_loss_slot : copper_proton_loss_failure_slot]).fetch_add(1U);
                                if (!draw.valid) break; // host rejects the entire run
                                proposed_loss = draw.loss;
                            }
                        }
                        if (proposed_loss >=
                            energy_MeV - energy_cutoff_MeV) {
                            removed_energy += energy_MeV;
                            energy_MeV = 0.0F;
                            break;
                        }
                        const auto loss = proposed_loss;
                        energy_MeV -= loss;
                        removed_energy += loss;
                        if (rate_total > 0.0F && copper_nuclear_tau_active) {
                            copper_nuclear_tau_remaining = (kUrbanBeamlinePath || global_urban)
                                ? sycl::fmax(0.0F,
                                    copper_nuclear_tau_remaining / rate_total - true_loss_path_mm) * rate_total
                                : sycl::fmax(0.0F, copper_nuclear_tau_remaining -
                                    rate_total * true_loss_path_mm);
                        }
                        if (in_copper) {
                            beamline_ever_in_copper_hist = true;
                            beamline_cu_true_path_hist_mm += true_loss_path_mm;
                            {
                                ++beamline_cu_steps_hist;
                                const auto& sc = correlated_scattering;
                                beamline_g_sum_hist_mm +=
                                    sc.proposal_valid ? sc.final_geom_path_mm
                                                      : transport_path;
                                beamline_t_sum_hist_mm +=
                                    sc.proposal_valid ? sc.final_true_path_mm
                                                      : transport_path;
                                beamline_delta_sum_hist_mm +=
                                    sc.proposal_valid ? sc.stable_delta_mm
                                                      : 0.0F;
                                beamline_raw2_sum_hist_mm2 +=
                                    sc.raw_displacement_r_mm *
                                    sc.raw_displacement_r_mm;
                                beamline_acc2_sum_hist_mm2 +=
                                    sc.displacement_mm.x *
                                        sc.displacement_mm.x +
                                    sc.displacement_mm.y *
                                        sc.displacement_mm.y +
                                    sc.displacement_mm.z *
                                        sc.displacement_mm.z;
                                switch (sc.displacement_branch) {
                                    case 1: ++beamline_disp_below_hist; break;
                                    case 2: ++beamline_disp_accept_hist; break;
                                    case 3: ++beamline_disp_reduce_hist; break;
                                    case 4: ++beamline_disp_cancel_hist; break;
                                    default: break;
                                }
                                if (sc.cth_rounded_to_one) {
                                    ++beamline_cth_one_hist;
                                }
                                switch (sc.limit_reason) {
                                    case 1: ++beamline_limit_user_hist; break;
                                    case 2: ++beamline_limit_msc_hist; break;
                                    case 3: ++beamline_limit_geom_hist; break;
                                    case 4: ++beamline_limit_range_hist; break;
                                    default: break;
                                }
                            }
                        }
                        const auto collision_energy_u = energy_MeV * inverse_mass_number;
                        const auto elastic_index = minibeam_elastic_nearest(
                            minibeam_copper_elastic_device,
                            minibeam_copper_elastic_count, collision_energy_u);
                        const auto elastic_rate = minibeam_copper_enable_elastic
                            ? minibeam_copper_elastic_device[
                                  elastic_index].macroscopic_rate_per_mm
                            : 0.0F;
                        const auto inelastic_rate = minibeam_copper_enable_inelastic
                            ? minibeam_linear_table(
                                  minibeam_copper_xs_energies_device,
                                  minibeam_copper_xs_values_device,
                                  minibeam_copper_xs_count, collision_energy_u)
                            : 0.0F;
                        const auto total_nuclear_rate = elastic_rate + inelastic_rate;
                        if (minibeam_copper_enable_mcs) {
                            direction_x = correlated_scattering.direction.x;
                            direction_y = correlated_scattering.direction.y;
                            direction_z = correlated_scattering.direction.z;
                        }
                        if (collision_in_step && total_nuclear_rate > 0.0F) {
                            copper_nuclear_tau_active = false;
                            const auto choose = rng::uniform01(
                                spot_seed, rng_history, beamline_step, 61) *
                                total_nuclear_rate;
                            if (choose < elastic_rate) {
                                if (copper_diagnostic_device) ++copper_history_diag.copper_elastic;
                                minibeam_primary_elastic = true;
                                const auto sample_index = minibeam_elastic_local_sample(
                                    minibeam_copper_elastic_device,
                                    minibeam_copper_elastic_count, collision_energy_u,
                                    rng::uniform01(spot_seed, rng_history,
                                                   beamline_step, 63));
                                const auto& sample = minibeam_copper_elastic_device[
                                    sample_index];
                                const auto outcome = elastic_two_body(
                                    energy_MeV, direction_x, direction_y, direction_z,
                                    1.0 - 2.0 * sample.transfer_fraction,
                                    rng::uniform01(spot_seed, rng_history,
                                                   beamline_step, 62),
                                    primary_rest_mass_MeV, sample.target_mass_MeV);
                                removed_energy += outcome.recoil_ke_MeV;
                                energy_MeV = outcome.projectile_ke_MeV;
                                direction_x = outcome.proj_dir_x;
                                direction_y = outcome.proj_dir_y;
                                direction_z = outcome.proj_dir_z;
                            } else {
                                if (copper_diagnostic_device) ++copper_history_diag.copper_inelastic;
                                sycl::atomic_ref<
                                    std::uint64_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    nuclear_count(minibeam_event_counts_device[0]);
                                nuclear_count.fetch_add(1U);
                                const auto lookup = cinel03_lookup_event_device(
                                    minibeam_copper_nodes_device,
                                    minibeam_copper_node_count,
                                    minibeam_copper_offsets_device,
                                    minibeam_copper_indices_device,
                                    minibeam_copper_event_count,
                                    primary_atomic_number, primary_mass_number,
                                    29, collision_energy_u,
                                    rng::uniform01(spot_seed, rng_history,
                                                   beamline_step, 63),
                                    rng::uniform01(spot_seed, rng_history,
                                                   beamline_step, 64));
                                float queued_energy = 0.0F;
                                if (lookup.status == Cinel03LookupStatus::Hit) {
                                    const auto& event = minibeam_copper_interactions_device[
                                        lookup.event_index];
                                    sycl::atomic_ref<
                                        std::uint64_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        generated_count(minibeam_event_counts_device[1]);
                                    generated_count.fetch_add(event.direct_product_count);
                                    const auto phi = 6.2831853071795864769F *
                                        rng::uniform01(spot_seed, rng_history,
                                                       beamline_step, 65);
                                    const auto event_cos = sycl::cos(phi);
                                    const auto event_sin = sycl::sin(phi);
                                    for (std::uint32_t product_index = 0;
                                         product_index < event.direct_product_count;
                                         ++product_index) {
                                        const auto flat_index =
                                            event.product_offset + product_index;
                                        if (flat_index >= minibeam_copper_product_count) break;
                                        const auto& product =
                                            minibeam_copper_products_device[flat_index];
                                        if ((product.role != 0 && product.role != 1) ||
                                            product.z <= 0 || product.a <= 0 ||
                                            product.kinetic_energy_MeV <= energy_cutoff_MeV) {
                                            continue;
                                        }
                                        if constexpr (kQueueCopperChildren) {
                                            const auto slot=sycl::atomic_ref<std::uint32_t,
                                                sycl::memory_order::relaxed,sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>(copper_primary_child_counts[0]).fetch_add(1U);
                                            if(slot<copper_primary_child_capacity) {
                                                copper_primary_child_tasks[slot]={global_history,rng_history,spot_seed,beamline_step,
                                                    flat_index,product_index,position_x_mm,position_y_mm,position_z_mm,
                                                    direction_x,direction_y,direction_z,event_cos,event_sin};
                                            }
                                        } else {
#include "detail/sycl_copper_primary_child.inc"
                                        }
                                    }
                                }
                                removed_energy = sycl::fmax(
                                    removed_energy,
                                    incident_beamline_energy - queued_energy -
                                    (beamline_air_loss_MeV-air_loss_at_block_entry));
                                energy_MeV = 0.0F;
                                break;
                            }
                        }
                        ++beamline_step;
                    }
                    if (energy_MeV > 0.0F) {
                        removed_energy = sycl::fmax(
                            0.0F, incident_beamline_energy - energy_MeV -
                            (beamline_air_loss_MeV-air_loss_at_block_entry));
                    }
                    beamline_removed_device[global_history] =
                        minibeam_air_urban && !beamline_ever_in_copper_hist ? 0.0F :
                        sycl::fmax(0.0F, removed_energy);
                }
                if (copper_diagnostic_device) {
                    copper_history_diag.terminal_x=position_x_mm;
                    copper_history_diag.terminal_y=position_y_mm;
                    copper_history_diag.terminal_z=position_z_mm;
                    copper_history_diag.terminal_energy=energy_MeV;
                    copper_history_diag.outcome=copper_history_diag.copper_inelastic ? 2U :
                        energy_MeV<=energy_cutoff_MeV ? 1U : direction_z<=1.0e-8F ? 3U :
                        copper_history_diag.loop_iterations>=100000 ? 4U :
                        position_z_mm>=minibeam_block_exit_z-1.0e-6F ? 5U : 6U;
                    copper_diagnostic_device[global_history]=copper_history_diag;
                }
                // Snapshot only: no RNG draws or transport-state changes.
                auto record_primary_phase = [&]() {
                    MinibeamPhaseSpaceRecord record{};
                    record.history = global_history;
                    record.slit = minibeam_source_slit;
                    record.copper_elastic = minibeam_primary_elastic ? 1U : 0U;
                    record.valid = 1U;
                    // Record the initial ray and actual copper traversal separately.
                    record.initial_ray_hits_copper =
                        minibeam_hits_copper ? 1U : 0U;
                    record.ever_in_copper =
                        beamline_ever_in_copper_hist ? 1U : 0U;
                    record.cumulative_cu_true_path_mm =
                        beamline_cu_true_path_hist_mm;
                    record.cu_steps = beamline_cu_steps_hist;
                    record.disp_below_min = beamline_disp_below_hist;
                    record.disp_accept = beamline_disp_accept_hist;
                    record.disp_reduce = beamline_disp_reduce_hist;
                    record.disp_cancel = beamline_disp_cancel_hist;
                    record.cth_eq_one = beamline_cth_one_hist;
                    record.cu_g_sum_mm = beamline_g_sum_hist_mm;
                    record.cu_t_sum_mm = beamline_t_sum_hist_mm;
                    record.cu_delta_sum_mm = beamline_delta_sum_hist_mm;
                    record.cu_raw_disp_sum2_mm2 = beamline_raw2_sum_hist_mm2;
                    record.cu_acc_disp_sum2_mm2 = beamline_acc2_sum_hist_mm2;
                    record.limit_user = beamline_limit_user_hist;
                    record.limit_msc = beamline_limit_msc_hist;
                    record.limit_geom = beamline_limit_geom_hist;
                    record.limit_range = beamline_limit_range_hist;
                    record.kinetic_energy_MeV = energy_MeV;
                    record.x_mm = position_x_mm;
                    record.y_mm = position_y_mm;
                    record.direction_x = direction_x;
                    record.direction_y = direction_y;
                    record.direction_z = direction_z;
                    minibeam_phase_space_device[global_history] = record;
                };
                if (phase_at_copper_exit && minibeam_phase_space_device != nullptr &&
                    energy_MeV > energy_cutoff_MeV && direction_z > 1.0e-8F &&
                    sycl::fabs(position_z_mm-minibeam_block_exit_z) < 1.0e-3F) {
                    record_primary_phase();
                }
                if (minibeam_air_urban && energy_MeV>energy_cutoff_MeV) {
                    if (!advance_air_to_plane(0.0F)) energy_MeV=0.0F;
                }
                if (enable_minibeam) {
                    beamline_air_loss_device[global_history] =
                        beamline_air_loss_MeV;
                }
                if (enable_minibeam && energy_MeV > energy_cutoff_MeV) {
                    beamline_primary_survivor_device[global_history] = energy_MeV;
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
                    if ((enable_voxel_scoring)) {
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
                    if (position_z_mm < 0.0F && sycl::fabs(direction_z) > 1.0e-8F) {
                        const auto t_plane = -position_z_mm / direction_z;
                        position_x_mm += t_plane * direction_x;
                        position_y_mm += t_plane * direction_y;
                        position_z_mm = 0.0F;
                    }
                }

                if (!phase_at_copper_exit && minibeam_phase_space_device != nullptr &&
                    energy_MeV > energy_cutoff_MeV &&
                    position_z_mm >= 0.0F && position_z_mm < phantom_length_mm) {
                    record_primary_phase();
                }

                entry = {rng_history, spot_seed, global_history, spot_initial_energy_MeV, energy_MeV, sampled_initial_energy_MeVu, position_x_mm, position_y_mm, position_z_mm, direction_x, direction_y, direction_z};
                if(copper_capture_entry_device)copper_capture_entry_device[global_history]=entry;
                }

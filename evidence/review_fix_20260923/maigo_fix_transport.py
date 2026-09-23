from pathlib import Path
root=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
p=root/'src/transport_sycl.cpp'
s=p.read_text()
start=s.index('                    const float macro_geom_ceiling_mm = step_mm;')
end=s.index('                    if (enable_energy_straggling && enable_step_stable_straggling)',start)
s=s[:start]+'''                    const float macro_geom_ceiling_mm = step_mm;
                    float urban_true_path_mm = step_mm;
                    float urban_geom_path_mm = step_mm;
                    float accepted_loss_path_mm = step_mm;
                    float urban_macro_range_mm = 0.0F;
                    bool urban_preflight_valid = false;
                    bool urban_range_limited = false;
                    UrbanV2LossTable water_urban_preflight_table{};
                    CorrelatedScatteringStep water_urban_accepted_step{};
                    auto water_urban_proposed_state = water_urban_state;
                    if (enable_minibeam && minibeam_water_primary_urban_v2 &&
                        (kProductionPrimaryPath || enable_multiple_scattering) &&
                        !in_ct && !in_insert && slab_layer_count == 0) {
                        water_urban_preflight_table = UrbanV2LossTable{
                            minibeam_water_urban_loss_e_device,
                            minibeam_water_urban_loss_r_device,
                            minibeam_water_urban_loss_d_device,
                            static_cast<int>(minibeam_water_urban_loss_count)};
                        // One accepted Urban step is one actual transport/loss
                        // step. Sample once using the pre-step energy, cache the
                        // result, then commit it after loss succeeds. Subsequent
                        // steps use the actual post-loss energy, including
                        // fluctuations, rather than a residual-range predictor.
                        const float requested_geom = sycl::fmin(
                            minibeam_water_primary_urban_max_step_mm, step_mm);
                        water_urban_accepted_step =
                            water_urban_v2_propose_and_sample(
                                Direction3F{direction_x, direction_y, direction_z},
                                energy_MeV, 6, 12,
                                minibeam_water_primary_urban_max_step_mm,
                                requested_geom, position_x_mm, position_y_mm,
                                position_z_mm, direction_x, direction_y, direction_z,
                                voxel_min_x_mm, voxel_max_x_mm, voxel_min_y_mm,
                                voxel_max_y_mm, 0.0F, phantom_length_mm,
                                !water_urban_seen_segment,
                                water_urban_proposed_state,
                                water_urban_preflight_table,
                                minibeam_water_urban_zeff_f,
                                minibeam_water_urban_radlen_mm_f, 1.0F,
                                spot_seed, rng_history,
                                static_cast<std::uint64_t>(steps), 0U, 70U);
                        int reason = urban_step_field_reason(
                            water_urban_accepted_step);
                        bool dust_complete = false;
                        if (reason == 0) {
                            reason = urban_step_progress_reason(
                                water_urban_accepted_step, 0.0F, step_mm,
                                requested_geom, dust_complete);
                        }
                        if (reason != 0 || dust_complete) {
                            urban_record_first_failure(
                                minibeam_event_counts_device, rng_history,
                                static_cast<std::uint64_t>(steps), 0U,
                                reason != 0 ? reason : 6);
                            break; // fail before any loss/scoring; never use the old macro step
                        }
                        urban_macro_range_mm =
                            water_urban_accepted_step.current_range_mm;
                        urban_true_path_mm =
                            water_urban_accepted_step.final_true_path_mm;
                        urban_geom_path_mm =
                            water_urban_accepted_step.final_geom_path_mm;
                        // This flag means the proposal reached its range limit.
                        // Actual particle termination is governed by the sampled
                        // post-step energy and the existing cutoff accounting.
                        urban_range_limited = water_urban_accepted_step.range_terminal;
                        urban_preflight_valid = true;
                        accepted_loss_path_mm = urban_true_path_mm;
                        step_mm = urban_geom_path_mm;
                    }
''' + s[end:]
anchor=s.index('                            // Table-driven Geant4-11.3.2 Urban for primary')
start=s.index('                                const float remaining_range =',anchor)
end=s.index('                                // Fix B5/C2.3:',start)
s=s[:start]+'''                                const auto seg_e = energy_MeV;
                                const auto urban_scatter = water_urban_accepted_step;
                                water_urban_state = water_urban_proposed_state;
''' +s[end:]
start=s.index('                                            const auto energy_at_crossing =',anchor)
end=s.index('                                            candidate_record.history =',start)
s=s[:start]+'''                                            const float crossing_fraction = sycl::clamp(
                                                (plane_depth_mm - seg_start_z) /
                                                    (seg_end_z - seg_start_z),
                                                0.0F, 1.0F);
                                            const auto energy_at_crossing = sycl::fmax(
                                                0.0F, energy_MeV -
                                                    crossing_fraction * deposited_MeV);
''' +s[end:]
s=s.replace('urban_range_terminal','urban_range_limited')
p.write_text(s)
print('one accepted Urban proposal per actual loss/transport step; cached sampling and actual crossing energy applied')

// Native Cu fragment step shared by primary-born and cascade transport.
// Include contract: the caller binds f_* aliases inside a fragment loop.
// Mutable: energy, position, direction, path/rate/collision, optical depth,
// Urban state and boundary flag. Identity/material inputs are read-only.
// f_terminal_tau is null for direct children, or points to the omitted
// inelastic optical-depth ledger at the cascade generation limit.
// A failure increments a fatal counter, then break exits the caller's track
// loop. The host rejects that dose; the fragment must not resume afterward.
const auto f_fail = [&](unsigned reason) {
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
        sycl::memory_scope::device, sycl::access::address_space::global_space>(
        minibeam_event_counts_device[copper_ion_transport_failure_slot]).fetch_add(1U);
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
        sycl::memory_scope::device, sycl::access::address_space::global_space>
        claim(minibeam_event_counts_device[copper_ion_first_failure_slot]);
    if (claim.fetch_add(1ULL) == 0ULL) {
        auto* first = minibeam_event_counts_device + copper_ion_first_failure_slot;
        first[copper_failure_field::reason] = reason;
        first[copper_failure_field::atomic_number] = f_charge;
        first[copper_failure_field::mass_number] = f_a;
        first[copper_failure_field::energy] = sycl::bit_cast<std::uint32_t>(f_energy);
        first[copper_failure_field::step] = f_step;
        first[copper_failure_field::path] = sycl::bit_cast<std::uint32_t>(f_path);
        first[copper_failure_field::optical_depth] = sycl::bit_cast<std::uint32_t>(f_tau);
    }

};
const auto f_count = [&](std::size_t slot) {
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
        sycl::memory_scope::device, sycl::access::address_space::global_space>(
        minibeam_event_counts_device[slot]).fetch_add(1U);
};
if (!f_in_copper) {
    const auto response=beamline_em.at(1,beamline_em.projectile(f_charge,f_a));
    if(!response.valid()){f_fail(4);break;}
    f_path=sycl::fmin(f_path,response.range(f_energy));
    const auto mean=response.mean_loss(f_energy,f_path);
    std::uint64_t draw_index=0;
    auto uniform=[&](){return (rng::random_u32(f_seed ^ 0xA17B09D5C628E341ULL,f_stream,
        (std::uint64_t(f_step)<<32)|draw_index++,124)>>8)*0x1p-24f;};
    const auto ion=beamline_em.projectile(f_charge,f_a);
    const auto draw=sample_beamline_fluctuation(f_energy,beamline_em.species[ion].mass_MeV,
        f_charge,f_a,mean,f_path,response,minibeam_copper_fragment_enable_energy_straggling,uniform);
    if(!draw.valid){f_fail(5);break;}
    f_energy=sycl::fmax(0.f,f_energy-draw.loss);
    f_x += f_path * f_direction.x;
    f_y += f_path * f_direction.y;
    f_z += f_path * f_direction.z;
    f_at_boundary = true;
} else {
    const auto f_species = copper_fragment_urban_device.projectile(f_charge, f_a);
    UrbanLossContext f_context = copper_fragment_urban_device.at(
        copper_fragment_urban_device.material(-2, minibeam_copper_density), f_species, f_energy);
    f_context.beamline_point=beamline_em.at(0,beamline_em.projectile(f_charge,f_a));
    if (f_species < 0 || !f_context.valid()) { f_fail(1); break; }
    const auto f_mass = copper_fragment_urban_device.species[f_species].mass_mev;
    const CopperElasticRecord* f_bank = nullptr;
    std::uint32_t f_bank_size = 0;
    auto f_elastic_rate = 0.0F;
    if (minibeam_copper_enable_elastic) {
        f_bank = copper_fragment_elastic_device + copper_fragment_elastic_offsets[f_species];
        f_bank_size = copper_fragment_elastic_offsets[f_species + 1] -
            copper_fragment_elastic_offsets[f_species];
        const auto e = f_energy / static_cast<float>(f_a);
        if (e < f_bank[0].energy_MeV_per_u || e > f_bank[f_bank_size-1].energy_MeV_per_u) {
            f_fail(2); break;
        }
        f_elastic_rate = f_bank[minibeam_elastic_nearest(f_bank, f_bank_size, e)].macroscopic_rate_per_mm;
    }
    const auto f_native_inelastic_rate = minibeam_copper_ion_inelastic_rate(
        minibeam_copper_ion_xs_device, minibeam_copper_ion_xs_present_device,
        minibeam_copper_ion_xs_grid_size, minibeam_copper_ion_xs_minimum_energy,
        minibeam_copper_ion_xs_inverse_step, f_energy, f_charge, f_a);
    // Generation-limited cascades omit inelastic reactions and retain their
    // omitted optical depth; direct primary children have no terminal ledger.
    f_rate = f_elastic_rate + (f_terminal_tau ? 0.0F : f_native_inelastic_rate);
    if (f_rate > 0.0F && !f_tau_active) {
        f_tau = -sycl::log(sycl::fmax(1.0e-7F,
            rng::uniform01(f_seed, f_stream, f_step, 2)));
        f_tau_active = true;
    }
    UrbanV2GeomCtx f_geom{};
    f_geom.cos_a = minibeam_cos; f_geom.sin_a = minibeam_sin;
    f_geom.radius_mm = minibeam_block_radius;
    f_geom.slit_count = minibeam_slit_count;
    f_geom.slit_width_mm = minibeam_slit_width;
    f_geom.slit_pitch_mm = minibeam_slit_pitch;
    f_geom.slit_half_len_mm = 0.5F * minibeam_slit_length;
    f_geom.slit_offset_mm = minibeam_slit_offset;
    f_geom.block_entrance_z_mm = minibeam_block_center_z - 0.5F * minibeam_block_thickness;
    f_geom.block_exit_z_mm = minibeam_block_center_z + 0.5F * minibeam_block_thickness;
    const auto f_scattering = copper_urban_v2_propose_and_sample(
        f_direction, f_energy, f_charge, f_a,
        f_rate > 0 ? sycl::fmin(minibeam_copper_max_step, f_tau / f_rate)
                   : minibeam_copper_max_step,
        f_path, f_x, f_y, f_z, f_direction.x, f_direction.y, f_direction.z,
        f_geom, f_at_boundary, f_urban_state, f_context, minibeam_copper_density,
        minibeam_copper_radiation_length, minibeam_copper_fragment_mcs_scale,
        f_seed, f_stream, f_step, 0U, 50, f_mass);
    if (!f_scattering.proposal_valid || f_scattering.numeric_guard_tripped ||
        f_scattering.limit_reason < 0 || urban_step_field_reason(f_scattering) != 0) {
        f_fail(!f_scattering.proposal_valid || f_scattering.limit_reason < 0
            ? 1000 - f_scattering.limit_reason
            : f_scattering.numeric_guard_tripped ? 200 : 100 + urban_step_field_reason(f_scattering));
        break;
    }
    f_count(copper_fragment_urban_slot);
    f_at_boundary = f_scattering.boundary_crossed;
    f_path = f_scattering.final_true_path_mm;
    f_collision = f_rate > 0 && f_tau_active && f_path >= f_tau / f_rate;
    f_x += f_scattering.final_geom_path_mm * f_direction.x + f_scattering.displacement_mm.x;
    f_y += f_scattering.final_geom_path_mm * f_direction.y + f_scattering.displacement_mm.y;
    f_z += f_scattering.final_geom_path_mm * f_direction.z + f_scattering.displacement_mm.z;
    if (f_terminal_tau) *f_terminal_tau += f_native_inelastic_rate * f_path;
    const auto f_range = f_scattering.current_range_mm;
    if (f_path >= f_range) { f_energy = 0; break; }
    auto f_mean = f_context.dedx(f_energy) * f_path;
    if (f_mean > 0.02F * f_energy) f_mean = f_energy - f_context.energy(f_range - f_path);
    std::uint64_t f_draw_index = 0;
    auto f_uniform = [&]() { return (rng::copper_loss_u32(f_seed,f_stream,f_step,f_draw_index++) >> 8) * 0x1p-24f; };
    const auto f_draw=sample_beamline_fluctuation(f_energy,f_mass,f_charge,f_a,f_mean,f_path,
        f_context.beamline_point,minibeam_copper_fragment_enable_energy_straggling,f_uniform);
    f_count(f_draw.valid ? copper_proton_fragment_loss_slot : copper_proton_loss_failure_slot);
    if (!f_draw.valid) break;
    if (f_draw.loss >= f_energy - energy_cutoff_MeV) { f_energy = 0; break; }
    f_energy -= f_draw.loss;
    f_direction = f_scattering.direction;
    if (!f_collision && f_tau_active) {
        const auto remaining_tau = f_tau - f_rate * f_path;
        // Division decided that the collision is still ahead. A rounded
        // rate*path can nevertheless equal tau; preserve that positive
        // distance instead of proposing a zero-length next Urban step.
        f_tau = remaining_tau > 0.0F || !(f_rate > 0.0F)
            ? remaining_tau : (f_tau / f_rate - f_path) * f_rate;
    }
    if (f_collision) {
        f_tau_active = false;
        const auto e = f_energy / static_cast<float>(f_a);
        if (minibeam_copper_enable_elastic) {
            if (e < f_bank[0].energy_MeV_per_u || e > f_bank[f_bank_size-1].energy_MeV_per_u) {
                f_fail(3); break;
            }
            f_elastic_rate = f_bank[minibeam_elastic_nearest(f_bank, f_bank_size, e)].macroscopic_rate_per_mm;
        }
        const auto inelastic = f_terminal_tau ? 0.0F : minibeam_copper_ion_inelastic_rate(
            minibeam_copper_ion_xs_device, minibeam_copper_ion_xs_present_device,
            minibeam_copper_ion_xs_grid_size, minibeam_copper_ion_xs_minimum_energy,
            minibeam_copper_ion_xs_inverse_step, f_energy, f_charge, f_a);
        if (!(f_elastic_rate + inelastic > 0)) {
            // The flight was proposed at the pre-loss energy. Crossing a
            // physical reaction threshold makes it a null collision, not
            // missing package coverage; restart the optical-depth clock.
            f_collision = false;
            f_count(copper_fragment_post_loss_null_slot);
        } else if (rng::uniform01(f_seed, f_stream, f_step, 61) * (f_elastic_rate + inelastic) < f_elastic_rate) {
            const auto& sample = f_bank[minibeam_elastic_local_sample(f_bank, f_bank_size, e,
                rng::uniform01(f_seed, f_stream, f_step, 63))];
            const auto outcome = elastic_two_body(f_energy,
                f_direction.x, f_direction.y, f_direction.z, 1.0 - 2.0 * sample.transfer_fraction,
                rng::uniform01(f_seed, f_stream, f_step, 62), f_mass, sample.target_mass_MeV);
            f_energy = outcome.projectile_ke_MeV;
            f_direction = {outcome.proj_dir_x, outcome.proj_dir_y, outcome.proj_dir_z};
            f_collision = false; // continue elastic survivors; only inelastic spawns a cascade
            f_count(copper_fragment_elastic_slot);
        }
    }
}

// Original primary-born Cu charged transport body; shared by inline and queued routes.
const auto local_direction =
    rotate_cinel03_event_azimuth(
        product.local_direction_x,
        product.local_direction_y,
        product.local_direction_z,
        event_cos, event_sin);
auto child_direction = rotate_local_direction(
    local_direction.x, local_direction.y,
    local_direction.z,
    Direction3F{direction_x, direction_y,
                direction_z});
auto child_energy = product.kinetic_energy_MeV;
auto child_x = position_x_mm;
auto child_y = position_y_mm;
auto child_z = position_z_mm;
std::uint32_t child_step = 0;
float child_copper_segment_path = 0.0F;
float child_nuclear_tau_remaining = 0.0F;
bool child_nuclear_tau_active = false;
UrbanV2TrackState child_urban_state{};
bool child_at_boundary = true;
const auto child_species_category =
    product.z == 6 ? 0U :
    product.z == 5 ? 1U :
    product.z == 4 ? 2U :
    product.z == 3 ? 3U :
    product.z == 2 ? 4U :
    (product.z == 1 && product.a == 1) ? 5U :
    (product.z == 1 && product.a == 2) ? 6U :
    (product.z == 1 && product.a == 3) ? 7U : 8U;
const auto child_rng_stream =
    rng::event_product_stream(
        rng::primary_descendant_stream(spot_seed, rng_history), beamline_step,
        rng::branch_role_beamline_charged,
        product_index);
while (child_energy > energy_cutoff_MeV &&
       child_direction.z > 1.0e-8F &&
       child_z < block_exit_z - 1.0e-6F &&
       child_step < 100000U) {
    const auto nominal_child_path = sycl::fmin(
        minibeam_copper_max_step,
        (block_exit_z - child_z) /
            child_direction.z);
    const auto child_path =
        minibeam_copper_exact_material_boundaries
        ? minibeam_path_to_material_boundary(
              child_x, child_y,
              child_direction.x,
              child_direction.y,
              minibeam_cos, minibeam_sin,
              minibeam_block_radius,
              minibeam_slit_count,
              minibeam_slit_width,
              minibeam_slit_pitch,
              0.5F * minibeam_slit_length,
              minibeam_slit_offset,
              nominal_child_path)
        : nominal_child_path;
    const auto child_mid_x = child_x +
        0.5F * child_path * child_direction.x;
    const auto child_mid_y = child_y +
        0.5F * child_path * child_direction.y;
    const auto child_in_copper =
        minibeam_point_in_copper(
            child_mid_x, child_mid_y,
            minibeam_cos, minibeam_sin,
            minibeam_block_radius,
            minibeam_slit_count,
            minibeam_slit_width,
            minibeam_slit_pitch,
            0.5F * minibeam_slit_length,
            minibeam_slit_offset);
    auto child_inelastic_rate = 0.0F;
    auto child_collision_in_step = false;
    auto child_transport_path = child_path;
    if (child_in_copper && !copper_ion_native_transport) {
        child_inelastic_rate =
            minibeam_copper_ion_inelastic_rate(
                minibeam_copper_ion_xs_device,
                minibeam_copper_ion_xs_present_device,
                minibeam_copper_ion_xs_grid_size,
                minibeam_copper_ion_xs_minimum_energy,
                minibeam_copper_ion_xs_inverse_step,
                child_energy,
                product.z, product.a);
        if (child_inelastic_rate > 0.0F &&
            !child_nuclear_tau_active) {
            const auto optical_uniform =
                sycl::fmax(
                    1.0e-7F,
                    rng::uniform01(
                        spot_seed,
                        child_rng_stream,
                        child_step, 2));
            child_nuclear_tau_remaining =
                -sycl::log(optical_uniform);
            child_nuclear_tau_active = true;
        }
        child_collision_in_step =
            child_inelastic_rate > 0.0F &&
            child_nuclear_tau_active &&
            child_nuclear_tau_remaining <=
                child_inelastic_rate * child_path;
        if (child_collision_in_step) {
            child_transport_path =
                child_nuclear_tau_remaining /
                child_inelastic_rate;
        }
    }
    if (copper_ion_native_transport) {
        auto& f_energy = child_energy; auto& f_x = child_x; auto& f_y = child_y; auto& f_z = child_z;
        auto& f_direction = child_direction; auto& f_path = child_transport_path;
        auto& f_rate = child_inelastic_rate; auto& f_collision = child_collision_in_step;
        auto& f_tau = child_nuclear_tau_remaining; auto& f_tau_active = child_nuclear_tau_active;
        auto& f_urban_state = child_urban_state; auto& f_at_boundary = child_at_boundary;
        const auto f_charge = product.z; const auto f_a = product.a;
        const auto f_seed = spot_seed; const auto f_stream = child_rng_stream; const auto f_step = child_step;
        const auto f_in_copper = child_in_copper; constexpr bool f_terminal = false;
        float f_ignored_tau = 0;
#include "sycl_copper_ion_fragment_step.inc"
    } else {
    child_x += child_transport_path *
        child_direction.x;
    child_y += child_transport_path *
        child_direction.y;
    child_z += child_transport_path *
        child_direction.z;
    }
    if (child_in_copper) {
      if (!copper_ion_native_transport) {
        const auto child_stopping =
            minibeam_copper_ion_stopping(
                minibeam_copper_sp_energies_device,
                minibeam_copper_sp_values_device,
                minibeam_copper_sp_count,
                minibeam_copper_ion_sp_ratios_device,
                minibeam_copper_ion_sp_present_device,
                child_energy, product.z, product.a, primary_atomic_number);
        const auto predictor_child_loss =
            child_stopping * child_transport_path;
        const auto child_midpoint_energy =
            sycl::fmax(
                0.0F,
                child_energy - 0.5F *
                    predictor_child_loss);
        const auto child_midpoint_stopping =
            minibeam_copper_ion_stopping(
                minibeam_copper_sp_energies_device,
                minibeam_copper_sp_values_device,
                minibeam_copper_sp_count,
                minibeam_copper_ion_sp_ratios_device,
                minibeam_copper_ion_sp_present_device,
                child_midpoint_energy,
                product.z, product.a, primary_atomic_number);
        const auto mean_child_loss =
            child_midpoint_stopping *
            child_transport_path;
        auto proposed_child_loss = mean_child_loss;
        if (minibeam_copper_fragment_enable_energy_straggling) {
            constexpr float copper_z_over_a_rel_water =
                (29.0F / 63.546F) / 0.55509F;
            const auto child_energy_u =
                child_midpoint_energy /
                static_cast<float>(product.a);
            const auto effective_charge =
                ion_effective_charge_device(
                    product.z, child_energy_u);
            const auto variance =
                condensed_total_loss_variance_MeV2_device(
                    child_energy_u, product.a,
                    effective_charge,
                    child_transport_path,
                    minibeam_copper_density,
                    copper_z_over_a_rel_water);
            const auto gaussian_u0 = sycl::fmax(
                rng::uniform01(
                    spot_seed, child_rng_stream,
                    child_step, 10),
                1.0e-12F);
            const auto gaussian_u1 = rng::uniform01(
                spot_seed, child_rng_stream,
                child_step, 11);
            constexpr float two_pi =
                6.2831853071795864769F;
            const auto gaussian = sycl::sqrt(
                -2.0F * sycl::log(gaussian_u0)) *
                sycl::cos(two_pi * gaussian_u1);
            proposed_child_loss = sycl::fmax(
                0.0F,
                mean_child_loss +
                    minibeam_copper_fragment_straggling_scale *
                        sycl::sqrt(sycl::fmax(
                            0.0F, variance)) *
                        gaussian);
        }
        if (minibeam_copper_fragment_enable_energy_straggling) {
            if (copper_proton_native_fluctuation && product.z == 1 && product.a == 1) {
                std::uint64_t draw_index = 0;
                auto uniform = [&]() { return (rng::random_u32(
                    spot_seed, child_rng_stream, (std::uint64_t(child_step) << 32) | draw_index++, 124) >> 8) * 0x1p-24f; };
                const auto draw = copper_proton_total_fluctuation(
                    child_energy, product.rest_mass, mean_child_loss, child_transport_path, minibeam_copper_density, uniform);
                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                    sycl::memory_scope::device, sycl::access::address_space::global_space>(
                    minibeam_event_counts_device[draw.valid ? copper_proton_fragment_loss_slot : copper_proton_loss_failure_slot]).fetch_add(1U);
                if (!draw.valid) break; // host rejects the entire run
                proposed_child_loss = draw.loss;
            }
        }
        if (proposed_child_loss >=
            child_energy -
                energy_cutoff_MeV) {
            child_energy = 0.0F;
            break;
        }
        const auto child_loss =
            proposed_child_loss;
        const auto scatter_energy =
            child_energy - 0.5F * child_loss;
        child_energy -= child_loss;
        if (!child_collision_in_step &&
            child_inelastic_rate > 0.0F &&
            child_nuclear_tau_active) {
            child_nuclear_tau_remaining -=
                child_inelastic_rate *
                child_transport_path;
        }
        if (minibeam_copper_enable_mcs) {
            const auto previous_path =
                child_copper_segment_path;
            child_copper_segment_path +=
                child_transport_path;
            const auto total_rms =
                highland_projected_rms_angle_device(
                    scatter_energy, product.z,
                    product.a,
                    child_copper_segment_path,
                    minibeam_copper_density,
                    minibeam_copper_radiation_length);
            const auto previous_rms =
                highland_projected_rms_angle_device(
                    scatter_energy, product.z,
                    product.a, previous_path,
                    minibeam_copper_density,
                    minibeam_copper_radiation_length);
            child_direction = scatter_direction(
                child_direction,
                minibeam_copper_fragment_mcs_scale *
                    sycl::sqrt(sycl::fmax(
                        0.0F,
                        total_rms * total_rms -
                            previous_rms *
                                previous_rms)),
                spot_seed, child_rng_stream,
                child_step, 0);
        }
      }
        if (child_collision_in_step) {
            child_nuclear_tau_active = false;
            if (minibeam_copper_cascade_queue_device != nullptr &&
                product.a > 0) {
                sycl::atomic_ref<
                    std::uint64_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
                    reaction_count(
                        minibeam_event_counts_device[
                            minibeam_fragment_cascade_interactions_slot]);
                reaction_count.fetch_add(1U);
                sycl::atomic_ref<
                    std::uint64_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
                    generation_reaction_count(
                        minibeam_event_counts_device[
                            minibeam_fragment_generation_interaction_slot]);
                generation_reaction_count.fetch_add(1U);
                const auto fragment_lookup =
                    cinel03_lookup_event_device(
                        minibeam_copper_nodes_device,
                        minibeam_copper_node_count,
                        minibeam_copper_offsets_device,
                        minibeam_copper_indices_device,
                        minibeam_copper_event_count,
                        product.z, product.a, 29,
                        child_energy /
                            static_cast<float>(product.a),
                        rng::uniform01(
                            spot_seed, child_rng_stream,
                            child_step, 20),
                        rng::uniform01(
                            spot_seed, child_rng_stream,
                            child_step, 21));
                if (fragment_lookup.status ==
                    Cinel03LookupStatus::Hit) {
                    sycl::atomic_ref<
                        std::uint64_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        hit_count(
                            minibeam_event_counts_device[
                                minibeam_fragment_cascade_hits_slot]);
                    hit_count.fetch_add(1U);
                    sycl::atomic_ref<
                        std::uint64_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        generation_hit_count(
                            minibeam_event_counts_device[
                                minibeam_fragment_generation_hit_slot]);
                    generation_hit_count.fetch_add(1U);
                    const auto& fragment_event =
                        minibeam_copper_interactions_device[
                            fragment_lookup.event_index];
                    const auto local_keV =
                        static_cast<std::uint64_t>(
                            sycl::fmax(
                                0.0F,
                                fragment_event
                                    .process_local_deposit_MeV) *
                                1000.0F + 0.5F);
                    sycl::atomic_ref<
                        std::uint64_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        local_energy(
                            minibeam_event_counts_device[
                                minibeam_fragment_cascade_local_keV_slot]);
                    local_energy.fetch_add(local_keV);
                    const auto fragment_phi =
                        6.2831853071795864769F *
                        rng::uniform01(
                            spot_seed, child_rng_stream,
                            child_step, 22);
                    const auto fragment_cos =
                        sycl::cos(fragment_phi);
                    const auto fragment_sin =
                        sycl::sin(fragment_phi);
                    auto accounted_charged = 0.0F;
                    auto replay_product_energy = 0.0F;
                    auto replay_product_rest_mass = 0.0F;
                    std::int32_t replay_product_a = 0;
                    for (std::uint32_t cascade_index = 0;
                         cascade_index <
                             fragment_event.direct_product_count;
                         ++cascade_index) {
                        const auto cascade_flat =
                            fragment_event.product_offset +
                            cascade_index;
                        if (cascade_flat >=
                            minibeam_copper_product_count) break;
                        const auto& cascade_product =
                            minibeam_copper_products_device[
                                cascade_flat];
                        replay_product_energy += sycl::fmax(
                            0.0F,
                            cascade_product.kinetic_energy_MeV);
                        replay_product_rest_mass += sycl::fmax(
                            0.0F,
                            cascade_product.rest_mass);
                        replay_product_a += sycl::max(
                            0, static_cast<int>(cascade_product.a));
                        const bool supported_charged =
                            (cascade_product.role == 0 ||
                             cascade_product.role == 1) &&
                            cascade_product.z > 0 &&
                            cascade_product.a > 0;
                        const auto count_slot =
                            supported_charged
                                ? minibeam_fragment_cascade_charged_slot
                                : (cascade_product.z <= 0
                                       ? minibeam_fragment_cascade_neutral_slot
                                       : minibeam_fragment_cascade_unsupported_slot);
                        sycl::atomic_ref<
                            std::uint64_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            product_count_ref(
                                minibeam_event_counts_device[
                                    count_slot]);
                        product_count_ref.fetch_add(1U);
                        if (!supported_charged ||
                            cascade_product.kinetic_energy_MeV <=
                                energy_cutoff_MeV) {
                            continue;
                        }
                        accounted_charged +=
                            cascade_product.kinetic_energy_MeV;
                        const auto cascade_local =
                            rotate_cinel03_event_azimuth(
                                cascade_product.local_direction_x,
                                cascade_product.local_direction_y,
                                cascade_product.local_direction_z,
                                fragment_cos, fragment_sin);
                        const auto cascade_direction =
                            rotate_local_direction(
                                cascade_local.x,
                                cascade_local.y,
                                cascade_local.z,
                                child_direction);
                        sycl::atomic_ref<
                            std::uint32_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            cascade_queue_count(
                                *minibeam_copper_cascade_count_device);
                        const auto cascade_output =
                            cascade_queue_count.fetch_add(1U);
                        if (cascade_output < max_secondaries) {
                            SecondaryParticle cascade_child{};
                            cascade_child.z = cascade_product.z;
                            cascade_child.a = cascade_product.a;
                            cascade_child.energy_MeV =
                                cascade_product.kinetic_energy_MeV;
                            cascade_child.pos_x_mm = child_x;
                            cascade_child.pos_y_mm = child_y;
                            cascade_child.pos_z_mm = child_z;
                            cascade_child.dir_x =
                                cascade_direction.x;
                            cascade_child.dir_y =
                                cascade_direction.y;
                            cascade_child.dir_z =
                                cascade_direction.z;
                            cascade_child.weight = 1.0F;
                            cascade_child.parent_history =
                                global_history;
                            cascade_child.rng_stream =
                                rng::event_product_stream(
                                    child_rng_stream,
                                    child_step,
                                    rng::branch_role_cascade_charged,
                                    cascade_index);
                            // Copper generation is carried by
                            // this dedicated queue, not by the
                            // downstream water-generation field.
                            cascade_child.generation = 1U;
                            cascade_child.birth_region =
                                minibeam_birth_region_copper;
                            minibeam_copper_cascade_queue_device[
                                cascade_output] = cascade_child;
                        } else {
                            sycl::atomic_ref<
                                std::uint64_t,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                                overflow_count(
                                    minibeam_event_counts_device[
                                        minibeam_fragment_cascade_overflow_slot]);
                            overflow_count.fetch_add(1U);
                        }
                    }
                    const auto untracked_keV =
                        static_cast<std::uint64_t>(
                            sycl::fmax(
                                0.0F,
                                child_energy -
                                    fragment_event
                                        .process_local_deposit_MeV -
                                    accounted_charged) *
                                1000.0F + 0.5F);
                    sycl::atomic_ref<
                        std::uint64_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        untracked_energy(
                            minibeam_event_counts_device[
                                minibeam_fragment_cascade_untracked_keV_slot]);
                    untracked_energy.fetch_add(
                        untracked_keV);
                    const auto selected_input =
                        fragment_lookup.selected_energy_MeV_per_u *
                        static_cast<float>(product.a);
                    const auto replay_output =
                        sycl::fmax(0.0F, fragment_event.parent_energy_MeV) +
                        sycl::fmax(
                            0.0F,
                            fragment_event.process_local_deposit_MeV) +
                        replay_product_energy;
                    const auto add_energy_counter =
                        [&](const std::size_t slot,
                            const float value) {
                            sycl::atomic_ref<
                                std::uint64_t,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                                counter(
                                    minibeam_event_counts_device[
                                        slot]);
                            counter.fetch_add(
                                static_cast<std::uint64_t>(
                                    sycl::fmax(0.0F, value) *
                                        1000.0F + 0.5F));
                        };
                    add_energy_counter(
                        minibeam_fragment_actual_input_keV_slot,
                        child_energy);
                    add_energy_counter(
                        minibeam_fragment_selected_input_keV_slot,
                        selected_input);
                    add_energy_counter(
                        minibeam_fragment_replay_output_keV_slot,
                        replay_output);
                    add_energy_counter(
                        minibeam_fragment_selection_mismatch_keV_slot,
                        sycl::fabs(child_energy - selected_input));
                    add_energy_counter(
                        minibeam_fragment_closure_mismatch_keV_slot,
                        sycl::fabs(selected_input - replay_output));
                    const bool parent_survives =
                        fragment_event.parent_energy_MeV > 0.0F;
                    const auto copper_target_mass =
                        fragment_event.target_a == 63
                            ? 58603.7301743F
                            : (fragment_event.target_a == 65
                                   ? 60465.0342192F
                                   : sycl::fmax(
                                         0.0F,
                                         static_cast<float>(
                                             fragment_event.target_a) *
                                                 931.49410242F -
                                             29.0F * 0.51099895F));
                    const auto mass_energy_in =
                        selected_input +
                        fragment_event.parent_rest_mass +
                        copper_target_mass;
                    const auto mass_energy_out =
                        replay_output + replay_product_rest_mass +
                        (parent_survives
                             ? fragment_event.parent_rest_mass
                             : 0.0F);
                    add_energy_counter(
                        minibeam_fragment_mass_energy_mismatch_keV_slot,
                        sycl::fabs(mass_energy_in -
                                   mass_energy_out));
                    const auto input_a =
                        static_cast<int>(product.a) +
                        static_cast<int>(fragment_event.target_a);
                    const auto output_a =
                        replay_product_a +
                        (parent_survives
                             ? static_cast<int>(
                                   fragment_event.parent_a)
                             : 0);
                    if (input_a != output_a) {
                        sycl::atomic_ref<
                            std::uint64_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            baryon_miss(
                                minibeam_event_counts_device[
                                    minibeam_fragment_baryon_mismatch_slot]);
                        baryon_miss.fetch_add(1U);
                    }
                } else {
                    const auto miss_index =
                        static_cast<std::size_t>(
                            fragment_lookup.status) - 1U;
                    if (miss_index < 6U) {
                        sycl::atomic_ref<
                            std::uint64_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            miss_count(
                                minibeam_event_counts_device[
                                    minibeam_fragment_cascade_miss_slot +
                                    miss_index]);
                        miss_count.fetch_add(1U);
                    }
                    sycl::atomic_ref<
                        std::uint64_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        generation_miss_count(
                            minibeam_event_counts_device[
                                minibeam_fragment_generation_miss_slot]);
                    generation_miss_count.fetch_add(1U);
                    sycl::atomic_ref<
                        std::uint64_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        species_miss(
                            minibeam_event_counts_device[
                                minibeam_fragment_miss_species_slot +
                                child_species_category]);
                    species_miss.fetch_add(1U);
                    const auto miss_energy_u =
                        child_energy /
                        static_cast<float>(product.a);
                    const auto miss_energy_bin =
                        sycl::min(
                            15U,
                            static_cast<std::uint32_t>(
                                sycl::fmax(0.0F,
                                           miss_energy_u) /
                                25.0F));
                    sycl::atomic_ref<
                        std::uint64_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        energy_miss(
                            minibeam_event_counts_device[
                            minibeam_fragment_miss_energy_slot +
                            miss_energy_bin]);
                    energy_miss.fetch_add(1U);
                    if (minibeam_fragment_miss_joint_counts_device != nullptr &&
                        miss_index < MinibeamDiagnostics::fragment_miss_reason_count) {
                        const auto za_index =
                            product.z >= 0 && product.z <= 6 &&
                                    product.a >= 0 && product.a <= 12
                                ? static_cast<std::size_t>(product.z) * 13U +
                                      static_cast<std::size_t>(product.a)
                                : MinibeamDiagnostics::fragment_miss_za_count - 1U;
                        const auto joint_energy_bin = sycl::min(
                            127U,
                            static_cast<std::uint32_t>(
                                sycl::fmax(0.0F, miss_energy_u) / 5.0F));
                        const auto joint_index =
                            ((za_index *
                                  MinibeamDiagnostics::fragment_miss_reason_count +
                              miss_index) *
                                 MinibeamDiagnostics::fragment_miss_joint_energy_bin_count) +
                            joint_energy_bin;
                        const auto block_entry_z =
                            minibeam_block_center_z -
                            0.5F * minibeam_block_thickness;
                        const auto collision_depth_um =
                            static_cast<std::uint64_t>(sycl::fmax(
                                0.0F, child_z - block_entry_z) * 1000.0F + 0.5F);
                        const auto path_to_plane = child_direction.z > 1.0e-8F
                            ? sycl::fmax(0.0F,
                                  (block_exit_z - child_z) /
                                      child_direction.z)
                            : 0.0F;
                        const auto remaining_um =
                            static_cast<std::uint64_t>(
                                minibeam_straight_copper_path_to_plane(
                                    child_x, child_y,
                                    child_direction.x,
                                    child_direction.y,
                                    minibeam_cos, minibeam_sin,
                                    minibeam_block_radius,
                                    minibeam_slit_count,
                                    minibeam_slit_width,
                                    minibeam_slit_pitch,
                                    0.5F * minibeam_slit_length,
                                    minibeam_slit_offset,
                                    path_to_plane) *
                                    1000.0F +
                                0.5F);
                        const auto energy_keV =
                            static_cast<std::uint64_t>(
                                sycl::fmax(0.0F, child_energy) *
                                    1000.0F +
                                0.5F);
                        sycl::atomic_ref<std::uint64_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                minibeam_fragment_miss_joint_counts_device[joint_index])
                            .fetch_add(1U);
                        sycl::atomic_ref<std::uint64_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                minibeam_fragment_miss_joint_energy_device[joint_index])
                            .fetch_add(energy_keV);
                        sycl::atomic_ref<std::uint64_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                minibeam_fragment_miss_joint_depth_device[joint_index])
                            .fetch_add(collision_depth_um);
                        sycl::atomic_ref<std::uint64_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                minibeam_fragment_miss_joint_remaining_device[joint_index])
                            .fetch_add(remaining_um);
                    }
                }
            }
            if (minibeam_fragment_phase_space_device != nullptr) {
            sycl::atomic_ref<
                std::uint64_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                absorbed_count(
                    minibeam_event_counts_device[
                        21U + child_species_category]);
            absorbed_count.fetch_add(1U);
            const auto absorbed_energy_keV =
                static_cast<std::uint64_t>(
                    sycl::fmax(0.0F, child_energy) *
                        1000.0F + 0.5F);
            sycl::atomic_ref<
                std::uint64_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                absorbed_energy(
                    minibeam_event_counts_device[
                        30U + child_species_category]);
            absorbed_energy.fetch_add(
                absorbed_energy_keV);
            const auto path_to_downstream_plane =
                sycl::fmax(
                    0.0F,
                    (block_exit_z - child_z) /
                        child_direction.z);
            const auto straight_copper_path_um =
                static_cast<std::uint64_t>(
                    minibeam_straight_copper_path_to_plane(
                        child_x, child_y,
                        child_direction.x,
                        child_direction.y,
                        minibeam_cos, minibeam_sin,
                        minibeam_block_radius,
                        minibeam_slit_count,
                        minibeam_slit_width,
                        minibeam_slit_pitch,
                        0.5F * minibeam_slit_length,
                        minibeam_slit_offset,
                        path_to_downstream_plane) *
                        1000.0F + 0.5F);
            sycl::atomic_ref<
                std::uint64_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                straight_copper_path(
                    minibeam_event_counts_device[
                        39U + child_species_category]);
            straight_copper_path.fetch_add(
                straight_copper_path_um);
            }
            child_energy = 0.0F;
            break;
        }
    } else {
        child_copper_segment_path = 0.0F;
    }
    ++child_step;
}
if (child_energy <= energy_cutoff_MeV ||
    child_direction.z <= 1.0e-8F ||
    child_step == 100000U) continue;
const auto to_water = -child_z / child_direction.z;
if (to_water < 0.0F) continue;
child_x += to_water * child_direction.x;
child_y += to_water * child_direction.y;
if (secondary_queue_device != nullptr) {
    sycl::atomic_ref<
        std::uint32_t,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        count_ref(*secondary_count_device);
    const auto output = count_ref.fetch_add(1U);
    if (output < max_secondaries) {
        SecondaryParticle child{};
        child.z = product.z;
        child.a = product.a;
        child.energy_MeV = child_energy;
        child.pos_x_mm = child_x;
        child.pos_y_mm = child_y;
        child.pos_z_mm = 0.0F;
        child.dir_x = child_direction.x;
        child.dir_y = child_direction.y;
        child.dir_z = child_direction.z;
        child.weight = 1.0F;
        child.parent_history = global_history;
        child.rng_stream = child_rng_stream;
        child.birth_region =
            minibeam_birth_region_copper;
        secondary_queue_device[output] = child;
        if (minibeam_fragment_phase_space_device != nullptr) {
            sycl::atomic_ref<
                std::uint32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                phase_count(
                    *minibeam_fragment_phase_space_count_device);
            const auto phase_index =
                phase_count.fetch_add(1U);
            if (phase_index < max_secondaries) {
                MinibeamFragmentPhaseSpaceRecord record{};
                record.history = global_history;
                record.atomic_number = product.z;
                record.mass_number = product.a;
                record.kinetic_energy_MeV = child_energy;
                record.x_mm = child_x;
                record.y_mm = child_y;
                record.direction_x = child_direction.x;
                record.direction_y = child_direction.y;
                record.direction_z = child_direction.z;
                minibeam_fragment_phase_space_device[
                    phase_index] = record;
            }
        }
        queued_energy += child_energy;
        sycl::atomic_ref<
            std::uint64_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>
            survivor_count(
                minibeam_event_counts_device[2]);
        survivor_count.fetch_add(1U);
        sycl::atomic_ref<
            std::uint64_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>
            species_count(
                minibeam_event_counts_device[
                    3U + child_species_category]);
        species_count.fetch_add(1U);
        const auto energy_keV =
            static_cast<std::uint64_t>(
                sycl::fmax(0.0F, child_energy) *
                    1000.0F + 0.5F);
        sycl::atomic_ref<
            std::uint64_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>
            species_energy(
                minibeam_event_counts_device[
                    12U + child_species_category]);
        species_energy.fetch_add(energy_keV);
    } else {
        sycl::atomic_ref<
            std::uint32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>
            overflow(*secondary_overflow_count_device);
        overflow.fetch_add(1U);
    }
}

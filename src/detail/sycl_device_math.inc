// Shared SYCL transport helpers included by transport_sycl_legacy.cpp and
// transport_sycl.cpp. Do not compile this file as a standalone translation unit.
//
// Include context (required):
//   - inside `namespace carbon`
//   - for helpers below, typically inside the file-local anonymous namespace
//   - SYCL headers and carbon/* headers already included by the parent .cpp
//
// Keep body text bit-identical to the historical legacy definitions so OFF/ON
// isolation remains numerically stable.

bool is_uniform_grid(const std::vector<double>& energies) {
    const auto expected_step = energies[1] - energies[0];
    for (std::size_t index = 2; index < energies.size(); ++index) {
        const auto actual_step = energies[index] - energies[index - 1];
        if (std::abs(actual_step - expected_step) > 1.0e-6 * expected_step) {
            return false;
        }
    }
    return true;
}

double event_duration_seconds(const sycl::event& event) {
    const auto start =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto end =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
    return static_cast<double>(end - start) * 1.0e-9;
}

struct Direction3F {
    float x;
    float y;
    float z;
};

inline float advance_representable(const float position,
                                   const float direction,
                                   const float distance) noexcept {
    const auto advanced = position + direction * distance;
    if (advanced != position || direction == 0.0F) {
        return advanced;
    }
    constexpr auto infinity = std::numeric_limits<float>::infinity();
    return sycl::nextafter(position, direction > 0.0F ? infinity : -infinity);
}

Direction3F rotate_local_direction(const float local_x,
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
    const auto inverse_output_norm =
        1.0F / sycl::sqrt(output.x * output.x + output.y * output.y + output.z * output.z);
    output = Direction3F{output.x * inverse_output_norm,
                         output.y * inverse_output_norm,
                         output.z * inverse_output_norm};
    return output;
}


float highland_projected_rms_angle_device(const float kinetic_energy_MeV,
                                          const int atomic_number,
                                          const int mass_number,
                                          const float path_length_mm,
                                          const float density_g_per_cm3,
                                          const float radiation_length_g_per_cm2 =
                                              static_cast<float>(
                                                  water_radiation_length_g_per_cm2)) noexcept {
    if (kinetic_energy_MeV <= 0.0F || atomic_number <= 0 || mass_number <= 0 ||
        path_length_mm <= 0.0F || density_g_per_cm3 <= 0.0F ||
        radiation_length_g_per_cm2 <= 0.0F) {
        return 0.0F;
    }
    const auto energy_MeV_per_u = kinetic_energy_MeV / static_cast<float>(mass_number);
    const auto total_energy_MeV_per_u =
        energy_MeV_per_u + static_cast<float>(nucleon_rest_mass_MeV);
    const auto momentum_MeV_per_c_per_u = sycl::sqrt(
        energy_MeV_per_u *
        (energy_MeV_per_u + 2.0F * static_cast<float>(nucleon_rest_mass_MeV)));
    const auto beta = momentum_MeV_per_c_per_u / total_energy_MeV_per_u;
    const auto momentum_MeV_per_c =
        static_cast<float>(mass_number) * momentum_MeV_per_c_per_u;
    const auto radiation_lengths =
        density_g_per_cm3 * (path_length_mm / 10.0F) /
        radiation_length_g_per_cm2;
    if (beta <= 0.0F || momentum_MeV_per_c <= 0.0F || radiation_lengths <= 0.0F) {
        return 0.0F;
    }
    const auto charge = static_cast<float>(atomic_number);
    const auto logarithm_argument =
        radiation_lengths * charge * charge / (beta * beta);
    const auto correction =
        sycl::fmax(0.0F, 1.0F + 0.038F * sycl::log(logarithm_argument));
    return static_cast<float>(highland_energy_constant_MeV) * charge /
           (beta * momentum_MeV_per_c) * sycl::sqrt(radiation_lengths) * correction;
}

float highland_projected_rms_angle_with_mass_device(
    const float kinetic_energy_MeV,
    const int atomic_number,
    const float rest_mass_MeV,
    const float path_length_mm,
    const float density_g_per_cm3,
    const float radiation_length_g_per_cm2 =
        static_cast<float>(water_radiation_length_g_per_cm2)) noexcept {
    if (kinetic_energy_MeV <= 0.0F || atomic_number <= 0 ||
        rest_mass_MeV <= 0.0F || path_length_mm <= 0.0F ||
        density_g_per_cm3 <= 0.0F || radiation_length_g_per_cm2 <= 0.0F) {
        return 0.0F;
    }
    const auto total_energy_MeV = kinetic_energy_MeV + rest_mass_MeV;
    const auto momentum_MeV_per_c = sycl::sqrt(
        kinetic_energy_MeV * (kinetic_energy_MeV + 2.0F * rest_mass_MeV));
    const auto beta = momentum_MeV_per_c / total_energy_MeV;
    const auto radiation_lengths =
        density_g_per_cm3 * (path_length_mm / 10.0F) /
        radiation_length_g_per_cm2;
    if (beta <= 0.0F || momentum_MeV_per_c <= 0.0F ||
        radiation_lengths <= 0.0F) {
        return 0.0F;
    }
    const auto charge = static_cast<float>(atomic_number);
    const auto correction = sycl::fmax(
        0.0F,
        1.0F + 0.038F * sycl::log(
            radiation_lengths * charge * charge / (beta * beta)));
    return static_cast<float>(highland_energy_constant_MeV) * charge /
           (beta * momentum_MeV_per_c) * sycl::sqrt(radiation_lengths) *
           correction;
}

inline float ion_effective_charge_device(const int atomic_number,
                                          const float energy_MeVu) noexcept {
    if (atomic_number <= 0 || energy_MeVu <= 0.0F) {
        return 0.0F;
    }
    constexpr float nucleon_mass_MeV = 931.49410242F;
    const auto gamma = 1.0F + energy_MeVu / nucleon_mass_MeV;
    const auto beta_squared = sycl::fmax(0.0F, 1.0F - 1.0F / (gamma * gamma));
    const auto beta = sycl::sqrt(beta_squared);
    const auto charge = static_cast<float>(atomic_number);
    return charge *
           (1.0F - sycl::exp(-125.0F * beta * sycl::pow(charge, -2.0F / 3.0F)));
}

inline float condensed_total_loss_variance_MeV2_device(
    const float energy_MeVu,
    const int mass_number,
    const float effective_charge,
    const float path_length_mm,
    const float density_g_per_cm3,
    const float z_over_a_rel_water = 1.0F) noexcept {
    if (energy_MeVu <= 0.0F || mass_number <= 0 || effective_charge <= 0.0F ||
        path_length_mm <= 0.0F || density_g_per_cm3 <= 0.0F) {
        return 0.0F;
    }
    constexpr float electron_mass_MeV = 0.51099895F;
    constexpr float nucleon_mass_MeV = 931.49410242F;
    constexpr float bethe_K_MeV_cm2_per_g = 0.307075F;
    constexpr float water_Z_over_A = 0.55509F;
    const auto gamma = 1.0F + energy_MeVu / nucleon_mass_MeV;
    const auto beta_squared =
        sycl::fmax(0.0F, 1.0F - 1.0F / (gamma * gamma));
    if (beta_squared <= 0.0F) {
        return 0.0F;
    }
    const auto projectile_mass_MeV =
        static_cast<float>(mass_number) * nucleon_mass_MeV;
    const auto mass_ratio = electron_mass_MeV / projectile_mass_MeV;
    const auto maximum_transfer_MeV =
        2.0F * electron_mass_MeV * beta_squared * gamma * gamma /
        (1.0F + 2.0F * gamma * mass_ratio + mass_ratio * mass_ratio);
    const auto relativistic_multiplier =
        (maximum_transfer_MeV / beta_squared - 0.5F * maximum_transfer_MeV) /
        (2.0F * electron_mass_MeV);
    const auto z_over_a = water_Z_over_A *
                          sycl::clamp(z_over_a_rel_water, 0.5F, 1.5F);
    return bethe_K_MeV_cm2_per_g * electron_mass_MeV * effective_charge *
           effective_charge * z_over_a * density_g_per_cm3 *
           (path_length_mm / 10.0F) * sycl::fmax(0.0F, relativistic_multiplier);
}

inline float condensed_total_loss_variance_with_mass_MeV2_device(
    const float kinetic_energy_MeV,
    const float rest_mass_MeV,
    const float effective_charge,
    const float path_length_mm,
    const float density_g_per_cm3,
    const float z_over_a_rel_water = 1.0F) noexcept {
    if (kinetic_energy_MeV <= 0.0F || rest_mass_MeV <= 0.0F ||
        effective_charge <= 0.0F || path_length_mm <= 0.0F ||
        density_g_per_cm3 <= 0.0F) {
        return 0.0F;
    }
    constexpr float electron_mass_MeV = 0.51099895F;
    constexpr float bethe_K_MeV_cm2_per_g = 0.307075F;
    constexpr float water_Z_over_A = 0.55509F;
    const auto gamma = 1.0F + kinetic_energy_MeV / rest_mass_MeV;
    const auto beta_squared =
        sycl::fmax(0.0F, 1.0F - 1.0F / (gamma * gamma));
    if (beta_squared <= 0.0F) {
        return 0.0F;
    }
    const auto mass_ratio = electron_mass_MeV / rest_mass_MeV;
    const auto maximum_transfer_MeV =
        2.0F * electron_mass_MeV * beta_squared * gamma * gamma /
        (1.0F + 2.0F * gamma * mass_ratio + mass_ratio * mass_ratio);
    const auto relativistic_multiplier =
        (maximum_transfer_MeV / beta_squared - 0.5F * maximum_transfer_MeV) /
        (2.0F * electron_mass_MeV);
    const auto z_over_a = water_Z_over_A *
                          sycl::clamp(z_over_a_rel_water, 0.5F, 1.5F);
    return bethe_K_MeV_cm2_per_g * electron_mass_MeV * effective_charge *
           effective_charge * z_over_a * density_g_per_cm3 *
           (path_length_mm / 10.0F) *
           sycl::fmax(0.0F, relativistic_multiplier);
}

Direction3F scatter_direction(const Direction3F direction,
                              const float projected_rms_angle_rad,
                              const std::uint64_t seed,
                              const std::uint64_t history_id,
                              const std::uint64_t step_index,
                              const std::uint32_t random_dimension) noexcept {
    if (projected_rms_angle_rad <= 0.0F) {
        return direction;
    }
    const auto uniform1 = sycl::fmax(
        rng::uniform01(seed, history_id, step_index, random_dimension), 1.0e-12F);
    const auto uniform2 =
        rng::uniform01(seed, history_id, step_index, random_dimension + 1U);
    constexpr float two_pi = 6.2831853071795864769F;
    const auto radius = sycl::sqrt(-2.0F * sycl::log(uniform1));
    const auto local_x = projected_rms_angle_rad * radius * sycl::cos(two_pi * uniform2);
    const auto local_y = projected_rms_angle_rad * radius * sycl::sin(two_pi * uniform2);
    return rotate_local_direction(local_x, local_y, 1.0F, direction);
}

struct CorrelatedScatteringStep {
    Direction3F direction;
    Direction3F displacement_mm;
    // Optional state at an interior path fraction.  This is sampled from the
    // Gaussian bridge conditional on the already sampled end state, so asking
    // for a diagnostic crossing cannot change the transported endpoint.  Tail
    // events contribute only when their sampled location precedes the crossing.
    Direction3F observation_direction{};
    Direction3F observation_displacement_mm{};
    bool observation_valid{false};
};

CorrelatedScatteringStep copper_fermi_eyges_tail_step(
    const Direction3F direction, const float kinetic_energy_MeV,
    const int atomic_number, const int mass_number, const float path_length_mm,
    const float density_g_per_cm3, const float radiation_length_g_per_cm2,
    const float amplitude_scale, const std::uint64_t seed,
    const std::uint64_t history_id, const std::uint64_t step_index,
    const std::uint32_t random_dimension) noexcept {
    CorrelatedScatteringStep result{direction, Direction3F{0.0F, 0.0F, 0.0F}};
    if (kinetic_energy_MeV <= 0.0F || atomic_number <= 0 || mass_number <= 0 ||
        path_length_mm <= 0.0F || density_g_per_cm3 <= 0.0F ||
        radiation_length_g_per_cm2 <= 0.0F || amplitude_scale <= 0.0F) {
        return result;
    }

    const auto energy_per_u = kinetic_energy_MeV / static_cast<float>(mass_number);
    const auto total_energy_per_u =
        energy_per_u + static_cast<float>(nucleon_rest_mass_MeV);
    const auto momentum_per_u = sycl::sqrt(
        energy_per_u *
        (energy_per_u + 2.0F * static_cast<float>(nucleon_rest_mass_MeV)));
    const auto beta = momentum_per_u / total_energy_per_u;
    const auto momentum = static_cast<float>(mass_number) * momentum_per_u;
    const auto radiation_length_mm =
        10.0F * radiation_length_g_per_cm2 / density_g_per_cm3;
    if (!(beta > 0.0F && momentum > 0.0F && radiation_length_mm > 0.0F)) {
        return result;
    }

    // Constrained by 150/250/300 MeV/u, 1/10 mm Geant4 11.3.2 UrbanMsc
    // Copper slabs. This local scattering power has no step-length logarithm.
    constexpr float scattering_energy_MeV = 13.0F;
    constexpr float core_variance_fraction = 0.8076494F;
    constexpr float tail_rate_per_mm = 0.1065470F;
    constexpr float wide_tail_fraction = 0.01563534F;
    constexpr float wide_tail_width_ratio = 4.522415F;
    const auto scaled_scattering_energy = amplitude_scale * scattering_energy_MeV;
    const auto charge = static_cast<float>(atomic_number);
    const auto scattering_ratio =
        scaled_scattering_energy * charge / (beta * momentum);
    const auto total_scattering_power =
        scattering_ratio * scattering_ratio / radiation_length_mm;
    const auto core_power = core_variance_fraction * total_scattering_power;
    const auto core_theta_sigma = sycl::sqrt(core_power * path_length_mm);

    const auto normal_pair = [&](const std::uint32_t dimension) {
        const auto u0 = sycl::fmax(
            rng::uniform01(seed, history_id, step_index, dimension), 1.0e-12F);
        const auto u1 = rng::uniform01(
            seed, history_id, step_index, dimension + 1U);
        constexpr float two_pi = 6.2831853071795864769F;
        const auto radius = sycl::sqrt(-2.0F * sycl::log(u0));
        return Direction3F{radius * sycl::cos(two_pi * u1),
                           radius * sycl::sin(two_pi * u1), 0.0F};
    };
    const auto angle_normal = normal_pair(random_dimension);
    const auto displacement_normal = normal_pair(random_dimension + 2U);
    auto local_theta_x = core_theta_sigma * angle_normal.x;
    auto local_theta_y = core_theta_sigma * angle_normal.y;
    const auto independent_displacement_sigma = sycl::sqrt(
        core_power * path_length_mm * path_length_mm * path_length_mm / 12.0F);
    auto local_displacement_x =
        0.5F * path_length_mm * local_theta_x +
        independent_displacement_sigma * displacement_normal.x;
    auto local_displacement_y =
        0.5F * path_length_mm * local_theta_y +
        independent_displacement_sigma * displacement_normal.y;

    const auto tail_mean = tail_rate_per_mm * path_length_mm;
    const auto zero_tail_probability = sycl::exp(-tail_mean);
    const auto one_tail_probability = zero_tail_probability * tail_mean;
    const auto count_uniform = rng::uniform01(
        seed, history_id, step_index, random_dimension + 4U);
    const auto tail_count = count_uniform < zero_tail_probability
        ? 0U
        : (count_uniform < zero_tail_probability + one_tail_probability ? 1U : 2U);
    const auto width_variance_normalizer =
        (1.0F - wide_tail_fraction) +
        wide_tail_fraction * wide_tail_width_ratio * wide_tail_width_ratio;
    const auto narrow_tail_sigma = sycl::sqrt(
        (1.0F - core_variance_fraction) * total_scattering_power /
        (tail_rate_per_mm * width_variance_normalizer));
    for (std::uint32_t tail_index = 0; tail_index < tail_count; ++tail_index) {
        const auto tail_dimension = random_dimension + 20U + 4U * tail_index;
        const auto location_uniform = rng::uniform01(
            seed, history_id, step_index, tail_dimension);
        const auto width_uniform = rng::uniform01(
            seed, history_id, step_index, tail_dimension + 1U);
        const auto tail_normal = normal_pair(tail_dimension + 2U);
        const auto tail_sigma = narrow_tail_sigma *
            (width_uniform < wide_tail_fraction ? wide_tail_width_ratio : 1.0F);
        const auto tail_theta_x = tail_sigma * tail_normal.x;
        const auto tail_theta_y = tail_sigma * tail_normal.y;
        const auto remaining_path = path_length_mm * (1.0F - location_uniform);
        local_theta_x += tail_theta_x;
        local_theta_y += tail_theta_y;
        local_displacement_x += remaining_path * tail_theta_x;
        local_displacement_y += remaining_path * tail_theta_y;
    }

    const auto parent_norm = sycl::sqrt(
        direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    const auto inverse_parent_norm = parent_norm > 0.0F ? 1.0F / parent_norm : 1.0F;
    const Direction3F w{direction.x * inverse_parent_norm,
                        direction.y * inverse_parent_norm,
                        parent_norm > 0.0F ? direction.z * inverse_parent_norm : 1.0F};
    const Direction3F reference =
        sycl::fabs(w.x) < 0.9F ? Direction3F{1.0F, 0.0F, 0.0F}
                               : Direction3F{0.0F, 1.0F, 0.0F};
    const auto projection = reference.x * w.x + reference.y * w.y + reference.z * w.z;
    Direction3F u{reference.x - projection * w.x,
                  reference.y - projection * w.y,
                  reference.z - projection * w.z};
    const auto inverse_u_norm = sycl::rsqrt(u.x * u.x + u.y * u.y + u.z * u.z);
    u = Direction3F{u.x * inverse_u_norm, u.y * inverse_u_norm, u.z * inverse_u_norm};
    const Direction3F v{w.y * u.z - w.z * u.y,
                        w.z * u.x - w.x * u.z,
                        w.x * u.y - w.y * u.x};
    result.displacement_mm = Direction3F{
        local_displacement_x * u.x + local_displacement_y * v.x,
        local_displacement_x * u.y + local_displacement_y * v.y,
        local_displacement_x * u.z + local_displacement_y * v.z};
    result.direction = rotate_local_direction(
        local_theta_x, local_theta_y, 1.0F, direction);
    return result;
}

struct WaterIonFermiEygesParameters {
    float core_scattering_energy_MeV;
    float tail_rate_per_water_mm;
    float tail_scattering_energy_MeV;
};

float interpolate_water_fe_parameter(const float energy_per_u,
                                     const float value_50,
                                     const float value_150,
                                     const float value_300) noexcept {
    if (energy_per_u <= 50.0F) return value_50;
    if (energy_per_u <= 150.0F) {
        const auto fraction = (energy_per_u - 50.0F) / 100.0F;
        return value_50 + fraction * (value_150 - value_50);
    }
    if (energy_per_u < 300.0F) {
        const auto fraction = (energy_per_u - 150.0F) / 150.0F;
        return value_150 + fraction * (value_300 - value_150);
    }
    return value_300;
}

WaterIonFermiEygesParameters water_ion_fermi_eyges_parameters(
    const int atomic_number, const int mass_number,
    const float energy_per_u,
    const bool use_species_water_parameters = true) noexcept {
    // Pure-EM G4_WATER slab fits at 50/150/300 MeV/u.  Each energy used
    // independent 1 and 10 mm, 250k-history TOPAS references.  Keep C12 and
    // uncalibrated ions on the previously validated C12-water candidate.
    if (use_species_water_parameters && atomic_number == 1 && mass_number == 1) {
        return WaterIonFermiEygesParameters{
            interpolate_water_fe_parameter(
                energy_per_u, 12.1250440F, 11.8620445F, 11.7235806F),
            interpolate_water_fe_parameter(
                energy_per_u, 0.002426742F, 0.002246132F, 0.001998740F),
            interpolate_water_fe_parameter(
                energy_per_u, 6.7809249F, 7.8095673F, 8.8033596F)};
    }
    if (use_species_water_parameters && atomic_number == 1 && mass_number == 2) {
        return WaterIonFermiEygesParameters{
            interpolate_water_fe_parameter(
                energy_per_u, 10.7251611F, 10.5823527F, 10.4433029F),
            interpolate_water_fe_parameter(
                energy_per_u, 0.003069940F, 0.003104010F, 0.003273779F),
            interpolate_water_fe_parameter(
                energy_per_u, 3.2194970F, 3.0651926F, 2.7842157F)};
    }
    if (use_species_water_parameters && atomic_number == 1 && mass_number == 3) {
        return WaterIonFermiEygesParameters{
            interpolate_water_fe_parameter(
                energy_per_u, 10.6735838F, 10.4564336F, 10.3058709F),
            interpolate_water_fe_parameter(
                energy_per_u, 0.003033585F, 0.003151119F, 0.003381651F),
            interpolate_water_fe_parameter(
                energy_per_u, 3.2822138F, 2.9271832F, 2.6076538F)};
    }
    if (use_species_water_parameters && atomic_number == 2 && mass_number == 4) {
        return WaterIonFermiEygesParameters{
            interpolate_water_fe_parameter(
                energy_per_u, 10.3786895F, 10.3427338F, 10.2402913F),
            interpolate_water_fe_parameter(
                energy_per_u, 0.003159817F, 0.003491817F, 0.003435895F),
            interpolate_water_fe_parameter(
                energy_per_u, 3.0908322F, 2.6518513F, 2.5755165F)};
    }
    return WaterIonFermiEygesParameters{9.9F, 0.0025F, 2.4F};
}

CorrelatedScatteringStep water_ion_fermi_eyges_tail_step(
    const Direction3F direction, const float kinetic_energy_MeV,
    const int atomic_number, const int mass_number,
    const float path_length_mm, const float density_g_per_cm3,
    const float radiation_length_g_per_cm2, const std::uint64_t seed,
    const std::uint64_t history_id, const std::uint64_t step_index,
    const std::uint32_t random_dimension,
    const float observation_fraction = -1.0F,
    const bool use_species_water_parameters = true) noexcept {
    CorrelatedScatteringStep result{direction, Direction3F{0.0F, 0.0F, 0.0F}};
    if (kinetic_energy_MeV <= 0.0F || atomic_number <= 0 || mass_number <= 0 ||
        path_length_mm <= 0.0F ||
        density_g_per_cm3 <= 0.0F || radiation_length_g_per_cm2 <= 0.0F) {
        return result;
    }

    const auto energy_per_u = kinetic_energy_MeV / static_cast<float>(mass_number);
    const auto total_energy_per_u =
        energy_per_u + static_cast<float>(nucleon_rest_mass_MeV);
    const auto momentum_per_u = sycl::sqrt(
        energy_per_u *
        (energy_per_u + 2.0F * static_cast<float>(nucleon_rest_mass_MeV)));
    const auto beta = momentum_per_u / total_energy_per_u;
    const auto momentum = static_cast<float>(mass_number) * momentum_per_u;
    const auto radiation_length_mm =
        10.0F * radiation_length_g_per_cm2 / density_g_per_cm3;
    if (!(beta > 0.0F && momentum > 0.0F && radiation_length_mm > 0.0F)) {
        return result;
    }

    // The core uses the current material's radiation length directly.  Tail
    // rates are fitted per physical millimetre in water, then scaled by the
    // ratio of water/current inverse radiation lengths for Schneider CT
    // materials.  This preserves the water calibration without fitting CT
    // dose and gives the same rate in native G4_WATER.
    const auto parameters = water_ion_fermi_eyges_parameters(
        atomic_number, mass_number, energy_per_u,
        use_species_water_parameters);
    constexpr float water_reference_radiation_length_mm = 360.8297746F;
    const auto tail_material_scale =
        water_reference_radiation_length_mm / radiation_length_mm;
    const auto tail_rate_per_mm =
        parameters.tail_rate_per_water_mm * tail_material_scale;
    const auto charge_over_beta_momentum =
        static_cast<float>(atomic_number) / (beta * momentum);
    const auto core_ratio =
        parameters.core_scattering_energy_MeV * charge_over_beta_momentum;
    const auto core_power =
        core_ratio * core_ratio / radiation_length_mm;
    const auto core_theta_sigma = sycl::sqrt(core_power * path_length_mm);

    const auto normal_pair = [&](const std::uint32_t dimension) {
        const auto u0 = sycl::fmax(
            rng::uniform01(seed, history_id, step_index, dimension), 1.0e-12F);
        const auto u1 = rng::uniform01(
            seed, history_id, step_index, dimension + 1U);
        constexpr float two_pi = 6.2831853071795864769F;
        const auto radius = sycl::sqrt(-2.0F * sycl::log(u0));
        return Direction3F{radius * sycl::cos(two_pi * u1),
                           radius * sycl::sin(two_pi * u1), 0.0F};
    };
    const auto angle_normal = normal_pair(random_dimension);
    const auto displacement_normal = normal_pair(random_dimension + 2U);
    auto local_theta_x = core_theta_sigma * angle_normal.x;
    auto local_theta_y = core_theta_sigma * angle_normal.y;
    const auto independent_displacement_sigma = sycl::sqrt(
        core_power * path_length_mm * path_length_mm * path_length_mm / 12.0F);
    auto local_displacement_x =
        0.5F * path_length_mm * local_theta_x +
        independent_displacement_sigma * displacement_normal.x;
    auto local_displacement_y =
        0.5F * path_length_mm * local_theta_y +
        independent_displacement_sigma * displacement_normal.y;

    const auto has_observation =
        observation_fraction > 0.0F && observation_fraction < 1.0F;
    auto observation_theta_x = 0.0F;
    auto observation_theta_y = 0.0F;
    auto observation_displacement_x = 0.0F;
    auto observation_displacement_y = 0.0F;
    if (has_observation) {
        // Integrated Brownian-motion bridge for (theta(t), y(t)) conditional
        // on the sampled (theta(L), y(L)).  Work in units where T=L=1;
        // rescaling below restores sqrt(TL) and sqrt(TL^3).
        const auto r = observation_fraction;
        const auto r2 = r * r;
        const auto r3 = r2 * r;
        const auto c00 = r;
        const auto c01 = r - 0.5F * r2;
        const auto c10 = 0.5F * r2;
        const auto c11 = 0.5F * r2 - r3 / 6.0F;
        const auto k00 = 4.0F * c00 - 6.0F * c01;
        const auto k01 = -6.0F * c00 + 12.0F * c01;
        const auto k10 = 4.0F * c10 - 6.0F * c11;
        const auto k11 = -6.0F * c10 + 12.0F * c11;
        const auto conditional_var_theta = sycl::fmax(
            0.0F, r - (k00 * c00 + k01 * c01));
        const auto conditional_cov =
            0.5F * r2 - (k00 * c10 + k01 * c11);
        const auto conditional_var_displacement = sycl::fmax(
            0.0F, r3 / 3.0F - (k10 * c10 + k11 * c11));
        const auto bridge_normal0 = normal_pair(random_dimension + 8U);
        const auto bridge_normal1 = normal_pair(random_dimension + 10U);
        const auto l00 = sycl::sqrt(conditional_var_theta);
        const auto l10 = l00 > 0.0F ? conditional_cov / l00 : 0.0F;
        const auto l11 = sycl::sqrt(sycl::fmax(
            0.0F, conditional_var_displacement - l10 * l10));
        const auto theta_scale = core_theta_sigma;
        const auto displacement_scale =
            core_theta_sigma * path_length_mm;
        const auto normalized_final_displacement_x =
            local_displacement_x / displacement_scale;
        const auto normalized_final_displacement_y =
            local_displacement_y / displacement_scale;
        const auto normalized_final_theta_x =
            local_theta_x / theta_scale;
        const auto normalized_final_theta_y =
            local_theta_y / theta_scale;
        const auto bridge_theta_x =
            k00 * normalized_final_theta_x +
            k01 * normalized_final_displacement_x +
            l00 * bridge_normal0.x;
        const auto bridge_theta_y =
            k00 * normalized_final_theta_y +
            k01 * normalized_final_displacement_y +
            l00 * bridge_normal0.y;
        const auto bridge_displacement_x =
            k10 * normalized_final_theta_x +
            k11 * normalized_final_displacement_x +
            l10 * bridge_normal0.x + l11 * bridge_normal1.x;
        const auto bridge_displacement_y =
            k10 * normalized_final_theta_y +
            k11 * normalized_final_displacement_y +
            l10 * bridge_normal0.y + l11 * bridge_normal1.y;
        observation_theta_x = theta_scale * bridge_theta_x;
        observation_theta_y = theta_scale * bridge_theta_y;
        observation_displacement_x =
            displacement_scale * bridge_displacement_x;
        observation_displacement_y =
            displacement_scale * bridge_displacement_y;
    }

    // Knuth's exact Poisson draw has no event-count truncation. The configured
    // mean is about 2.5e-4 per normal 0.1 mm step, so the expected loop cost is
    // negligible. Conditional event locations are uniform along the step.
    const auto tail_mean = tail_rate_per_mm * path_length_mm;
    const auto poisson_limit = sycl::exp(-tail_mean);
    auto poisson_product = 1.0F;
    std::uint32_t tail_count = 0U;
    while (true) {
        poisson_product *= sycl::fmax(
            rng::uniform01(seed, history_id, step_index,
                           random_dimension + 4U + tail_count),
            1.0e-12F);
        if (poisson_product <= poisson_limit) break;
        ++tail_count;
    }
    const auto tail_sigma =
        parameters.tail_scattering_energy_MeV * charge_over_beta_momentum;
    for (std::uint32_t tail_index = 0; tail_index < tail_count; ++tail_index) {
        const auto dimension = random_dimension + 32U + 3U * tail_index;
        const auto location = rng::uniform01(
            seed, history_id, step_index, dimension);
        const auto tail_normal = normal_pair(dimension + 1U);
        const auto tail_theta_x = tail_sigma * tail_normal.x;
        const auto tail_theta_y = tail_sigma * tail_normal.y;
        const auto remaining_path = path_length_mm * (1.0F - location);
        local_theta_x += tail_theta_x;
        local_theta_y += tail_theta_y;
        local_displacement_x += remaining_path * tail_theta_x;
        local_displacement_y += remaining_path * tail_theta_y;
        if (has_observation && location <= observation_fraction) {
            observation_theta_x += tail_theta_x;
            observation_theta_y += tail_theta_y;
            const auto observation_remaining_path =
                path_length_mm * (observation_fraction - location);
            observation_displacement_x +=
                observation_remaining_path * tail_theta_x;
            observation_displacement_y +=
                observation_remaining_path * tail_theta_y;
        }
    }

    const auto parent_norm = sycl::sqrt(
        direction.x * direction.x + direction.y * direction.y +
        direction.z * direction.z);
    const auto inverse_parent_norm =
        parent_norm > 0.0F ? 1.0F / parent_norm : 1.0F;
    const Direction3F w{direction.x * inverse_parent_norm,
                        direction.y * inverse_parent_norm,
                        parent_norm > 0.0F
                            ? direction.z * inverse_parent_norm : 1.0F};
    const Direction3F reference =
        sycl::fabs(w.x) < 0.9F ? Direction3F{1.0F, 0.0F, 0.0F}
                               : Direction3F{0.0F, 1.0F, 0.0F};
    const auto projection =
        reference.x * w.x + reference.y * w.y + reference.z * w.z;
    Direction3F u{reference.x - projection * w.x,
                  reference.y - projection * w.y,
                  reference.z - projection * w.z};
    const auto inverse_u_norm =
        sycl::rsqrt(u.x * u.x + u.y * u.y + u.z * u.z);
    u = Direction3F{u.x * inverse_u_norm, u.y * inverse_u_norm,
                    u.z * inverse_u_norm};
    const Direction3F v{w.y * u.z - w.z * u.y,
                        w.z * u.x - w.x * u.z,
                        w.x * u.y - w.y * u.x};
    result.displacement_mm = Direction3F{
        local_displacement_x * u.x + local_displacement_y * v.x,
        local_displacement_x * u.y + local_displacement_y * v.y,
        local_displacement_x * u.z + local_displacement_y * v.z};
    result.direction = rotate_local_direction(
        local_theta_x, local_theta_y, 1.0F, direction);
    if (has_observation) {
        result.observation_displacement_mm = Direction3F{
            observation_displacement_x * u.x + observation_displacement_y * v.x,
            observation_displacement_x * u.y + observation_displacement_y * v.y,
            observation_displacement_x * u.z + observation_displacement_y * v.z};
        result.observation_direction = rotate_local_direction(
            observation_theta_x, observation_theta_y, 1.0F, direction);
        result.observation_valid = true;
    }
    return result;
}

CorrelatedScatteringStep water_c12_fermi_eyges_tail_step(
    const Direction3F direction, const float kinetic_energy_MeV,
    const float path_length_mm, const float density_g_per_cm3,
    const float radiation_length_g_per_cm2, const std::uint64_t seed,
    const std::uint64_t history_id, const std::uint64_t step_index,
    const std::uint32_t random_dimension,
    const float observation_fraction = -1.0F) noexcept {
    return water_ion_fermi_eyges_tail_step(
        direction, kinetic_energy_MeV, 6, 12, path_length_mm,
        density_g_per_cm3, radiation_length_g_per_cm2, seed, history_id,
        step_index, random_dimension, observation_fraction);
}

// Runtime scope levels mirror fermi_eyges_species in the YAML parser:
// 0=C12, 1=+He4, 2=+p/d/t, 3=all supported charged ions.
bool ion_uses_fermi_eyges(const int scope, const int atomic_number,
                          const int mass_number) noexcept {
    if (atomic_number == 6 && mass_number == 12) return true;
    if (scope >= 1 && atomic_number == 2 && mass_number == 4) return true;
    if (scope >= 2 && atomic_number == 1 &&
        mass_number >= 1 && mass_number <= 3) return true;
    return scope >= 3 && atomic_number > 0 && mass_number > 0;
}

// Advance one complete ion material step with the local FE process. This is
// the common broad-beam/minibeam entry point. When an interior observation
// path is supplied, the helper returns the conditional bridge state while the
// caller remains responsible only for scorer storage. The returned endpoint
// displacement is relative to a straight drift along the input direction.
// p/d/t/He-4 use their independent water tables. C12 and other uncalibrated
// ions retain the C12-water candidate, with charge/(beta*p) scaling.
CorrelatedScatteringStep ion_fermi_eyges_transport_step(
    const Direction3F direction, const float kinetic_energy_before_loss_MeV,
    const int atomic_number, const int mass_number,
    const float sampled_energy_loss_MeV, const float path_length_mm,
    const float max_segment_mm, const float density_g_per_cm3,
    const float radiation_length_g_per_cm2, const std::uint64_t seed,
    const std::uint64_t history_id, const std::uint64_t transport_step_index,
    const std::uint32_t random_dimension,
    const float observation_path_mm = -1.0F,
    const bool use_species_water_parameters = true) noexcept {
    CorrelatedScatteringStep result{
        direction, Direction3F{0.0F, 0.0F, 0.0F}};
    if (!(path_length_mm > 0.0F && max_segment_mm > 0.0F)) return result;

    auto segment_direction = direction;
    auto total_offset = Direction3F{0.0F, 0.0F, 0.0F};
    auto observation_total_offset = Direction3F{0.0F, 0.0F, 0.0F};
    auto traversed_mm = 0.0F;
    std::uint32_t segment_index = 0U;
    while (traversed_mm < path_length_mm) {
        const auto segment_mm = sycl::fmin(
            max_segment_mm, path_length_mm - traversed_mm);
        const auto energy_fraction =
            (traversed_mm + 0.5F * segment_mm) / path_length_mm;
        const auto segment_energy = sycl::fmax(
            0.0F, kinetic_energy_before_loss_MeV -
                      energy_fraction * sampled_energy_loss_MeV);
        const auto observes_this_segment =
            observation_path_mm >= traversed_mm &&
            observation_path_mm <= traversed_mm + segment_mm;
        const auto observation_fraction = observes_this_segment
            ? sycl::clamp(
                  (observation_path_mm - traversed_mm) / segment_mm,
                  1.0e-6F, 1.0F - 1.0e-6F)
            : -1.0F;
        const auto segment = water_ion_fermi_eyges_tail_step(
            segment_direction, segment_energy, atomic_number, mass_number,
            segment_mm,
            density_g_per_cm3, radiation_length_g_per_cm2, seed, history_id,
            transport_step_index * 1024U + segment_index,
            random_dimension, observation_fraction,
            use_species_water_parameters);
        if (segment.observation_valid) {
            const auto local_path = observation_fraction * segment_mm;
            observation_total_offset.x =
                total_offset.x + segment_direction.x * local_path +
                segment.observation_displacement_mm.x;
            observation_total_offset.y =
                total_offset.y + segment_direction.y * local_path +
                segment.observation_displacement_mm.y;
            observation_total_offset.z =
                total_offset.z + segment_direction.z * local_path +
                segment.observation_displacement_mm.z;
            result.observation_direction = segment.observation_direction;
            result.observation_valid = true;
        }
        total_offset.x += segment_direction.x * segment_mm +
                          segment.displacement_mm.x;
        total_offset.y += segment_direction.y * segment_mm +
                          segment.displacement_mm.y;
        total_offset.z += segment_direction.z * segment_mm +
                          segment.displacement_mm.z;
        segment_direction = segment.direction;
        traversed_mm += segment_mm;
        ++segment_index;
    }
    result.direction = segment_direction;
    result.displacement_mm = Direction3F{
        total_offset.x - direction.x * path_length_mm,
        total_offset.y - direction.y * path_length_mm,
        total_offset.z - direction.z * path_length_mm};
    if (result.observation_valid) {
        result.observation_displacement_mm = Direction3F{
            observation_total_offset.x - direction.x * observation_path_mm,
            observation_total_offset.y - direction.y * observation_path_mm,
            observation_total_offset.z - direction.z * observation_path_mm};
    }
    return result;
}

CorrelatedScatteringStep c12_fermi_eyges_transport_step(
    const Direction3F direction, const float kinetic_energy_before_loss_MeV,
    const float sampled_energy_loss_MeV, const float path_length_mm,
    const float max_segment_mm, const float density_g_per_cm3,
    const float radiation_length_g_per_cm2, const std::uint64_t seed,
    const std::uint64_t history_id, const std::uint64_t transport_step_index,
    const std::uint32_t random_dimension,
    const float observation_path_mm = -1.0F) noexcept {
    return ion_fermi_eyges_transport_step(
        direction, kinetic_energy_before_loss_MeV, 6, 12,
        sampled_energy_loss_MeV, path_length_mm, max_segment_mm,
        density_g_per_cm3, radiation_length_g_per_cm2, seed, history_id,
        transport_step_index, random_dimension, observation_path_mm);
}

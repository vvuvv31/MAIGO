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
    // Explicit urban_v2 step proposal/finalization (R4).  Defaults keep the
    // legacy highland/fermi_eyges/urban paths bit-identical: consumers that
    // do not understand the contract must ignore these fields.
    // proposed_t: MSC true-path proposal; final_t: resolved true length used
    //   for loss/fluctuation/reaction paths; final_g: geometric length used
    //   for position advance; delta: stable (t-g) saved on the conversion
    //   branch (never float(t)-float(g) post-hoc).
    // limit_reason: 0=none/unlimited, 1=range, 2=geometry, 3=collision, 4=stop.
    // boundary_crossed: geometry truncated the proposal (final_g < proposed_g).
    // one_minus_cth: 1-cos(theta) saved before rounding (cth~1 diagnostics).
    float proposed_true_path_mm{0.0F};
    float final_true_path_mm{0.0F};
    float final_geom_path_mm{0.0F};
    float stable_delta_mm{0.0F};
    float one_minus_cth{0.0F};
    int limit_reason{0};
    bool boundary_crossed{false};
    bool proposal_valid{false};
};

// Stable geometric->true conversion shared by host diagnostics and the device
// sampler (R3).  Computes t and delta=(t-g) directly on the conversion branch
// in double precision, then narrows.  Never forms float(t)-float(g).
struct UrbanV2TruePath {
    float true_path_mm;
    float delta_mm;
};

inline UrbanV2TruePath copper_urban_v2_true_path_and_delta(
    const float geom_path_mm, const float transport_mfp_mm,
    const float range_mm) noexcept {
    UrbanV2TruePath out{geom_path_mm, 0.0F};
    constexpr double kTauSmall = 1.0e-16;
    constexpr double kTLimitMinFix2 = 1.0e-6;  // mm
    if (!(geom_path_mm > 0.0F) || !(transport_mfp_mm > 0.0F)) {
        return out;
    }
    const double g = static_cast<double>(geom_path_mm);
    const double lam = static_cast<double>(transport_mfp_mm);
    if (!(g >= kTLimitMinFix2) || !(g > lam * kTauSmall)) {
        return out;
    }
    const double ratio = g / lam;
    double t;
    double delta;
    if (ratio < 1.0e-3) {
        // Taylor of -lam*(log1p(-r)+r) = g*r*(1/2+r/3+r^2/4+...).
        // Pure arithmetic: no log1p call, no cancellation, portable to
        // device double without relying on a double log1p overload.
        const double series =
            0.5 + ratio * (1.0 / 3.0 + ratio * (0.25 + ratio * 0.2));
        delta = g * ratio * series;
        t = g + delta;
    } else if (ratio < 1.0) {
        // delta = -lam*(log1p(-r)+r) == t-g, evaluated without cancellation.
        const double log_term =
            static_cast<double>(sycl::log1p(static_cast<float>(-ratio)));
        t = -lam * log_term;
        delta = -lam * (log_term + ratio);
        if (!(delta > 0.0)) delta = 0.0;
    } else {
        t = static_cast<double>(range_mm > 0.0F ? range_mm : geom_path_mm);
        delta = t - g;
        if (!(delta > 0.0)) delta = 0.0;
    }
    double t_clamped = t;
    if (t_clamped < g) t_clamped = g;
    if (range_mm > 0.0F && t_clamped > static_cast<double>(range_mm)) {
        t_clamped = static_cast<double>(range_mm);
        delta = t_clamped - g;
        if (!(delta > 0.0)) delta = 0.0;
    }
    out.true_path_mm = static_cast<float>(t_clamped);
    out.delta_mm = static_cast<float>(delta);
    return out;
}

// Process-level displacement acceptance with DISTINCT reference thresholds
// (R2).  geom_min_mm is the G4VMultipleScattering geometry tolerance;
// min_displacement_mm is the separate physics displacement floor
// (minDisplacement2 family).  They must not share the model geommin constant.
// Values below are STRUCTURAL PLACEHOLDERS pending a real reference trace
// (AlongStepDoIt postSafety/geomMin/minDisplacement2); the regression test
// R2 (raw=1e-4, safety=1e-5 -> reduce) pins the required ordering.
constexpr float kUrbanV2GeomMinMm = 1.0e-6F;       // UNVALIDATED placeholder
constexpr float kUrbanV2MinDisplacementMm = 1.0e-7F;  // UNVALIDATED placeholder

inline float copper_urban_v2_accept_displacement(
    const float raw_x, const float raw_y, const float safety_mm,
    const float geom_min_mm, const float min_displacement_mm,
    float& accepted_x, float& accepted_y, int& branch) noexcept {
    accepted_x = 0.0F;
    accepted_y = 0.0F;
    branch = 0;  // 0=zero, 1=accept, 2=reduce, 3=cancel
    const double dx = static_cast<double>(raw_x);
    const double dy = static_cast<double>(raw_y);
    const double disp_r = sycl::sqrt(dx * dx + dy * dy);
    if (!(disp_r > 0.0)) {
        return 0.0F;
    }
    const double post =
        0.99 * static_cast<double>(safety_mm > 0.0F ? safety_mm : 0.0F);
    if (disp_r <= post) {
        accepted_x = raw_x;
        accepted_y = raw_y;
        branch = 1;
        return static_cast<float>(disp_r);
    }
    if (post > static_cast<double>(geom_min_mm)) {
        const double scale = post / disp_r;
        accepted_x = static_cast<float>(dx * scale);
        accepted_y = static_cast<float>(dy * scale);
        branch = 2;
        return static_cast<float>(post);
    }
    if (post <= static_cast<double>(min_displacement_mm)) {
        branch = 3;
        return 0.0F;
    }
    const double scale = post / disp_r;
    accepted_x = static_cast<float>(dx * scale);
    accepted_y = static_cast<float>(dy * scale);
    branch = 2;
    return static_cast<float>(post);
}

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
    // Extreme wide-angle tail component.  The 1/10 mm slab fit above matches the
    // core and the q50-q95 angles but underproduces the >40 mrad tail at 60 mm
    // collimator thickness (TOPAS/GPU: 0.950 at >40 mrad, 0.789 at >80 mrad,
    // 0.518 at >160 mrad on the single-spot 250 MeV/u water-entrance phase
    // space).  A third, rarer and much wider Poisson component restores the
    // extreme tail while the normalizer keeps the total tail variance fixed.
    constexpr float extreme_tail_fraction = 0.0F;
    constexpr float extreme_tail_width_ratio = 10.0F;
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
        (1.0F - wide_tail_fraction - extreme_tail_fraction) +
        wide_tail_fraction * wide_tail_width_ratio * wide_tail_width_ratio +
        extreme_tail_fraction * extreme_tail_width_ratio * extreme_tail_width_ratio;
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
            (width_uniform < extreme_tail_fraction
                 ? extreme_tail_width_ratio
                 : (width_uniform < extreme_tail_fraction + wide_tail_fraction
                        ? wide_tail_width_ratio
                        : 1.0F));
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

// ---------------------------------------------------------------------------
// Faithful port of Geant4 11.3.2 G4UrbanMscModel for the Copper minibeam
// collimator (Zeff = 29, A = 63.546).  Implements ComputeCrossSectionPerAtom
// -> lambda0, ComputeTheta0, SampleCosineTheta, SimpleScattering and
// SampleDisplacementNew (dispAlg96 == false).  Step limitation is delegated to
// the caller's transport step; the caller scales the result with mcs_scale.
// ---------------------------------------------------------------------------

inline float copper_urban_cross_section_per_atom_cm2(
    const float kin_energy_MeV, const int projectile_charge,
    const int mass_number) noexcept {
    constexpr float electron_mass_MeV = 0.51099895F;
    constexpr float nucleon_mass_MeV = 931.49410242F;
    constexpr float bohr_radius_cm = 5.29177210903e-9F;
    constexpr float hbarc_MeV_cm = 197.3269804e-13F;
    constexpr float classic_electron_radius_cm = 2.8179403262e-13F;
    constexpr float barn_cm2 = 1.0e-24F;
    constexpr float low_energy_limit_MeV = 10.0F;
    constexpr float copper_zeffective = 29.0F;
    if (!(kin_energy_MeV > 0.0F) || projectile_charge <= 0 || mass_number <= 0) {
        return 0.0F;
    }

    const auto charge = static_cast<float>(projectile_charge);
    const auto mass = static_cast<float>(mass_number) * nucleon_mass_MeV;

    // electron-equivalent kinetic energy of a heavy ion
    const auto tau_ion = kin_energy_MeV / mass;
    const auto c_ion = mass * tau_ion * (tau_ion + 2.0F) /
                       (electron_mass_MeV * (tau_ion + 1.0F));
    const auto w = c_ion - 2.0F;
    const auto tau_e = 0.5F * (w + sycl::sqrt(w * w + 4.0F * c_ion));
    const auto ekin = electron_mass_MeV * tau_e;
    const auto etot = ekin + electron_mass_MeV;
    const auto beta2 = ekin * (etot + electron_mass_MeV) / (etot * etot);
    const auto bg2 = ekin * (etot + electron_mass_MeV) /
                     (electron_mass_MeV * electron_mass_MeV);

    const auto z23 = sycl::pow(copper_zeffective, 2.0F / 3.0F);
    const auto epsfactor = 2.0F * electron_mass_MeV * electron_mass_MeV *
                           bohr_radius_cm * bohr_radius_cm /
                           (hbarc_MeV_cm * hbarc_MeV_cm);
    const auto eps = epsfactor * bg2 / z23;

    float sigma = 0.0F;
    if (eps < 1.0e-4F) {
        sigma = 2.0F * eps * eps;
    } else if (eps < 1.0e10F) {
        sigma = sycl::log(1.0F + 2.0F * eps) -
                2.0F * eps / (1.0F + 2.0F * eps);
    } else {
        sigma = sycl::log(2.0F * eps) - 1.0F + 1.0F / eps;
    }
    sigma *= charge * charge * copper_zeffective * copper_zeffective /
             (beta2 * bg2);

    constexpr float t_lim = low_energy_limit_MeV;
    const auto beta2_lim =
        t_lim * (t_lim + 2.0F * electron_mass_MeV) /
        ((t_lim + electron_mass_MeV) * (t_lim + electron_mass_MeV));
    const auto bg2_lim = t_lim * (t_lim + 2.0F * electron_mass_MeV) /
                         (electron_mass_MeV * electron_mass_MeV);
    constexpr float sigma_factor =
        6.2831853071795864769F * classic_electron_radius_cm *
        classic_electron_radius_cm;

    if (ekin <= t_lim) {
        // Copper lies in Zdat[4]=26 .. Zdat[5]=29 with ratZ = 1, so the
        // interpolated cpositron coefficient is exactly cpositron[5].
        constexpr float cpositron_z29[22] = {
            24.83F, 17.06F, 10.84F, 7.355F, 5.767F, 3.707F, 2.546F, 1.996F,
            1.759F, 1.465F, 1.311F, 1.252F, 1.234F, 1.228F, 1.238F, 1.241F,
            1.237F, 1.222F, 1.201F, 1.184F, 1.174F, 1.159F};
        constexpr float t_dat[22] = {
            1.0e-4F, 2.0e-4F, 4.0e-4F, 7.0e-4F, 1.0e-3F, 2.0e-3F, 4.0e-3F,
            7.0e-3F, 1.0e-2F, 2.0e-2F, 4.0e-2F, 7.0e-2F, 1.0e-1F, 2.0e-1F,
            4.0e-1F, 7.0e-1F, 1.0F, 2.0F, 4.0F, 7.0F, 10.0F, 20.0F};
        int i_t = 21;
        while (i_t >= 0 && t_dat[i_t] >= ekin) {
            --i_t;
        }
        i_t = sycl::clamp(i_t, 0, 20);
        const auto t0 = t_dat[i_t];
        const auto e0 = t0 + electron_mass_MeV;
        const auto b2_small = t0 * (e0 + electron_mass_MeV) / (e0 * e0);
        const auto t1 = t_dat[i_t + 1];
        const auto e1 = t1 + electron_mass_MeV;
        const auto b2_big = t1 * (e1 + electron_mass_MeV) / (e1 * e1);
        const auto rat_b2 = (beta2 - b2_small) / (b2_big - b2_small);
        const auto cc1 = cpositron_z29[i_t];
        const auto cc2 = cpositron_z29[i_t + 1];
        sigma *= sigma_factor / (cc1 + rat_b2 * (cc2 - cc1));
    } else {
        constexpr float sig0_z29 = 13.24F * barn_cm2;  // Zdat[5]
        constexpr float hecorr_z29 = 74.510F;
        sigma = bg2_lim * sig0_z29 *
                (1.0F + hecorr_z29 * (beta2 - beta2_lim)) / bg2;
    }
    sigma *= 1.0F + 0.30F / (1.0F + sycl::sqrt(1000.0F * ekin));
    return sigma;
}

inline float copper_urban_transport_mfp_mm(
    const float kin_energy_MeV, const int projectile_charge,
    const int mass_number, const float density_g_per_cm3) noexcept {
    if (!(density_g_per_cm3 > 0.0F)) {
        return 0.0F;
    }
    const auto sigma = copper_urban_cross_section_per_atom_cm2(
        kin_energy_MeV, projectile_charge, mass_number);
    if (!(sigma > 0.0F)) {
        return 0.0F;
    }
    constexpr float avogadro = 6.02214076e23F;
    constexpr float copper_molar_mass = 63.546F;
    const auto number_density =
        avogadro * density_g_per_cm3 / copper_molar_mass;
    return 10.0F / (number_density * sigma);
}

struct CopperUrbanCoefficients {
    float coeffth1;
    float coeffth2;
    float coeffc1;
    float coeffc2;
    float coeffc3;
    float coeffc4;
};

inline CopperUrbanCoefficients copper_urban_coefficients(
    const float zeffective) noexcept {
    const auto w = sycl::pow(zeffective, 1.0F / 6.0F);
    const auto z13 = w * w;
    const auto facz = 0.990395F + w * (-0.168386F + w * 0.093286F);
    CopperUrbanCoefficients result{};
    result.coeffth1 = facz * (1.0F - 8.7780e-2F / zeffective);
    result.coeffth2 = facz * (4.0780e-2F + 1.7315e-4F * zeffective);
    result.coeffc1 = 2.3785F - z13 * (4.1981e-1F - z13 * 6.3100e-2F);
    result.coeffc2 = 4.7526e-1F + z13 * (1.7694F - z13 * 3.3885e-1F);
    result.coeffc3 = 2.3683e-1F - z13 * (1.8111F - z13 * 3.2774e-1F);
    result.coeffc4 = 1.7888e-2F + z13 * (1.9659e-2F - z13 * 2.6664e-3F);
    return result;
}

inline float copper_urban_theta0(
    const float path_length_mm, const float kin_energy_MeV,
    const float radiation_length_mm, const int projectile_charge,
    const int mass_number,
    const CopperUrbanCoefficients& coefficients) noexcept {
    const auto mass =
        static_cast<float>(mass_number) * static_cast<float>(nucleon_rest_mass_MeV);
    const auto invbetacp = (kin_energy_MeV + mass) /
                           (kin_energy_MeV * (kin_energy_MeV + 2.0F * mass));
    const auto y = path_length_mm / radiation_length_mm;
    const auto theta0 = 13.6F * static_cast<float>(projectile_charge) *
                        sycl::sqrt(y) * invbetacp;
    return theta0 * (coefficients.coeffth1 +
                     coefficients.coeffth2 * sycl::log(y));
}

CorrelatedScatteringStep copper_urban_msc_step(
    const Direction3F direction, const float kinetic_energy_MeV,
    const int atomic_number, const int mass_number, const float path_length_mm,
    const float density_g_per_cm3, const float radiation_length_g_per_cm2,
    const float range_mm, const float amplitude_scale, const std::uint64_t seed,
    const std::uint64_t history_id, const std::uint64_t step_index,
    const std::uint32_t random_dimension) noexcept {
    CorrelatedScatteringStep result{direction, Direction3F{0.0F, 0.0F, 0.0F}};
    if (kinetic_energy_MeV <= 0.0F || atomic_number <= 0 || mass_number <= 0 ||
        path_length_mm <= 0.0F || density_g_per_cm3 <= 0.0F ||
        radiation_length_g_per_cm2 <= 0.0F || amplitude_scale <= 0.0F) {
        return result;
    }

    const auto radiation_length_mm =
        10.0F * radiation_length_g_per_cm2 / density_g_per_cm3;
    const auto lambda0_mm = copper_urban_transport_mfp_mm(
        kinetic_energy_MeV, atomic_number, mass_number, density_g_per_cm3);
    if (!(radiation_length_mm > 0.0F && lambda0_mm > 0.0F)) {
        return result;
    }

    constexpr float copper_zeffective = 29.0F;
    const auto coefficients = copper_urban_coefficients(copper_zeffective);
    const auto tau = path_length_mm / lambda0_mm;
    constexpr float tau_big = 8.0F;
    constexpr float tau_small = 1.0e-16F;
    constexpr float num_lim = 0.01F;
    constexpr float one_third = 1.0F / 3.0F;
    constexpr float one_sixth = 1.0F / 6.0F;
    constexpr float one_twelfth = 1.0F / 12.0F;
    constexpr float t_limit_min = 1.0e-7F;  // mm (10 * 0.01 nm)
    constexpr float pi = 3.14159265358979323846F;
    constexpr float two_pi = 6.2831853071795864769F;

    float x_mean_th = 0.0F;
    float x2_mean_th = 0.0F;
    if (tau < num_lim) {
        x_mean_th = 1.0F - tau * (1.0F - 0.5F * tau);
        x2_mean_th = 1.0F - tau * (5.0F - 6.25F * tau) * one_third;
    } else {
        x_mean_th = sycl::exp(-tau);
        x2_mean_th = (1.0F + 2.0F * sycl::exp(-2.5F * tau)) * one_third;
    }
    const auto simple_scattering = [&](const std::uint32_t dimension) {
        const auto a = (2.0F * x_mean_th + 9.0F * x2_mean_th - 3.0F) /
                       (2.0F * x_mean_th - 3.0F * x2_mean_th + 1.0F);
        const auto probability = (a + 2.0F) * x_mean_th / a;
        const auto u0 = sycl::fmax(
            rng::uniform01(seed, history_id, step_index, dimension), 1.0e-12F);
        const auto u1 = rng::uniform01(
            seed, history_id, step_index, dimension + 1U);
        return u1 < probability
                   ? -1.0F + 2.0F * sycl::exp(sycl::log(u0) / (a + 1.0F))
                   : -1.0F + 2.0F * u0;
    };

    float cth = 1.0F;
    if (tau >= tau_big) {
        cth = -1.0F + 2.0F * rng::uniform01(
                             seed, history_id, step_index, random_dimension);
    } else if (tau >= tau_small) {
        float theta0;
        if (path_length_mm > t_limit_min) {
            theta0 = amplitude_scale * copper_urban_theta0(
                path_length_mm, kinetic_energy_MeV, radiation_length_mm,
                atomic_number, mass_number, coefficients);
        } else {
            theta0 = sycl::sqrt(path_length_mm / t_limit_min) *
                     amplitude_scale * copper_urban_theta0(
                         t_limit_min, kinetic_energy_MeV, radiation_length_mm,
                         atomic_number, mass_number, coefficients);
        }
        const auto theta2 = theta0 * theta0;
        if (theta2 < tau_small) {
            return result;
        }
        if (theta0 > pi * one_sixth) {
            cth = simple_scattering(random_dimension);
        } else {
            auto x = theta2 * (1.0F - theta2 * one_twelfth);
            if (theta2 > num_lim) {
                const auto sth = 2.0F * sycl::sin(0.5F * theta0);
                x = sth * sth;
            }
            const auto ltau = sycl::log(tau);
            const auto u = sycl::exp(ltau * one_sixth);
            const auto xx = sycl::log(lambda0_mm / radiation_length_mm);
            auto xsi = coefficients.coeffc1 +
                       u * (coefficients.coeffc2 + coefficients.coeffc3 * u) +
                       coefficients.coeffc4 * xx;
            xsi = sycl::fmax(xsi, 1.9F);
            auto c = xsi;
            if (sycl::fabs(c - 3.0F) < 0.001F) {
                c = 3.001F;
            } else if (sycl::fabs(c - 2.0F) < 0.001F) {
                c = 2.001F;
            }
            const auto c1 = c - 1.0F;
            const auto ea = sycl::exp(-xsi);
            const auto eaa = 1.0F - ea;
            const auto x_mean1 = 1.0F - (1.0F - (1.0F + xsi) * ea) * x / eaa;
            const auto x0 = 1.0F - xsi * x;
            if (x_mean1 <= 0.999F * x_mean_th) {
                cth = simple_scattering(random_dimension);
            } else {
                const auto b = 1.0F + (c - xsi) * x;
                const auto b1 = b + 1.0F;
                const auto bx = c * x;
                const auto eb1 = sycl::exp(sycl::log(b1) * c1);
                const auto ebx = sycl::exp(sycl::log(bx) * c1);
                const auto d = ebx / eb1;
                const auto x_mean2 =
                    (x0 + d - (bx - b1 * d) / (c - 2.0F)) / (1.0F - d);
                const auto f1x0 = ea / eaa;
                const auto f2x0 = c1 / (c * (1.0F - d));
                const auto probability = f2x0 / (f1x0 + f2x0);
                const auto q_probability =
                    x_mean_th / (probability * x_mean1 +
                                 (1.0F - probability) * x_mean2);
                const auto u0 = rng::uniform01(
                    seed, history_id, step_index, random_dimension);
                const auto u1 = rng::uniform01(
                    seed, history_id, step_index, random_dimension + 1U);
                if (u0 < q_probability) {
                    if (u1 < probability) {
                        cth = 1.0F + sycl::log(
                            ea + rng::uniform01(seed, history_id, step_index,
                                                random_dimension + 2U) *
                                     eaa) *
                                         x;
                    } else {
                        auto var = (1.0F - d) * rng::uniform01(
                            seed, history_id, step_index, random_dimension + 2U);
                        if (var < num_lim * d) {
                            var /= d * c1;
                            cth = -1.0F + var * (1.0F - 0.5F * var * c) *
                                              (2.0F + (c - xsi) * x);
                        } else {
                            cth = 1.0F + x * (c - xsi -
                                              c * sycl::exp(
                                                      -sycl::log(var + d) / c1));
                        }
                    }
                } else {
                    cth = -1.0F + 2.0F * u1;
                }
            }
        }
    }
    cth = sycl::clamp(cth, -1.0F, 1.0F);

    // direction and lateral displacement in the local step frame
    const auto sth = sycl::sqrt(sycl::fmax(0.0F, (1.0F - cth) * (1.0F + cth)));
    const auto phi = two_pi * rng::uniform01(
                                 seed, history_id, step_index,
                                 random_dimension + 3U);
    const auto r_max = path_length_mm * sth;
    float radius = 0.73F * r_max;
    float sqrt_x = 1.0F;
    constexpr float displacement_eps = 1.0e-3F;
    if (range_mm > 0.0F && r_max / range_mm >= displacement_eps) {
        constexpr float x0 = 0.73F;
        const auto alpha = sycl::log(7.33F) / x0;
        const auto a1 = 1.0F - x0;
        const auto a2 = 1.0F - sycl::exp(-alpha * x0);
        const auto a3 = sycl::exp(alpha * x0) - 1.0F;
        const auto w1 = 2.0F * a2 / (alpha * a1 + 2.0F * a2);
        const auto u0 = rng::uniform01(
            seed, history_id, step_index, random_dimension + 4U);
        const auto u1 = rng::uniform01(
            seed, history_id, step_index, random_dimension + 5U);
        const auto x = u0 < w1 ? sycl::log(1.0F + a3 * u1) / alpha
                               : 1.0F - a1 * sycl::sqrt(1.0F - u1);
        sqrt_x = sycl::sqrt(sycl::fmax(0.0F, x));
        radius = sqrt_x * r_max;
    }
    const auto sigma = 0.1F + 0.9F * sqrt_x;
    const auto normal_u0 = sycl::fmax(
        rng::uniform01(seed, history_id, step_index, random_dimension + 6U),
        1.0e-12F);
    const auto normal_u1 = rng::uniform01(
        seed, history_id, step_index, random_dimension + 7U);
    const auto psi = sigma * sycl::sqrt(-2.0F * sycl::log(normal_u0)) *
                     sycl::cos(two_pi * normal_u1);
    const auto phi_displacement = phi + psi;
    const auto local_displacement_x = radius * sycl::cos(phi_displacement);
    const auto local_displacement_y = radius * sycl::sin(phi_displacement);

    const auto parent_norm = sycl::sqrt(
        direction.x * direction.x + direction.y * direction.y +
        direction.z * direction.z);
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
        sth * sycl::cos(phi), sth * sycl::sin(phi), cth, direction);
    return result;
}

// ---------------------------------------------------------------------------
// Research-only Urban v2 for the Copper collimator.  Fixes the confirmed
// inequivalences of copper_urban_msc_step against the active Geant4 11.3.2
// branch (GenericIon -> UrbanMsc, StepLim=Minimal, DispFlag:1, Alg96=1):
//   D1/D2 Alg96 SampleDisplacement with rmax = sqrt(t^2 - z^2)
//   D3   z -> t conversion (ComputeTrueStepLength, par1<0 branch)
//   D6   step-end energy prediction
//   D7   FP32-stable delta = t - z, log1p/expm1
// The caller supplies the geometric step and an isotropic safety.  It is not
// wired to a production default.
// ---------------------------------------------------------------------------
CorrelatedScatteringStep copper_urban_v2_msc_step(
    const Direction3F direction, const float kinetic_energy_MeV,
    const int atomic_number, const int mass_number, const float geom_path_mm,
    const float density_g_per_cm3, const float radiation_length_g_per_cm2,
    const float range_mm, const float dedx_MeV_per_mm, const float safety_mm,
    const float amplitude_scale, const std::uint64_t seed,
    const std::uint64_t history_id, const std::uint64_t step_index,
    const std::uint32_t random_dimension) noexcept {
    CorrelatedScatteringStep result{direction, Direction3F{0.0F, 0.0F, 0.0F}};
    if (kinetic_energy_MeV <= 0.0F || atomic_number <= 0 || mass_number <= 0 ||
        geom_path_mm <= 0.0F || density_g_per_cm3 <= 0.0F ||
        radiation_length_g_per_cm2 <= 0.0F || amplitude_scale <= 0.0F) {
        return result;
    }

    constexpr float tausmall = 1.0e-16F;
    constexpr float tlimitminfix2 = 1.0e-6F;  // mm (1 nm)
    // NOTE: the legacy model geommin (1e-3 mm) must NOT gate the process
    // displacement branch (R2).  Reduction uses kUrbanV2GeomMinMm and the
    // physics floor kUrbanV2MinDisplacementMm (both UNVALIDATED placeholders
    // pending a real AlongStepDoIt trace).
    constexpr float pi = 3.14159265358979323846F;
    constexpr float two_pi = 6.2831853071795864769F;
    constexpr float copper_zeffective = 29.0F;
    const auto radiation_length_mm =
        10.0F * radiation_length_g_per_cm2 / density_g_per_cm3;
    const auto lambda0_mm = copper_urban_transport_mfp_mm(
        kinetic_energy_MeV, atomic_number, mass_number, density_g_per_cm3);
    if (!(radiation_length_mm > 0.0F && lambda0_mm > 0.0F)) {
        return result;
    }
    const auto coefficients = copper_urban_coefficients(copper_zeffective);

    // D3/R3: geometric -> true path via the shared stable helper, which
    // computes and saves delta on the conversion branch (no float(t)-float(g)).
    const auto true_path_state = copper_urban_v2_true_path_and_delta(
        geom_path_mm, lambda0_mm, range_mm);
    float true_path_mm = true_path_state.true_path_mm;
    const auto delta_path = true_path_state.delta_mm;
    result.proposed_true_path_mm = true_path_mm;
    result.final_true_path_mm = true_path_mm;
    result.final_geom_path_mm = geom_path_mm;
    result.stable_delta_mm = delta_path;
    result.limit_reason = 0;
    result.boundary_crossed = false;
    result.proposal_valid = true;
    // NOTE (open, R-range): range_mm here is still the caller placeholder
    // energy/stopping.  GetRange->ionisation->theRangeTableForLoss provenance
    // (material-cuts couple, restricted DEDX, inverse range, ion scaling)
    // needs a reference table + extractor fixture; not faked here.

    // D6 (OPEN, structure only): step-end energy prediction (SampleScattering).
    // Reference selects by t/range with GetEnergy(range-t) for large t/range,
    // E-t*DEDX above 1% range, plus lambda1/effective-tau and pre/post energy
    // combination in ComputeTheta0.  Without the restricted-loss range table
    // + extractor fixture only the unconditional linear fallback below is
    // available; it is NOT an equivalent implementation.  Real process
    // ordering (no unconditional use of post-random-loss Epost) also open.
    const auto step_end_energy = sycl::fmax(
        0.0F, kinetic_energy_MeV - true_path_mm * dedx_MeV_per_mm);
    if (true_path_mm <= tlimitminfix2 || step_end_energy <= 1.0e-6F) {
        return result;
    }

    // angular sampling (same core as copper_urban_msc_step)
    const auto tau = true_path_mm / lambda0_mm;
    float x_mean_th = 0.0F;
    float x2_mean_th = 0.0F;
    constexpr float num_lim = 0.01F;
    constexpr float one_third = 1.0F / 3.0F;
    constexpr float one_sixth = 1.0F / 6.0F;
    constexpr float one_twelfth = 1.0F / 12.0F;
    if (tau < num_lim) {
        x_mean_th = 1.0F - tau * (1.0F - 0.5F * tau);
        x2_mean_th = 1.0F - tau * (5.0F - 6.25F * tau) * one_third;
    } else {
        x_mean_th = sycl::exp(-tau);
        x2_mean_th = (1.0F + 2.0F * sycl::exp(-2.5F * tau)) * one_third;
    }
    const auto simple_scattering = [&](const std::uint32_t dimension) {
        const auto a = (2.0F * x_mean_th + 9.0F * x2_mean_th - 3.0F) /
                       (2.0F * x_mean_th - 3.0F * x2_mean_th + 1.0F);
        const auto probability = (a + 2.0F) * x_mean_th / a;
        const auto u0 = sycl::fmax(
            rng::uniform01(seed, history_id, step_index, dimension), 1.0e-12F);
        const auto u1 = rng::uniform01(
            seed, history_id, step_index, dimension + 1U);
        return u1 < probability
                   ? -1.0F + 2.0F * sycl::exp(sycl::log(u0) / (a + 1.0F))
                   : -1.0F + 2.0F * u0;
    };

    float cth = 1.0F;
    if (tau >= 8.0F) {
        cth = -1.0F + 2.0F * rng::uniform01(
                             seed, history_id, step_index, random_dimension);
    } else if (tau >= tausmall) {
        const auto theta0 = amplitude_scale * copper_urban_theta0(
            true_path_mm, step_end_energy, radiation_length_mm,
            atomic_number, mass_number, coefficients);
        const auto theta2 = theta0 * theta0;
        if (theta2 >= tausmall) {
            if (theta0 > pi * one_sixth) {
                cth = simple_scattering(random_dimension);
            } else {
                auto x = theta2 * (1.0F - theta2 * one_twelfth);
                if (theta2 > num_lim) {
                    const auto sth = 2.0F * sycl::sin(0.5F * theta0);
                    x = sth * sth;
                }
                const auto ltau = sycl::log(tau);
                const auto u = sycl::exp(ltau * one_sixth);
                const auto xx = sycl::log(lambda0_mm / radiation_length_mm);
                auto xsi = coefficients.coeffc1 +
                           u * (coefficients.coeffc2 +
                                coefficients.coeffc3 * u) +
                           coefficients.coeffc4 * xx;
                xsi = sycl::fmax(xsi, 1.9F);
                auto c = xsi;
                if (sycl::fabs(c - 3.0F) < 0.001F) {
                    c = 3.001F;
                } else if (sycl::fabs(c - 2.0F) < 0.001F) {
                    c = 2.001F;
                }
                const auto c1 = c - 1.0F;
                const auto ea = sycl::exp(-xsi);
                const auto eaa = 1.0F - ea;
                const auto x_mean1 =
                    1.0F - (1.0F - (1.0F + xsi) * ea) * x / eaa;
                const auto x0 = 1.0F - xsi * x;
                if (x_mean1 <= 0.999F * x_mean_th) {
                    cth = simple_scattering(random_dimension);
                } else {
                    const auto b = 1.0F + (c - xsi) * x;
                    const auto b1 = b + 1.0F;
                    const auto bx = c * x;
                    const auto eb1 = sycl::exp(sycl::log(b1) * c1);
                    const auto ebx = sycl::exp(sycl::log(bx) * c1);
                    const auto d = ebx / eb1;
                    const auto x_mean2 =
                        (x0 + d - (bx - b1 * d) / (c - 2.0F)) / (1.0F - d);
                    const auto f1x0 = ea / eaa;
                    const auto f2x0 = c1 / (c * (1.0F - d));
                    const auto probability = f2x0 / (f1x0 + f2x0);
                    const auto q_probability =
                        x_mean_th / (probability * x_mean1 +
                                     (1.0F - probability) * x_mean2);
                    const auto u0 = rng::uniform01(
                        seed, history_id, step_index, random_dimension);
                    const auto u1 = rng::uniform01(
                        seed, history_id, step_index, random_dimension + 1U);
                    if (u0 < q_probability) {
                        if (u1 < probability) {
                            cth = 1.0F + sycl::log(
                                ea + rng::uniform01(
                                         seed, history_id, step_index,
                                         random_dimension + 2U) *
                                         eaa) *
                                             x;
                        } else {
                            auto var = (1.0F - d) * rng::uniform01(
                                seed, history_id, step_index,
                                random_dimension + 2U);
                            if (var < num_lim * d) {
                                var /= d * c1;
                                cth = -1.0F + var * (1.0F - 0.5F * var * c) *
                                                  (2.0F + (c - xsi) * x);
                            } else {
                                cth = 1.0F + x * (c - xsi -
                                                  c * sycl::exp(
                                                          -sycl::log(var + d) /
                                                          c1));
                            }
                        }
                    } else {
                        cth = -1.0F + 2.0F * u1;
                    }
                }
            }
        }
    }
    cth = sycl::clamp(cth, -1.0F, 1.0F);
    // Save 1-cos(theta) BEFORE rounding-sensitive use (R3 angle part).  The
    // sqrt below still guards, but diagnostics must use this value.
    result.one_minus_cth = sycl::fmax(0.0F, 1.0F - cth);

    const auto sth = sycl::sqrt(sycl::fmax(0.0F, (1.0F - cth) * (1.0F + cth)));
    const auto phi = two_pi * rng::uniform01(
                                 seed, history_id, step_index,
                                 random_dimension + 3U);

    // D1/D2: Alg96 SampleDisplacement.  rmax^2 = t^2 - z^2 = delta*(2t-delta),
    // using the stable branch-saved delta (never recomputed as t-z).
    float local_displacement_x = 0.0F;
    float local_displacement_y = 0.0F;
    const auto rmax2 = delta_path * (2.0F * true_path_mm - delta_path);
    if (rmax2 > 0.0F) {
        constexpr float cbeta = 2.160F;
        const auto cbeta1 = 1.0F - sycl::exp(-cbeta * pi);
        const auto radius = 0.73F * sycl::sqrt(rmax2);
        const auto u0 = rng::uniform01(
            seed, history_id, step_index, random_dimension + 4U);
        const auto u1 = rng::uniform01(
            seed, history_id, step_index, random_dimension + 5U);
        const auto psi = -sycl::log1p(-u0 * cbeta1) / cbeta;
        const auto phi_displacement = u1 < 0.5F ? phi + psi : phi - psi;
        auto disp_x = radius * sycl::cos(phi_displacement);
        auto disp_y = radius * sycl::sin(phi_displacement);
        // D5/R2: process-level acceptance with distinct reference
        // thresholds (shared helper; no model-geommin conflation).
        float accepted_x = 0.0F;
        float accepted_y = 0.0F;
        int accept_branch = 0;
        copper_urban_v2_accept_displacement(
            disp_x, disp_y, safety_mm, kUrbanV2GeomMinMm,
            kUrbanV2MinDisplacementMm, accepted_x, accepted_y, accept_branch);
        local_displacement_x = accepted_x;
        local_displacement_y = accepted_y;
    }

    const auto parent_norm = sycl::sqrt(
        direction.x * direction.x + direction.y * direction.y +
        direction.z * direction.z);
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
        sth * sycl::cos(phi), sth * sycl::sin(phi), cth, direction);
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

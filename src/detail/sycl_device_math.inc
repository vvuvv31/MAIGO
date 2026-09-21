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
    // Urban raw vs process-accepted displacement accounting (diagnostic only).
    // raw_displacement_r_mm: |Urban raw| before the AlongStepDoIt gate.
    // displacement_branch: 0=zero, 1=r2<=minDisplacement2, 2=full accept,
    //   3=reduced, 4=cancelled.
    // cth_rounded_to_one: final float cth == 1.0F (information already lost
    //   there; the pre-rounding small quantity is one_minus_cth).
    float raw_displacement_r_mm{0.0F};
    int displacement_branch{0};
    bool cth_rounded_to_one{false};
    // Loss range at step start (theRangeTableForLoss semantics).
    float current_range_mm{0.0F};
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

// Stable small-angle sample (Fix B).  one_minus_cth is computed DIRECTLY
// from each sampling expression, never as 1-cth after the float cth has
// rounded to 1.  Only the near-+1 branches need this; far-from-1 branches
// carry the algebraically identical form for uniformity.
// FP32 semantics closest to the Geant4 double reference:
//   - direction uses sth = sqrt(omcth*(2-omcth)), exact in float even when
//     cth has rounded to 1.0F;
//   - the |cth|>=1 bad-angle protection (which in double zeroes the
//     displacement) is evaluated on omcth: bad iff !(omcth>0)||!(omcth<2),
//     i.e. only for a true zero/straight-back scatter, never for a merely
//     rounded cth==1.0F.  Bypassing the protection to force nonzero
//     displacement is forbidden; applying it to rounded floats would kill
//     almost every small-angle displacement, which is equally wrong.
struct UrbanV2AngularSample {
    float cth;
    float one_minus_cth;
};

// Process-level displacement acceptance: EXACT mirror of the
// G4VMultipleScattering::AlongStepDoIt displacement gate (Geant4 11.3.2,
// G4VMultipleScattering::G4VMultipleScattering ctor):
//   geomMin = 0.05*CLHEP::nm = 5e-8 mm
//   minDisplacement2 = geomMin*geomMin = 2.5e-15 mm^2
// Gate (double precision):
//   r2 = |raw|^2; if r2 <= minDisplacement2 -> no position change;
//   else dispR = sqrt(r2), postSafety = 0.99*isotropicSafety(endpoint, dispR);
//   if postSafety > 0 && dispR <= postSafety -> full accept;
//   else if postSafety > geomMin -> reduce by postSafety/dispR;
//   else -> cancel.
// The two thresholds are NOT independent tunables: minDisplacement2 is the
// SQUARE of the single geomMin constant. The earlier single 1e-3 mm model
// geommin and the 1e-6/1e-7 placeholders are deleted, not kept as options.
//
// Ownership is explicit: G4UrbanMscModel::SampleScattering(oldDirection,
// safety) does NOT use its safety argument in 11.3.2; boundary acceptance
// belongs to G4VMultipleScattering::AlongStepDoIt. MAIGO fuses sampling +
// process acceptance in one device helper for performance, but the comment
// and the branch accounting below keep "Urban raw displacement" vs
// "process-level accepted displacement" distinct. Do not claim the Urban
// sampler itself uses safety.
constexpr float kUrbanV2GeomMinMm = 5.0e-8F;  // 0.05 nm, G4 reference
constexpr double kUrbanV2MinDisplacement2Mm2 =
    2.5e-15;  // geomMin^2, G4 reference

// Diagnostic-only branch ids (no RNG effect).
// 0 = zero raw displacement, 1 = r2 <= minDisplacement2 (no position change),
// 2 = full accept, 3 = reduced to postSafety, 4 = cancelled.
inline float copper_urban_v2_accept_displacement(
    const float raw_x, const float raw_y, const float safety_mm,
    float& accepted_x, float& accepted_y, int& branch) noexcept {
    accepted_x = 0.0F;
    accepted_y = 0.0F;
    branch = 0;
    const double dx = static_cast<double>(raw_x);
    const double dy = static_cast<double>(raw_y);
    const double r2 = dx * dx + dy * dy;
    if (!(r2 > 0.0)) {
        return 0.0F;
    }
    if (r2 <= kUrbanV2MinDisplacement2Mm2) {
        branch = 1;
        return 0.0F;
    }
    const double disp_r = sycl::sqrt(r2);
    const double post =
        0.99 * static_cast<double>(safety_mm > 0.0F ? safety_mm : 0.0F);
    if (post > 0.0 && disp_r <= post) {
        accepted_x = raw_x;
        accepted_y = raw_y;
        branch = 2;
        return static_cast<float>(disp_r);
    }
    if (post > static_cast<double>(kUrbanV2GeomMinMm)) {
        const double scale = post / disp_r;
        accepted_x = static_cast<float>(dx * scale);
        accepted_y = static_cast<float>(dy * scale);
        branch = 3;
        return static_cast<float>(post);
    }
    branch = 4;
    return 0.0F;
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
// Faithful fMinimal / loss-range / internal-energy implementation for the
// Copper GenericIon Urban path (Geant4 11.3.2, verified file by file from
// the v11.3.2 sources fetched 2026-09-21; sha256 recorded in the task log):
//   G4UrbanMscModel::ComputeTruePathLengthLimit  (:434-667)
//   G4UrbanMscModel::ComputeGeomPathLength       (:671-728)
//   G4UrbanMscModel::ComputeTrueStepLength       (:732-773)
//   G4UrbanMscModel::SampleScattering            (:777-820)
//   G4UrbanMscModel::SampleCosineTheta           (:824-968)
//   G4UrbanMscModel::ComputeTheta0               (:972-1019)
//   G4UrbanMscModel::SampleDisplacement (Alg96)  (:1023-1045)
//   G4UrbanMscModel::{Randomizetlimit,ComputeStepmin,ComputeTlimitmin} (.hh)
//   G4VMultipleScattering::AlongStep{GPIL,DoIt}  (utils/src, :218-370)
//   G4VMscModel::{GetRange,GetEnergy,GetDEDX}    (utils/src, :139-219)
// Active branch only: GenericIon, UrbanMsc, StepLim=Minimal, Rfact=0.2,
// DispFlag=1, Alg96=1. Electron SafetyPlus/skin machinery is NOT ported.
// ---------------------------------------------------------------------------

// Exact Geant4 11.3.2 model constants (G4UrbanMscModel ctor + G4VMscModel.hh).
constexpr float kUrbanV2Dtrl = 0.05F;
constexpr float kUrbanV2Taulim = 1.0e-6F;
constexpr float kUrbanV2TlimitMinFix = 1.0e-8F;    // 0.01 nm
constexpr float kUrbanV2TlimitMinFix2 = 1.0e-6F;   // 1 nm
constexpr float kUrbanV2TauSmall = 1.0e-16F;
constexpr float kUrbanV2TauBig = 8.0F;
constexpr float kUrbanV2FacRange = 0.2F;  // MscMuHadRangeFactor (TOPAS log: Rfact=0.2)
constexpr float kUrbanV2LambdaLimit = 1.0F;  // mm (TOPAS log: Llim=1 mm)
constexpr float kUrbanV2CopperZeff = 29.0F;  // elemental G4_Cu: Zeff == Z exactly

// Loss-range table for C12 in Cu with the PRODUCTION material-cuts couple
// (theRangeTableForLoss semantics). Extracted from TOPAS/Geant4 11.3.2 via
// CarbonLossRangeNtuple (GetRangeFromRestricteDEDX / GetKinEnergy / GetDEDX
// with the default-region 0.05 mm cuts). Device arrays are ascending in E;
// R(E) is strictly increasing. No Geant4 runtime on device.
struct UrbanV2LossTable {
    const float* e_total_mev{nullptr};
    const float* range_mm{nullptr};
    const float* dedx_mev_per_mm{nullptr};
    int count{0};
};

inline int urban_v2_loss_locate(const float* xs, const int n, const float x) noexcept {
    if (n < 2 || xs == nullptr) return -1;
    if (x <= xs[0]) return 0;
    if (x >= xs[n - 1]) return n - 2;
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (xs[mid] <= x) lo = mid;
        else hi = mid;
    }
    return lo;
}

inline float urban_v2_loss_range(const UrbanV2LossTable& table,
                                 const float e_total_mev) noexcept {
    const int i = urban_v2_loss_locate(table.e_total_mev, table.count, e_total_mev);
    if (i < 0) return 0.0F;
    const float e0 = table.e_total_mev[i], e1 = table.e_total_mev[i + 1];
    const float r0 = table.range_mm[i], r1 = table.range_mm[i + 1];
    const float f = (e_total_mev - e0) / (e1 - e0);
    return r0 + sycl::clamp(f, 0.0F, 1.0F) * (r1 - r0);
}

inline float urban_v2_loss_energy(const UrbanV2LossTable& table,
                                  const float range_mm) noexcept {
    const int i = urban_v2_loss_locate(table.range_mm, table.count, range_mm);
    if (i < 0) return 0.0F;
    const float r0 = table.range_mm[i], r1 = table.range_mm[i + 1];
    const float e0 = table.e_total_mev[i], e1 = table.e_total_mev[i + 1];
    const float f = (range_mm - r0) / (r1 - r0);
    return e0 + sycl::clamp(f, 0.0F, 1.0F) * (e1 - e0);
}

inline float urban_v2_loss_dedx(const UrbanV2LossTable& table,
                                const float e_total_mev) noexcept {
    const int i = urban_v2_loss_locate(table.e_total_mev, table.count, e_total_mev);
    if (i < 0) return 0.0F;
    const float e0 = table.e_total_mev[i], e1 = table.e_total_mev[i + 1];
    const float d0 = table.dedx_mev_per_mm[i], d1 = table.dedx_mev_per_mm[i + 1];
    const float f = (e_total_mev - e0) / (e1 - e0);
    return d0 + sycl::clamp(f, 0.0F, 1.0F) * (d1 - d0);
}

// Persistent per-track fMinimal state (Phase 2). Only the active branch's
// cross-step state: tlimit (init 1e10 mm) and firstStep. Everything else
// (range, lambda0, par1..3) is recomputed per step. Electron-only skin/
// smallstep/rangeinit state is deliberately NOT carried.
struct UrbanV2TrackState {
    float tlimit_mm{1.0e10F};
    bool first_step{true};
};

// Copper cache numbers, derived at runtime from Zeff with the exact
// InitialiseModelCache formulas (no transcribed magic constants).
struct UrbanV2CopperCache {
    float doverrb;
    float factmin;
    float stepmina;
    float stepminb;
    float z23;
};

inline UrbanV2CopperCache copper_urban_v2_cache() noexcept {
    constexpr float zeff = kUrbanV2CopperZeff;
    UrbanV2CopperCache c{};
    c.doverrb = 1.15F - 9.76e-4F * zeff;
    c.factmin = 0.001F;  // dispAlg96 == true
    c.stepmina = 27.725F / (1.0F + 0.203F * zeff);
    c.stepminb = 6.152F / (1.0F + 0.111F * zeff);
    c.z23 = sycl::pow(zeff, 2.0F / 3.0F);
    return c;
}

// t -> g conversion WITH par capture (ComputeGeomPathLength mirror).
// Needs the loss table + lambda1 only on the par1>0 branch
// (t >= range*dtrl without E<mass or t==range); small production steps stay
// on the par1<0 branch exactly like the previous implementation.
struct UrbanV2TrueToGeom {
    float g_mm{0.0F};
    float par1{-1.0F};
    float par2{0.0F};
    float par3{0.0F};
};

inline UrbanV2TrueToGeom copper_urban_v2_true_to_geom(
    const float t_mm, const float lambda0_mm, const float range_mm,
    const float epre_mev, const int atomic_number, const int mass_number,
    const float density_g_per_cm3, const float mass_mev,
    const UrbanV2LossTable& table) noexcept {
    // Float32 + sycl:: math (device-safe): relative error ~1e-7 on g, far
    // below any physics need; the reference double differs negligibly here.
    UrbanV2TrueToGeom out{};
    if (!(t_mm > 0.0F) || !(lambda0_mm > 0.0F)) {
        return out;
    }
    const float tau = t_mm / lambda0_mm;
    float z;
    if (tau <= 1.0e-16F) {
        z = t_mm < lambda0_mm ? t_mm : lambda0_mm;
    } else if (t_mm < range_mm * 0.05F) {
        z = (tau < 1.0e-6F) ? t_mm * (1.0F - 0.5F * tau)
                             : lambda0_mm * (1.0F - sycl::exp(-tau));
    } else if (epre_mev < mass_mev || t_mm == range_mm) {
        const float par1 = 1.0F / range_mm;
        const float par2 = range_mm / lambda0_mm;
        const float par3 = 1.0F + par2;
        z = (t_mm < range_mm)
            ? (1.0F - sycl::exp(par3 * sycl::log(1.0F - t_mm / range_mm))) /
                  (par1 * par3)
            : 1.0F / (par1 * par3);
        out.par1 = par1;
        out.par2 = par2;
        out.par3 = par3;
    } else {
        const float rfin =
            sycl::fmax(range_mm - t_mm, 0.01F * range_mm);
        const float t1 = urban_v2_loss_energy(table, rfin);
        const float lam1 = (t1 > 0.0F)
            ? copper_urban_transport_mfp_mm(t1, atomic_number, mass_number,
                                            density_g_per_cm3)
            : 0.0F;
        const float par1 = (lambda0_mm - lam1) / (lambda0_mm * t_mm);
        if (!(par1 > 0.0F)) {
            // Documented guard: float quantization can give lambda1 >=
            // lambda0 (reference double never hits this); fall back to the
            // par1<0 branch rather than producing inf/NaN.
            z = (tau < 1.0e-6F) ? t_mm * (1.0F - 0.5F * tau)
                                 : lambda0_mm * (1.0F - sycl::exp(-tau));
        } else {
            const float par2 = 1.0F / (par1 * lambda0_mm);
            const float par3 = 1.0F + par2;
            z = (1.0F - sycl::exp(par3 * sycl::log(lam1 / lambda0_mm))) /
                (par1 * par3);
            out.par1 = par1;
            out.par2 = par2;
            out.par3 = par3;
        }
    }
    if (z > lambda0_mm) z = lambda0_mm;
    out.g_mm = z;
    return out;
}

// fMinimal limiter proposal (ComputeTruePathLengthLimit mirror for the
// active C12 branch). t_input is the external TRUE candidate
// (min of user ceiling / collision / ... converted to true length).
// Diagnostic exit_path: 0=tiny, 1=range-vs-safety, 2=distance, 3=full.
struct UrbanV2LimitProposal {
    float t_msc_mm{0.0F};
    float g_msc_mm{0.0F};
    float range_mm{0.0F};
    float lambda0_mm{0.0F};
    float tlimitmin_mm{0.0F};
    float tlimit_used_mm{0.0F};
    float par1{-1.0F};
    float par2{0.0F};
    float par3{0.0F};
    bool limited_by_msc{false};
    bool no_displacement{false};
    bool valid{false};
    int exit_path{-1};
};

inline UrbanV2LimitProposal copper_urban_v2_limit_step(
    const float t_input_true_mm, const float epre_mev,
    const int atomic_number, const int mass_number,
    const float density_g_per_cm3, const float safety_start_mm,
    const bool at_geom_boundary, UrbanV2TrackState& state,
    const UrbanV2LossTable& table, const std::uint64_t seed,
    const std::uint64_t history_id, const std::uint64_t step_index) noexcept {
    UrbanV2LimitProposal out{};
    if (!(t_input_true_mm > 0.0F) || !(epre_mev > 0.0F)) {
        return out;
    }
    const float mass_mev =
        static_cast<float>(mass_number) * static_cast<float>(nucleon_rest_mass_MeV);
    const float range = urban_v2_loss_range(table, epre_mev);
    const float lambda0 = copper_urban_transport_mfp_mm(
        epre_mev, atomic_number, mass_number, density_g_per_cm3);
    if (!(range > 0.0F) || !(lambda0 > 0.0F)) {
        return out;
    }
    out.range_mm = range;
    out.lambda0_mm = lambda0;
    float t = sycl::fmin(t_input_true_mm, range);

    // Extreme small step: no displacement, convert, firstStep NOT cleared
    // (early return skips it in the reference).
    if (t < kUrbanV2TlimitMinFix) {
        const auto conv = copper_urban_v2_true_to_geom(
            t, lambda0, range, epre_mev, atomic_number, mass_number,
            density_g_per_cm3, mass_mev, table);
        out.t_msc_mm = t;
        out.g_msc_mm = conv.g_mm;
        out.no_displacement = true;
        out.valid = true;
        out.exit_path = 0;
        return out;
    }

    // presafety: full isotropic safety, NO cap at t (the t==range branch
    // below is reachable only if safety can exceed t; a capped presafety
    // would make it dead code, which it is not in the reference).
    const float presafety = safety_start_mm;

    // Stop here if small step or range is less than safety.
    if ((t == range && t < presafety) || t < kUrbanV2TlimitMinFix) {
        const auto conv = copper_urban_v2_true_to_geom(
            t, lambda0, range, epre_mev, atomic_number, mass_number,
            density_g_per_cm3, mass_mev, table);
        out.t_msc_mm = t;
        out.g_msc_mm = conv.g_mm;
        out.no_displacement = true;
        out.valid = true;
        out.exit_path = 1;
        return out;
    }

    // Far from geometry boundary: no MSC limitation needed.
    const auto cache = copper_urban_v2_cache();
    const float distance = range * cache.doverrb;
    if (distance < presafety) {
        const auto conv = copper_urban_v2_true_to_geom(
            t, lambda0, range, epre_mev, atomic_number, mass_number,
            density_g_per_cm3, mass_mev, table);
        out.t_msc_mm = t;
        out.g_msc_mm = conv.g_mm;
        out.no_displacement = true;
        out.valid = true;
        out.exit_path = 2;
        return out;
    }

    // fMinimal: persistent tlimit, recomputed ONLY at a geometry boundary
    // as facrange*max(range, lambda0), floored by tlimitmin. (The earlier
    // doc note claiming min() was wrong; the reference takes max().)
    const float rat = epre_mev;  // total kinetic MeV, invmev == 1/MeV
    const float stepmin = lambda0 * cache.factmin /
        (0.002F + rat * (cache.stepmina + cache.stepminb * rat));
    // tlow=5keV branch skipped: transport cutoff (0.1 MeV) >> tlow.
    const float tlimitmin =
        sycl::fmax(0.87F * cache.z23 * stepmin, kUrbanV2TlimitMinFix);
    out.tlimitmin_mm = tlimitmin;
    float tlimit = state.tlimit_mm;
    if (at_geom_boundary) {
        tlimit = kUrbanV2FacRange *
            sycl::fmax(range, lambda0);
        tlimit = sycl::fmax(tlimit, tlimitmin);
        state.tlimit_mm = tlimit;
    }
    out.tlimit_used_mm = tlimit;
    if (tlimit < t) {
        // Randomizetlimit: Gauss(tlimit, 0.1*(tlimit-tlimitmin)), floored.
        // Dedicated dims 58/59 (verified free in transport_sycl.cpp).
        const float u0 = sycl::fmax(
            rng::uniform01(seed, history_id, step_index, 58U), 1.0e-12F);
        const float u1 = rng::uniform01(seed, history_id, step_index, 59U);
        constexpr float gauss_pi = 6.2831853071795864769F;
        const float gauss = sycl::sqrt(-2.0F * sycl::log(u0)) *
            sycl::cos(gauss_pi * u1);
        float rt = tlimit + 0.1F * (tlimit - tlimitmin) * gauss;
        rt = sycl::fmax(rt, tlimitmin);
        const float t_new = sycl::fmin(t, rt);
        out.limited_by_msc = (t_new < t);
        t = t_new;
    }
    const auto conv = copper_urban_v2_true_to_geom(
        t, lambda0, range, epre_mev, atomic_number, mass_number,
        density_g_per_cm3, mass_mev, table);
    out.t_msc_mm = t;
    out.g_msc_mm = conv.g_mm;
    out.par1 = conv.par1;
    out.par2 = conv.par2;
    out.par3 = conv.par3;
    out.valid = true;
    out.exit_path = 3;
    state.first_step = false;
    return out;
}

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
// Full SampleScattering mirror (Phase 4, Geant4 11.3.2 :777-820 +
// SampleCosineTheta :824-968 with effective-tau, relloss, tsmall/extreme,
// geometric-mean invbetacp and lambdaeff). t_final_mm is the DoIt-level
// resolved true length; the caller (propose_and_sample) owns the GPIL/DoIt
// order, the t<range sampling gate and the endpoint postSafety.
// The legacy copper_urban_v2_msc_step below is kept byte-identical as the
// frozen tested unit for the pre-fMinimal semantics; this function is the
// production path. RNG dimensions/order match the legacy core exactly.
CorrelatedScatteringStep copper_urban_v2_sample_full(
    const Direction3F direction, const float epre_mev,
    const int atomic_number, const int mass_number,
    const float t_final_mm, const float delta_final_mm,
    const float density_g_per_cm3,
    const float radiation_length_g_per_cm2, const float range_mm,
    const float lambda0_mm, const float tlimitmin_mm,
    const UrbanV2LossTable& table, const float safety_end_mm,
    const float amplitude_scale, const bool allow_displacement,
    const std::uint64_t seed, const std::uint64_t history_id,
    const std::uint64_t step_index,
    const std::uint32_t random_dimension) noexcept {
    CorrelatedScatteringStep result{direction, Direction3F{0.0F, 0.0F, 0.0F}};
    if (epre_mev <= 0.0F || atomic_number <= 0 || mass_number <= 0 ||
        t_final_mm <= 0.0F || density_g_per_cm3 <= 0.0F ||
        radiation_length_g_per_cm2 <= 0.0F || amplitude_scale <= 0.0F) {
        return result;
    }
    constexpr float tausmall = 1.0e-16F;
    constexpr float tlimitminfix = 1.0e-8F;
    constexpr float pi = 3.14159265358979323846F;
    constexpr float two_pi = 6.2831853071795864769F;
    constexpr float copper_zeffective = 29.0F;
    const float mass_mev =
        static_cast<float>(mass_number) * static_cast<float>(nucleon_rest_mass_MeV);
    const auto radiation_length_mm =
        10.0F * radiation_length_g_per_cm2 / density_g_per_cm3;
    if (!(radiation_length_mm > 0.0F && lambda0_mm > 0.0F &&
          range_mm > 0.0F)) {
        return result;
    }
    const auto coefficients = copper_urban_coefficients(copper_zeffective);

    // SampleScattering: no scattering at/over range.
    if (t_final_mm >= range_mm) {
        return result;
    }
    // Internal energy prediction (:784-790). Restricted DEDX from the loss
    // table (GetDEDX with the production couple), NOT the caller's
    // condensed-loss stopping and NOT post-straggling Epost (ordering kept).
    float kin_energy = epre_mev;
    if (t_final_mm > range_mm * kUrbanV2Dtrl) {
        kin_energy = urban_v2_loss_energy(table, range_mm - t_final_mm);
    } else if (t_final_mm > range_mm * 0.01F) {
        kin_energy -= t_final_mm * urban_v2_loss_dedx(table, epre_mev);
    }
    if (kin_energy < 0.0F) kin_energy = 0.0F;
    if ((t_final_mm <= tlimitminfix) || (t_final_mm < tausmall * lambda0_mm) ||
        (kin_energy <= 1.0e-6F)) {
        return result;
    }

    // Effective tau with lambda1 (:831-839).
    float tau = t_final_mm / lambda0_mm;
    float lambdaeff = lambda0_mm;
    if (kin_energy != epre_mev) {
        const float lambda1 = copper_urban_transport_mfp_mm(
            kin_energy, atomic_number, mass_number, density_g_per_cm3);
        if (lambda1 > 0.0F &&
            sycl::fabs(lambda1 - lambda0_mm) > lambda0_mm * 0.01F) {
            tau = t_final_mm * sycl::log(lambda0_mm / lambda1) /
                (lambda0_mm - lambda1);
            lambdaeff = t_final_mm / tau;
        }
    }
    const float current_tau = tau;

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
        UrbanV2AngularSample out{};
        if (u1 < probability) {
            const auto e =
                sycl::exp(sycl::log(u0) / (a + 1.0F));
            out.cth = -1.0F + 2.0F * e;
            out.one_minus_cth = 2.0F * (1.0F - e);
        } else {
            out.cth = -1.0F + 2.0F * u0;
            out.one_minus_cth = 2.0F * (1.0F - u0);
        }
        return out;
    };

    float cth = 1.0F;
    float omcth = 0.0F;
    if (tau >= kUrbanV2TauBig) {
        const auto u = rng::uniform01(
            seed, history_id, step_index, random_dimension);
        cth = -1.0F + 2.0F * u;
        omcth = 2.0F * (1.0F - u);
    } else if (tau >= tausmall) {
        // Too-large relative loss falls back to isotropic-ish sampling.
        const float relloss = 1.0F - kin_energy / epre_mev;
        if (relloss > 0.50F) {
            const auto sampled = simple_scattering(random_dimension);
            cth = sampled.cth;
            omcth = sampled.one_minus_cth;
        } else {
            const float tsmall =
                sycl::fmin(tlimitmin_mm, kUrbanV2LambdaLimit);
            const bool extreme_step = !(t_final_mm > tsmall);
            // ComputeTheta0 mirror with pre/post geometric-mean invbetacp.
            const float use_t =
                extreme_step ? tsmall : t_final_mm;
            float inv = (kin_energy + mass_mev) /
                (kin_energy * (kin_energy + 2.0F * mass_mev));
            if (kin_energy != epre_mev) {
                inv = sycl::sqrt(inv * (epre_mev + mass_mev) /
                                 (epre_mev * (epre_mev + 2.0F * mass_mev)));
            }
            const float y = use_t / radiation_length_mm;
            float theta0 = 13.6F * static_cast<float>(atomic_number) *
                sycl::sqrt(y) * inv;
            theta0 *= (coefficients.coeffth1 +
                       coefficients.coeffth2 * sycl::log(y));
            if (extreme_step) {
                theta0 *= sycl::sqrt(t_final_mm / tsmall);
            }
            theta0 *= amplitude_scale;
            const auto theta2 = theta0 * theta0;
            if (theta2 >= tausmall) {
                if (theta0 > pi * one_sixth) {
                    const auto sampled = simple_scattering(random_dimension);
                    cth = sampled.cth;
                    omcth = sampled.one_minus_cth;
                } else {
                    auto x = theta2 * (1.0F - theta2 * one_twelfth);
                    if (theta2 > num_lim) {
                        const auto sth_tmp = 2.0F * sycl::sin(0.5F * theta0);
                        x = sth_tmp * sth_tmp;
                    }
                    const auto ltau = sycl::log(tau);
                    const auto u = extreme_step
                        ? sycl::exp(sycl::log(tsmall / lambda0_mm) * one_sixth)
                        : sycl::exp(ltau * one_sixth);
                    const auto xx = sycl::log(lambdaeff / radiation_length_mm);
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
                        const auto sampled = simple_scattering(random_dimension);
                        cth = sampled.cth;
                        omcth = sampled.one_minus_cth;
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
                                const auto u2 = rng::uniform01(
                                    seed, history_id, step_index,
                                    random_dimension + 2U);
                                const auto log_arg = ea + u2 * eaa;
                                const auto log_val = sycl::log(log_arg);
                                cth = 1.0F + log_val * x;
                                omcth = -log_val * x;
                            } else {
                                auto var = (1.0F - d) * rng::uniform01(
                                    seed, history_id, step_index,
                                    random_dimension + 2U);
                                if (var < num_lim * d) {
                                    var /= d * c1;
                                    const auto q = var * (1.0F - 0.5F * var * c) *
                                                   (2.0F + (c - xsi) * x);
                                    cth = -1.0F + q;
                                    omcth = 2.0F - q;
                                } else {
                                    const auto e = c * sycl::exp(
                                        -sycl::log(var + d) / c1);
                                    const auto dq = x * (c - xsi - e);
                                    cth = 1.0F + dq;
                                    omcth = -dq;
                                }
                            }
                        } else {
                            cth = -1.0F + 2.0F * u1;
                            omcth = 2.0F * (1.0F - u1);
                        }
                    }
                }
            }
        }
    }
    cth = sycl::clamp(cth, -1.0F, 1.0F);
    result.one_minus_cth = omcth;
    result.cth_rounded_to_one = (cth == 1.0F);

    const auto sth = sycl::sqrt(
        sycl::fmax(0.0F, omcth * (2.0F - omcth)));
    const auto phi = two_pi * rng::uniform01(
                                 seed, history_id, step_index,
                                 random_dimension + 3U);

    // Displacement: latDisplasment && currentTau >= tausmall, Alg96 active
    // (SampleDisplacement mirror). Urban RAW displacement uses no safety;
    // the process gate (AlongStepDoIt) follows in the shared helper.
    float local_displacement_x = 0.0F;
    float local_displacement_y = 0.0F;
    result.raw_displacement_r_mm = 0.0F;
    result.displacement_branch = 0;
    result.stable_delta_mm = delta_final_mm;
    const auto bad_cth = !(omcth > 0.0F) || !(omcth < 2.0F);
    const auto rmax2 = delta_final_mm * (2.0F * t_final_mm - delta_final_mm);
    if (!bad_cth && allow_displacement && current_tau >= tausmall &&
        rmax2 > 0.0F) {
        constexpr float cbeta = 2.160F;
        const auto cbeta1 = 1.0F - sycl::exp(-cbeta * pi);
        const auto radius = 0.73F * sycl::sqrt(rmax2);
        const auto u0 = rng::uniform01(
            seed, history_id, step_index, random_dimension + 4U);
        const auto u1 = rng::uniform01(
            seed, history_id, step_index, random_dimension + 5U);
        // -log1p(-u0*cbeta1): matches -log(1-u0*cbeta1) to float precision
        // with the same draws; dims rd+4/rd+5 consumed only when displacing,
        // exactly like flatArray(2) inside SampleDisplacement.
        const auto psi = -sycl::log1p(-u0 * cbeta1) / cbeta;
        const auto phi_displacement = u1 < 0.5F ? phi + psi : phi - psi;
        const auto raw_x = radius * sycl::cos(phi_displacement);
        const auto raw_y = radius * sycl::sin(phi_displacement);
        result.raw_displacement_r_mm =
            sycl::sqrt(raw_x * raw_x + raw_y * raw_y);
        float accepted_x = 0.0F;
        float accepted_y = 0.0F;
        int accept_branch = 0;
        copper_urban_v2_accept_displacement(
            raw_x, raw_y, safety_end_mm, accepted_x, accepted_y, accept_branch);
        local_displacement_x = accepted_x;
        local_displacement_y = accepted_y;
        result.displacement_branch = accept_branch;
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

struct UrbanV2GeomCtx {
    float cos_a{1.0F};
    float sin_a{0.0F};
    float radius_mm{0.0F};
    int slit_count{0};
    float slit_width_mm{0.0F};
    float slit_pitch_mm{0.0F};
    float slit_half_len_mm{0.0F};
    float slit_offset_mm{0.0F};
    float block_entrance_z_mm{0.0F};
    float block_exit_z_mm{0.0F};
};

// ComputeTrueStepLength mirror for a geometry-truncated step: invert the
// FINAL geometric length with the limiter's par set, clamped to
// [g_final, t_msc]. Untruncated steps return t_msc untouched (the
// geomStepLength == zPathLength shortcut).
inline float copper_urban_v2_finalize_true(
    const float g_final_mm, const float g_prop_mm, const float t_msc_mm,
    const float lambda0_mm, const float range_mm, const float par1,
    const float par3) noexcept {
    if (!(g_final_mm >= g_prop_mm)) {
        if (g_final_mm < 1.0e-6F) {
            return g_final_mm;
        }
        if (g_final_mm > lambda0_mm * 1.0e-16F) {
            float t;
            if (par1 < 0.0F) {
                t = -lambda0_mm * sycl::log(1.0F - g_final_mm / lambda0_mm);
            } else {
                const float par4 = par1 * par3;
                t = (par4 * g_final_mm < 1.0F)
                    ? (1.0F - sycl::exp(sycl::log(1.0F - par4 * g_final_mm) /
                                        par3)) /
                          par1
                    : range_mm;
            }
            if (t < g_final_mm) t = g_final_mm;
            if (t > t_msc_mm) t = t_msc_mm;
            return t;
        }
    } else {
        return t_msc_mm;
    }
    return g_final_mm;
}

// Full per-step GPIL+DoIt orchestration (Phase 5):
//   external TRUE candidate (user ceiling) -> fMinimal limit -> true->geom
//   -> geometry truncation -> geom->final-true -> scatter.
// Binding (limit_reason): 1=user/external, 2=msc, 3=geometry, 4=range (no
// scatter sampled). min() commutes, so evaluating geometry reachability
// before the limiter (our loop shape) yields the same minimum as the
// reference order; the comment chain records the mapping.
CorrelatedScatteringStep copper_urban_v2_propose_and_sample(
    const Direction3F direction, const float epre_mev,
    const int atomic_number, const int mass_number,
    const float user_ceiling_mm, const float boundary_path_mm,
    const float pos_x_mm, const float pos_y_mm, const float pos_z_mm,
    const float dir_x, const float dir_y, const float dir_z,
    const UrbanV2GeomCtx& geom, const bool at_boundary,
    UrbanV2TrackState& state, const UrbanV2LossTable& table,
    const float density_g_per_cm3, const float radiation_length_g_per_cm2,
    const float amplitude_scale, const std::uint64_t seed,
    const std::uint64_t history_id, const std::uint64_t step_index,
    const std::uint32_t random_dimension) noexcept {
    CorrelatedScatteringStep result{direction, Direction3F{0.0F, 0.0F, 0.0F}};
    if (!(epre_mev > 0.0F) || !(user_ceiling_mm > 0.0F) ||
        !(boundary_path_mm > 0.0F) || table.count < 2 ||
        table.e_total_mev == nullptr) {
        return result;
    }
    const float lambda0_now = copper_urban_transport_mfp_mm(
        epre_mev, atomic_number, mass_number, density_g_per_cm3);
    if (!(lambda0_now > 0.0F)) {
        return result;
    }
    // Step-start safety for the limiter presafety (full isotropic, uncapped).
    const float safety_start = minibeam_isotropic_safety_at_point(
        pos_x_mm, pos_y_mm, pos_z_mm, geom.cos_a, geom.sin_a, geom.radius_mm,
        geom.slit_count, geom.slit_width_mm, geom.slit_pitch_mm,
        geom.slit_half_len_mm, geom.slit_offset_mm, geom.block_entrance_z_mm,
        geom.block_exit_z_mm);
    // External TRUE candidate from the user ceiling (par1<0 inversion;
    // ceiling 0.25 << lambda0 ~ 2e6 so the small-ratio branch is exact).
    const float t_ext =
        copper_urban_v2_true_path_and_delta(user_ceiling_mm, lambda0_now,
                                            1.0e30F)
            .true_path_mm;
    auto limit = copper_urban_v2_limit_step(
        t_ext, epre_mev, atomic_number, mass_number, density_g_per_cm3,
        safety_start, at_boundary, state, table, seed, history_id, step_index);
    if (!limit.valid) {
        return result;
    }
    const float g_prop = limit.g_msc_mm;
    const float g_final =
        g_prop < boundary_path_mm ? g_prop : boundary_path_mm;
    const bool truncated = (g_final < g_prop);
    float t_final = copper_urban_v2_finalize_true(
        g_final, g_prop, limit.t_msc_mm, limit.lambda0_mm, limit.range_mm,
        limit.par1, limit.par3);
    // AlongStepDoIt line 319: protection against wrong t->g->t conversion.
    if (t_final > t_ext) t_final = t_ext;
    // FP32 subtlety: t_final and g_final are bit-identical floats here
    // (true delta ~6e-10 << ulp(0.05) ~3.7e-9), so double(t)-double(g) is
    // EXACTLY zero and would kill rmax/displacement. The branch-correct
    // series delta (same helper as the tested R3 path) carries the real
    // small quantity; t/g fields coinciding is harmless (energy uses t).
    const float delta_final = copper_urban_v2_true_path_and_delta(
        g_final, limit.lambda0_mm, limit.range_mm).delta_mm;

    result.proposed_true_path_mm = limit.t_msc_mm;
    result.final_true_path_mm = t_final;
    result.final_geom_path_mm = g_final;
    result.stable_delta_mm = static_cast<float>(
        delta_final > 0.0 ? delta_final : 0.0);
    result.boundary_crossed = truncated;
    result.proposal_valid = true;
    result.current_range_mm = limit.range_mm;

    // DoIt sampling gate: t<range && t>geomMin, else identity (last stop or
    // geometry sliver). Binding 4 covers both no-scatter exits here.
    if (!(t_final < limit.range_mm && t_final > kUrbanV2GeomMinMm)) {
        result.limit_reason = 4;
        return result;
    }
    // Endpoint postSafety at the post-geometry position (fNewPosition).
    const float end_x = pos_x_mm + g_final * dir_x;
    const float end_y = pos_y_mm + g_final * dir_y;
    const float end_z = pos_z_mm + g_final * dir_z;
    const float safety_end = minibeam_isotropic_safety_at_point(
        end_x, end_y, end_z, geom.cos_a, geom.sin_a, geom.radius_mm,
        geom.slit_count, geom.slit_width_mm, geom.slit_pitch_mm,
        geom.slit_half_len_mm, geom.slit_offset_mm, geom.block_entrance_z_mm,
        geom.block_exit_z_mm);
    result = copper_urban_v2_sample_full(
        direction, epre_mev, atomic_number, mass_number, t_final,
        static_cast<float>(delta_final > 0.0 ? delta_final : 0.0),
        density_g_per_cm3, radiation_length_g_per_cm2, limit.range_mm,
        limit.lambda0_mm, limit.tlimitmin_mm, table, safety_end,
        amplitude_scale, !limit.no_displacement, seed, history_id, step_index,
        random_dimension);
    result.proposed_true_path_mm = limit.t_msc_mm;
    result.final_true_path_mm = t_final;
    result.final_geom_path_mm = g_final;
    result.stable_delta_mm = static_cast<float>(
        delta_final > 0.0 ? delta_final : 0.0);
    result.boundary_crossed = truncated;
    result.proposal_valid = true;
    result.current_range_mm = limit.range_mm;
    if (t_final >= limit.range_mm) {
        result.limit_reason = 4;
    } else if (truncated) {
        result.limit_reason = 3;
    } else if (limit.limited_by_msc) {
        result.limit_reason = 2;
    } else {
        result.limit_reason = 1;
    }
    return result;
}

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
        UrbanV2AngularSample out{};
        if (u1 < probability) {
            const auto e =
                sycl::exp(sycl::log(u0) / (a + 1.0F));
            out.cth = -1.0F + 2.0F * e;
            out.one_minus_cth = 2.0F * (1.0F - e);
        } else {
            out.cth = -1.0F + 2.0F * u0;
            out.one_minus_cth = 2.0F * (1.0F - u0);
        }
        return out;
    };

    float cth = 1.0F;
    float omcth = 0.0F;  // branch-direct 1-cos(theta), pre-rounding
    if (tau >= 8.0F) {
        const auto u = rng::uniform01(
            seed, history_id, step_index, random_dimension);
        cth = -1.0F + 2.0F * u;
        omcth = 2.0F * (1.0F - u);
    } else if (tau >= tausmall) {
        const auto theta0 = amplitude_scale * copper_urban_theta0(
            true_path_mm, step_end_energy, radiation_length_mm,
            atomic_number, mass_number, coefficients);
        const auto theta2 = theta0 * theta0;
        if (theta2 >= tausmall) {
            if (theta0 > pi * one_sixth) {
                const auto sampled = simple_scattering(random_dimension);
                cth = sampled.cth;
                omcth = sampled.one_minus_cth;
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
                    const auto sampled = simple_scattering(random_dimension);
                    cth = sampled.cth;
                    omcth = sampled.one_minus_cth;
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
                            const auto u2 = rng::uniform01(
                                seed, history_id, step_index,
                                random_dimension + 2U);
                            const auto log_arg = ea + u2 * eaa;
                            const auto log_val = sycl::log(log_arg);
                            cth = 1.0F + log_val * x;
                            omcth = -log_val * x;
                        } else {
                            auto var = (1.0F - d) * rng::uniform01(
                                seed, history_id, step_index,
                                random_dimension + 2U);
                            if (var < num_lim * d) {
                                var /= d * c1;
                                const auto q = var * (1.0F - 0.5F * var * c) *
                                               (2.0F + (c - xsi) * x);
                                cth = -1.0F + q;
                                omcth = 2.0F - q;
                            } else {
                                const auto e = c * sycl::exp(
                                    -sycl::log(var + d) / c1);
                                const auto dq = x * (c - xsi - e);
                                cth = 1.0F + dq;
                                omcth = -dq;
                            }
                        }
                    } else {
                        cth = -1.0F + 2.0F * u1;
                        omcth = 2.0F * (1.0F - u1);
                    }
                }
            }
        }
    }
    cth = sycl::clamp(cth, -1.0F, 1.0F);
    // Branch-direct small quantity (Fix B): omcth was computed from the
    // sampling expression at every site above, NOT as 1-cth here.  If float
    // cth already rounded to 1, that information is gone from cth but intact
    // in omcth.
    result.one_minus_cth = omcth;
    result.cth_rounded_to_one = (cth == 1.0F);

    // Accurate FP32 polar sine from the small quantity: exact even when cth
    // rounded to 1.  Clamp only guards float noise (omcth slightly outside
    // [0,2] maps to the bad-angle protection below, not to a silent zero).
    const auto sth = sycl::sqrt(
        sycl::fmax(0.0F, omcth * (2.0F - omcth)));
    const auto phi = two_pi * rng::uniform01(
                                 seed, history_id, step_index,
                                 random_dimension + 3U);

    // D1/D2: Alg96 SampleDisplacement.  rmax^2 = t^2 - z^2 = delta*(2t-delta),
    // using the stable branch-saved delta (never recomputed as t-z).
    // Geant4 |cth|>=1 protection, evaluated on omcth (double-closest
    // semantics): a merely rounded cth==1.0F must NOT zero the displacement.
    float local_displacement_x = 0.0F;
    float local_displacement_y = 0.0F;
    result.raw_displacement_r_mm = 0.0F;
    result.displacement_branch = 0;
    const auto bad_cth = !(omcth > 0.0F) || !(omcth < 2.0F);
    const auto rmax2 = delta_path * (2.0F * true_path_mm - delta_path);
    if (!bad_cth && rmax2 > 0.0F) {
        constexpr float cbeta = 2.160F;
        const auto cbeta1 = 1.0F - sycl::exp(-cbeta * pi);
        const auto radius = 0.73F * sycl::sqrt(rmax2);
        const auto u0 = rng::uniform01(
            seed, history_id, step_index, random_dimension + 4U);
        const auto u1 = rng::uniform01(
            seed, history_id, step_index, random_dimension + 5U);
        const auto psi = -sycl::log1p(-u0 * cbeta1) / cbeta;
        const auto phi_displacement = u1 < 0.5F ? phi + psi : phi - psi;
        // Urban RAW displacement (no safety involved: SampleScattering in
        // 11.3.2 ignores its safety argument).
        const auto raw_x = radius * sycl::cos(phi_displacement);
        const auto raw_y = radius * sycl::sin(phi_displacement);
        result.raw_displacement_r_mm =
            sycl::sqrt(raw_x * raw_x + raw_y * raw_y);
        // Process-level acceptance (AlongStepDoIt gate, shared helper).
        float accepted_x = 0.0F;
        float accepted_y = 0.0F;
        int accept_branch = 0;
        copper_urban_v2_accept_displacement(
            raw_x, raw_y, safety_mm, accepted_x, accepted_y, accept_branch);
        local_displacement_x = accepted_x;
        local_displacement_y = accepted_y;
        result.displacement_branch = accept_branch;
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

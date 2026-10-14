// SYCL transport helpers for Unified EM and independent beamline/recoil modules. Do not compile this file as a standalone translation unit.
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
                                                  water_radiation_length_g_per_cm2),
    const float rest_mass_override_MeV = 0.0F) noexcept {
    const float mass_per_u = rest_mass_override_MeV > 0.0F && mass_number > 0
        ? rest_mass_override_MeV / static_cast<float>(mass_number)
        : static_cast<float>(nucleon_rest_mass_MeV);
    if (kinetic_energy_MeV <= 0.0F || atomic_number <= 0 || mass_number <= 0 ||
        path_length_mm <= 0.0F || density_g_per_cm3 <= 0.0F ||
        radiation_length_g_per_cm2 <= 0.0F) {
        return 0.0F;
    }
    const auto energy_MeV_per_u = kinetic_energy_MeV / static_cast<float>(mass_number);
    const auto total_energy_MeV_per_u =
        energy_MeV_per_u + mass_per_u;
    const auto momentum_MeV_per_c_per_u = sycl::sqrt(
        energy_MeV_per_u *
        (energy_MeV_per_u + 2.0F * mass_per_u));
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

// --- Unified MSC transport model (Highland theta sampler -> MscTransportModel).
// First implementation: Highland + lateral displacement placeholder.
// A single MCS step returns BOTH the outgoing direction and the intra-step
// lateral displacement, instead of only theta:
//
//   pre-step state -> true step -> energy loss -> apply_msc() -> post-step
//
// Highland mode is endpoint scattering by definition: displacement_mm = 0 and
// the caller advances position along the pre-step direction:
//   position += seg_dir * step + displacement (== seg_dir * step here).
// Correlated models (Fermi-Eyges tail, Urban v2) fill a non-zero displacement
// through their own CorrelatedScatteringStep paths; they are selected before
// these Highland helpers are reached. Do NOT add a hinge/displacement to the
// Highland helpers: failed.md 2026-09-18 shows per-step Fermi-Eyges
// displacement retaining Highland angular variance worsens Bragg valley/PVDR,
// with retry allowed only for validated step-invariant scattering power.
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
    // True only when the accepted true path exhausted the same pre-step
    // active-loss range carried by current_range_mm.
    bool range_terminal{false};
    bool energy_prediction_valid{true};
    float pre_step_energy_mev{0.0F};
    float predicted_internal_energy_mev{0.0F};
    // Fix C1: set when any numeric guard repaired a conversion result
    // (g>t rounding fixup, finalize clamp trips). Counted by the transport
    // kernel into a diagnostic slot; reachable-domain tests assert zero.
    bool numeric_guard_tripped{false};
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
//   else dispR = sqrt(r2),
//        postSafety = 0.99*min(isotropicSafety(endpoint), dispR) — the
//        navigator caps ComputeSafety at maxLength=dispR (fix B2), so the
//        far-field accept is really a 0.99 reduction, not a full accept;
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
    // G4SafetyHelper::ComputeSafety(fNewPosition, dispR) caps the reported
    // safety at dispR (navigator maxLength; fix B2). Without the cap the
    // far-field branch wrongly fully accepts (reference scales EVERY
    // accepted displacement by 0.99); with it, branch 2 is reachable only
    // through the uncapped corner paths, exactly as in AlongStepDoIt.
    const double capped_safety =
        (safety_mm > 0.0F) ? sycl::fmin(static_cast<double>(safety_mm),
                                         disp_r)
                            : 0.0;
    const double post = 0.99 * capped_safety;
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

// ---------------------------------------------------------------------------
// Faithful fMinimal / loss-range / internal-energy implementation for the
// Copper GenericIon Urban path (Geant4 11.3.2, verified file by file from
// the v11.3.2 sources fetched 2026-09-21, recorded in the task log):
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

// Material data and the current packaged Urban/loss-process context.
struct UrbanV2Material {
    UrbanLossContext loss_context{};
    float zeff{29.0F};
    float radlen_mm{0.0F};  // must be set by the wrapper (guarded > 0)
    int projectile_z{6};
    int projectile_a{12};
    float mass_mev{12.0F * 931.49410242F};
    float density_g_per_cm3{8.96F};
};

struct UrbanV2MaterialCache {
    float doverrb{1.0F};
    float factmin{0.001F};
    float stepmina{1.0F};
    float stepminb{1.0F};
    float z23{1.0F};
};

inline UrbanV2MaterialCache urban_v2_material_cache(const float zeff) noexcept {
    UrbanV2MaterialCache c{};
    c.doverrb = 1.15F - 9.76e-4F * zeff;
    c.factmin = 0.001F;  // dispAlg96 == true
    c.stepmina = 27.725F / (1.0F + 0.203F * zeff);
    c.stepminb = 6.152F / (1.0F + 0.111F * zeff);
    c.z23 = sycl::pow(zeff, 2.0F / 3.0F);
    return c;
}

inline float urban_v2_transport_mfp(const UrbanV2Material& mat, float energy_mev) noexcept {
    return mat.loss_context.lambda(energy_mev);
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

// Safety context: kind 0 = Cu collimator (cylindrical + slit helper),
// kind 1 = axis-aligned water box. Uniform per launch; no divergence cost.
struct UrbanV2SafetyCtx {
    int kind{0};
    UrbanV2GeomCtx cu{};
    float box_x0{0.0F};
    float box_x1{0.0F};
    float box_y0{0.0F};
    float box_y1{0.0F};
    float box_z0{0.0F};
    float box_z1{0.0F};
    // Physical voxel face distance, negative for legacy whole-box/Cu contexts.
    float box_boundary_path_mm{-1.0F};
    // kind 2: water surrounding a full-depth rectangular daughter ROI.
    float inner_x0{},inner_x1{},inner_y0{},inner_y1{};
};

// A physical voxel face is owned by the outgoing direction at an exact tie.
// Correct the quotient estimate against representable face coordinates: the
// quotient alone loses low x bits when (x - grid_min) has a large offset.
struct UrbanVoxelAxis {
    int index{-1};
    float lo{0.0F}, hi{0.0F};
    bool valid{false};
};
inline float urban_voxel_face(const int i, const int count,
                              const float lo, const float hi,
                              const float pitch) noexcept {
    // Centre anchoring preserves the exact x=0 face in the symmetric TOPAS
    // grid. Adjacent cells call this same expression for their shared face.
    return sycl::fma(static_cast<float>(i) - 0.5F * count, pitch,
                     0.5F * (lo + hi));
}
inline UrbanVoxelAxis urban_voxel_axis(const float p, const float d,
                                       const float lo, const float hi,
                                       const float pitch, const int count) noexcept {
    UrbanVoxelAxis a{};
    if (!(pitch > 0.0F) || count < 1 || !sycl::isfinite(p) ||
        p < lo || p > hi || (p == lo && d < 0.0F) ||
        (p == hi && d >= 0.0F)) return a;
    int i = sycl::clamp(static_cast<int>(sycl::floor((p-lo)/pitch)), 0, count-1);
    for (int correction = 0; correction < 3; ++correction) {
        if (i < 0 || i >= count) return a;
        const float l = urban_voxel_face(i, count, lo, hi, pitch);
        const float h = urban_voxel_face(i+1, count, lo, hi, pitch);
        if (p < l || (p == l && d < 0.0F)) { --i; continue; }
        if (p > h || (p == h && d >= 0.0F)) { ++i; continue; }
        a.index=i; a.lo=l; a.hi=h; a.valid=true; return a;
    }
    return a;
}
inline bool urban_reaches_voxel_face(const float path, const float distance) noexcept {
    return distance > 0.0F && sycl::isfinite(distance) &&
        (path >= distance || distance-path <=
             2.0F * 1.1920928955078125e-7F * sycl::fabs(distance));
}
struct UrbanVoxelCell {
    UrbanV2SafetyCtx safety{};
    int ix{-1}, iy{-1}, iz{-1};
    float face_distance{0.0F};
    Direction3F face_position{};
    unsigned face_mask{0U};
    bool valid{false};
};
inline UrbanVoxelCell urban_voxel_cell(
    const Direction3F p, const Direction3F d,
    const Direction3F lo, const Direction3F hi,
    const Direction3F pitch, const int nx, const int ny, const int nz) noexcept {
    UrbanVoxelCell c{};
    const auto x=urban_voxel_axis(p.x,d.x,lo.x,hi.x,pitch.x,nx);
    const auto y=urban_voxel_axis(p.y,d.y,lo.y,hi.y,pitch.y,ny);
    const auto z=urban_voxel_axis(p.z,d.z,lo.z,hi.z,pitch.z,nz);
    if (!(x.valid && y.valid && z.valid)) return c;
    c.ix=x.index; c.iy=y.index; c.iz=z.index;
    c.safety.kind=1;
    c.safety.box_x0=x.lo; c.safety.box_x1=x.hi;
    c.safety.box_y0=y.lo; c.safety.box_y1=y.hi;
    c.safety.box_z0=z.lo; c.safety.box_z1=z.hi;
    constexpr float inf=std::numeric_limits<float>::infinity();
    c.face_position={d.x<0.0F?x.lo:x.hi,d.y<0.0F?y.lo:y.hi,d.z<0.0F?z.lo:z.hi};
    const float dx=d.x==0.0F?inf:(c.face_position.x-p.x)/d.x;
    const float dy=d.y==0.0F?inf:(c.face_position.y-p.y)/d.y;
    const float dz=d.z==0.0F?inf:(c.face_position.z-p.z)/d.z;
    c.face_distance=sycl::fmin(dx,sycl::fmin(dy,dz));
    if (!(c.face_distance>0.0F) || !sycl::isfinite(c.face_distance)) return c;
    if (urban_reaches_voxel_face(c.face_distance,dx)) c.face_mask|=1U;
    if (urban_reaches_voxel_face(c.face_distance,dy)) c.face_mask|=2U;
    if (urban_reaches_voxel_face(c.face_distance,dz)) c.face_mask|=4U;
    c.safety.box_boundary_path_mm=c.face_distance;
    c.valid=true;
    return c;
}
inline Direction3F urban_voxel_snap_endpoint(const UrbanVoxelCell& c,
                                              const float path,
                                              Direction3F p) noexcept {
    if (urban_reaches_voxel_face(path,c.face_distance)) {
        if (c.face_mask&1U) p.x=c.face_position.x;
        if (c.face_mask&2U) p.y=c.face_position.y;
        if (c.face_mask&4U) p.z=c.face_position.z;
    }
    return p;
}

inline float urban_v2_box_safety(const float x_mm, const float y_mm,
                                 const float z_mm,
                                 const UrbanV2SafetyCtx& ctx) noexcept {
    float s = ctx.box_x1 - x_mm;
    if (x_mm - ctx.box_x0 < s) s = x_mm - ctx.box_x0;
    if (ctx.box_y1 - y_mm < s) s = ctx.box_y1 - y_mm;
    if (y_mm - ctx.box_y0 < s) s = y_mm - ctx.box_y0;
    if (ctx.box_z1 - z_mm < s) s = ctx.box_z1 - z_mm;
    if (z_mm - ctx.box_z0 < s) s = z_mm - ctx.box_z0;
    return s > 0.0F ? s : 0.0F;
}

inline float urban_v2_safety_at(const float x_mm, const float y_mm,
                                const float z_mm,
                                const UrbanV2SafetyCtx& ctx) noexcept {
    if (ctx.kind == 1 || ctx.kind == 2) {
        const float outer=urban_v2_box_safety(x_mm, y_mm, z_mm, ctx);
        if(ctx.kind==1)return outer;
        const float dx=sycl::fmax(0.f,sycl::fmax(ctx.inner_x0-x_mm,x_mm-ctx.inner_x1));
        const float dy=sycl::fmax(0.f,sycl::fmax(ctx.inner_y0-y_mm,y_mm-ctx.inner_y1));
        return sycl::fmin(outer,sycl::sqrt(dx*dx+dy*dy));
    }
    return minibeam_isotropic_safety_at_point(
        x_mm, y_mm, z_mm, ctx.cu.cos_a, ctx.cu.sin_a, ctx.cu.radius_mm,
        ctx.cu.slit_count, ctx.cu.slit_width_mm, ctx.cu.slit_pitch_mm,
        ctx.cu.slit_half_len_mm, ctx.cu.slit_offset_mm,
        ctx.cu.block_entrance_z_mm, ctx.cu.block_exit_z_mm);
}

// Persistent per-track fMinimal state (Phase 2). Only the active branch's
// cross-step state: tlimit (init 1e10 mm) and firstStep. Everything else
// (range, lambda0, par1..3) is recomputed per step. Electron-only skin/
// smallstep/rangeinit state is deliberately NOT carried.
struct UrbanV2TrackState {
    float tlimit_mm{1.0e10F};
    // fMinimal NEVER recomputes tlimitmin (G4UrbanMscModel::StartTracking
    // sets 10*tlimitminfix once; the fMinimal branch of
    // ComputeTruePathLengthLimit has no ComputeStepmin/ComputeTlimitmin
    // call). Frozen here at the reference value; the per-step recompute
    // (E3: ~3e-3 mm, 3e4x too large) is removed as part of fix B3.
    float tlimitmin_mm{1.0e-7F};
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

// t -> g conversion WITH par capture (ComputeGeomPathLength mirror) AND the
// stable branch-consistent delta (fix B4: delta belongs to the taken
// conversion branch, computed in double from the same float inputs — never
// float(t)-float(g) post-hoc, which is exactly 0 below the float ulp).
// Needs the loss table + lambda1 only on the par1>0 branch
// (t >= range*dtrl without E<mass or t==range); small production steps stay
// on the par1<0 branch exactly like the previous implementation.
struct UrbanV2TrueToGeom {
    float g_mm{0.0F};
    float delta_mm{0.0F};
    float par1{-1.0F};
    float par2{0.0F};
    float par3{0.0F};
    bool range_power_branch{false};
    // Set when the g>t rounding guard below fires (<= 1 ulp repair with the
    // delta kept: normal reference-class protection, expected whenever
    // delta < ulp(t)/2; accuracy is asserted per-row by tests).
    bool rounding_fixup{false};
};

// Exact ulp-above (nextafter(x,+inf)-x) via bit arithmetic (fix C5).
// sycl::nextafter is NOT bit-exact on all backends (this toolchain's host
// CPU implementation steps 2 ulp; the PTX device one is exact). All C1/C2
// ulp thresholds (8-ulp gate, 4-ulp tolerance, 2-ulp dust) use this so the
// analysis holds bitwise on host and device.
inline float urban_ulp_above(const float x) noexcept {
    if (!sycl::isfinite(x)) return 0.0F;
    std::uint32_t b = sycl::bit_cast<std::uint32_t>(x);
    if (b == 0x7F800000U) return 0.0F;  // +inf: no ulp above.
    if (b == 0x80000000U) {
        b = 0x00000001U;  // -0.0 -> smallest denormal.
    } else if (b & 0x80000000U) {
        --b;  // negatives: toward +inf means magnitude shrinks.
    } else {
        ++b;
    }
    return sycl::bit_cast<float>(b) - x;
}

// Constant-lambda (t, delta) pair for Geant4's explicit small-t branch.
// Range-law branches keep their own exponent and never switch physics at an
// FP32 representability boundary. Cancellation-free: series below tau=1e-2,
// lambda*(tau+expm1(-tau)) above.
inline double urban_v2_const_lambda_delta_mm(const double td,
                                             const double lamd,
                                             const double taud,
                                             const float expm1_neg_tau) noexcept {
    if (taud < 1.0e-2) {
        const double series = 1.0 - taud * (1.0 / 3.0 - taud / 12.0);
        return td * td / (2.0 * lamd) * series;
    }
    return lamd * (taud + static_cast<double>(expm1_neg_tau));
}

// Stable delta for the Geant4 range branch. The branch condition guarantees
// u=t/R >= approximately 0.05. For small a, expand exp(a*log1p(-u)) in a; this is the
// same range-law function and avoids subtracting two O(a*u) terms to recover
// the O(a*u^2) delta. It does not switch to the constant-lambda law.
inline double urban_v2_range_branch_delta_mm(const double range_mm,
                                             const double lambda0_mm,
                                             const double t_mm) noexcept {
    if (!(range_mm > 0.0) || !(lambda0_mm > 0.0) || !(t_mm > 0.0) ||
        t_mm > range_mm) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (t_mm == range_mm) {
        const double a = range_mm / lambda0_mm;
        return range_mm * a / (1.0 + a);
    }
    const double a = range_mm / lambda0_mm;
    const double u = t_mm / range_mm;
    // The caller selects this branch using float(t) >= float(0.05*R).
    // Rounding that product can leave the exact double ratio just below
    // 0.05. The same analytic range law is valid throughout 0 < u < 1;
    // rejecting that representable boundary incorrectly creates NaN steps.
    if (!(u > 0.0) || !(u < 1.0) || !(a > 0.0) || !sycl::isfinite(a)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double q = 1.0 - u;
    const double log1mu = sycl::log1p(-u);
    double bracket;
    const double al = a * log1mu;
    if (a < 1.0e-3 && sycl::fabs(al) < 0.1) {
        // a*u + q*expm1(a*l), expanding expm1 through order 10. With
        // |a*l|<0.1, the first omitted term is below 4e-25 absolute.
        double term = al;
        double em1 = term;
        for (int n = 2; n <= 10; ++n) {
            term *= al / static_cast<double>(n);
            em1 += term;
        }
        bracket = a * u + q * em1;
    } else {
        bracket = a * u + q * sycl::expm1(al);
    }
    return range_mm / (1.0 + a) * bracket;
}

inline double urban_v2_parameter_branch_delta_mm(const double par1,
                                                  const double par3,
                                                  const double t_mm) noexcept {
    if (!(par1 > 0.0) || !(par3 > 1.0) || !(t_mm > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double u = par1 * t_mm;
    if (!(u > 0.0) || u > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (u == 1.0) {
        return t_mm * (par3 - 1.0) / par3;
    }
    const double bracket = par3 * u + sycl::expm1(par3 * sycl::log1p(-u));
    return bracket / (par1 * par3);
}

inline double urban_v2_inverse_power_delta_mm(const double par1,
                                               const double par3,
                                               const double geom_mm,
                                               const double range_mm,
                                               const double range_ratio_a = 0.0) noexcept {
    const bool range_power = range_ratio_a > 0.0;
    if (!(par1 > 0.0) || !(geom_mm > 0.0) ||
        (!range_power && !(par3 > 1.0)) ||
        (range_power && (!(range_mm > 0.0) ||
                         !sycl::isfinite(range_ratio_a))) ||
        !sycl::isfinite(par1 + par3 + geom_mm)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double c = range_power ? range_ratio_a / (1.0 + range_ratio_a)
                                 : 1.0 - 1.0 / par3;
    const double ratio = range_power ? geom_mm / range_mm : par1 * geom_mm;
    const double x = range_power ? ratio + range_ratio_a * ratio
                                 : par3 * ratio;
    const double b = range_power ? 1.0 - c : 1.0 / par3;
    if (x > 1.0) return std::numeric_limits<double>::quiet_NaN();
    if (x == 1.0) return c / par1;
    if (!(x > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    if (x < 1.0e-4) {
        // Binomial delta series, accumulated from its positive x^2 term so
        // the a*x^2 scale survives even when t and g share the same FP32 ulp.
        double coefficient = b * c / 2.0;
        double x_power = x * x;
        double sum = coefficient * x_power;
        for (int n = 3; n <= 10; ++n) {
            const double factor = range_power
                ? (static_cast<double>(n - 2) + c) / static_cast<double>(n)
                : (static_cast<double>(n - 1) - b) / static_cast<double>(n);
            coefficient *= factor;
            x_power *= x;
            sum += coefficient * x_power;
        }
        return sum / par1;
    }
    // Algebraically identical to [1-(1-x)^b-b*x]/par1, rewritten as
    // c*x - (1-x)*expm1(-c*log1p(-x)). This removes the O(x) subtraction
    // and remains the same inverse power law, including c -> 0 limits.
    const double q = 1.0 - x;
    return (c * x - q * sycl::expm1(-c * sycl::log1p(-x))) / par1;
}

inline UrbanV2TrueToGeom copper_urban_v2_true_to_geom(
    const float t_mm, const float lambda0_mm, const float range_mm,
    const float epre_mev, const UrbanV2Material& mat) noexcept {
    // Contract: 0 < t <= range (the limiter clamps t to range before every
    // production call, mirroring ComputeTruePathLengthLimit). g is accurate
    // to a few ulp on every branch (verified vs the executed G4 triple);
    // delta's derivation per branch is documented below.
    UrbanV2TrueToGeom out{};
    if (!(t_mm > 0.0F) || !(lambda0_mm > 0.0F) ||
        !sycl::isfinite(t_mm + lambda0_mm) ||
        !sycl::isfinite(range_mm + epre_mev)) {
        return out;
    }
    // Geant4 11.3.2 source boundary: ComputeGeomPathLength bypasses the
    // conversion for t <= 1 nm. The old g4_conv.csv was collected with an
    // unbound model and is invalid as a production loss/transport oracle.
    if (t_mm <= 1.0e-6F) {
        out.g_mm = t_mm;
        out.delta_mm = 0.0F;
        return out;
    }
    const float tau = t_mm / lambda0_mm;
    const double td = static_cast<double>(t_mm);
    const double lamd = static_cast<double>(lambda0_mm);
    const double taud = td / lamd;
    float z;
    double delta = 0.0;
    if (tau <= 1.0e-16F) {
        z = t_mm < lambda0_mm ? t_mm : lambda0_mm;
        delta = 0.0;  // tau<=1e-16 forces t<<lambda0 here; chord == path.
    } else if (t_mm < range_mm * 0.05F) {
        // Stable constant-lambda conversion (fix C1).  The old form
        // z = lambda0*(1-exp(-tau)) loses the chord in FP32: at tau = 2e-5
        // the (1-exp) subtraction carries ~6e-8 absolute error, i.e. a
        // lambda0*6e-8 error on z that DWARFS delta = t^2/2lambda0, so the
        // old "delta >= 100 ulps, hence double(t)-double(z) is safe" comment
        // was wrong (it bounded delta's own rounding, not z's input error).
        // Now: g = -lambda*expm1(-tau) (~1 ulp, no cancellation), delta from
        // the shared cancellation-free helper.
        const float e = sycl::expm1(-tau);
        z = -lambda0_mm * e;
        delta = urban_v2_const_lambda_delta_mm(td, lamd, taud, e);
    } else if (epre_mev < mat.mass_mev || t_mm == range_mm) {
        const float par1 = 1.0F / range_mm;
        const float par2 = range_mm / lambda0_mm;
        const float par3 = 1.0F + par2;
        const double a = static_cast<double>(range_mm) / lamd;
        const double u = td / static_cast<double>(range_mm);
        const double range_chord = t_mm == range_mm
            ? static_cast<double>(range_mm) / (1.0 + a)
            : -static_cast<double>(range_mm) *
                  sycl::expm1((1.0 + a) * sycl::log1p(-u)) / (1.0 + a);
        z = static_cast<float>(range_chord);
        out.par1 = par1;
        out.par2 = par2;
        out.par3 = par3;
        out.range_power_branch = true;
        // Preserve this branch's range law even when t and g round to the
        // same FP32 value. A representability threshold must not select a
        // different physical formula.
        delta = urban_v2_range_branch_delta_mm(
            static_cast<double>(range_mm), lamd, td);
    } else {
        const float rfin =
            sycl::fmax(range_mm - t_mm, 0.01F * range_mm);
        const float t1 = mat.loss_context.energy(rfin);
        const float lam1 =
            (t1 > 0.0F) ? urban_v2_transport_mfp(mat, t1) : 0.0F;
        const float par1 = (lambda0_mm - lam1) / (lambda0_mm * t_mm);
        if (!(par1 > 0.0F)) {
            // A non-positive parameter means the post-step loss context did
            // not reduce the transport mean free path as required here.
            // Preserve the failure for the accepted-step gate to report.
            out.g_mm = std::numeric_limits<float>::quiet_NaN();
            out.delta_mm = std::numeric_limits<float>::quiet_NaN();
            return out;
        } else {
            const float par2 = 1.0F / (par1 * lambda0_mm);
            const float par3 = 1.0F + par2;
            const double p1d = static_cast<double>(par1);
            const double p3d = static_cast<double>(par3);
            const double u = p1d * td;
            z = static_cast<float>(
                -sycl::expm1(p3d * sycl::log1p(-u)) / (p1d * p3d));
            out.par1 = par1;
            out.par2 = par2;
            out.par3 = par3;
            delta = urban_v2_parameter_branch_delta_mm(
                static_cast<double>(par1), static_cast<double>(par3), td);
        }
    }
    if (z > lambda0_mm) {
        z = lambda0_mm;
        delta = td - lamd;  // clamped chord: delta follows the clamp.
    }
    // Chord never exceeds path: with the expm1 form z > t is possible only
    // by <= 1 ulp product rounding, which is EXPECTED (not pathological)
    // whenever delta < ulp(t)/2. Repair honestly: snap the chord to t but
    // KEEP the separately-computed delta (a plain float, representable down
    // to 1e-38). Zeroing delta here would delete a representable quantity
    // exactly as the old B4 bug did. Repair error: g off by <= 1 ulp,
    // delta exact. Flagged for tests; production treats it as normal
    // reference-class protection (only finalize-class trips are counted as
    // numeric faults).
    if (z > t_mm) {
        z = t_mm;
        out.rounding_fixup = true;
    }
    if (!sycl::isfinite(delta) || delta < 0.0) {
        out.g_mm = std::numeric_limits<float>::quiet_NaN();
        out.delta_mm = std::numeric_limits<float>::quiet_NaN();
        return out;
    }
    out.g_mm = z;
    out.delta_mm = static_cast<float>(delta);
    return out;
}

// fMinimal limiter proposal (ComputeTruePathLengthLimit mirror for the
// active C12 branch). t_input is the external TRUE candidate
// (min of user ceiling / collision / ... converted to true length).
// Diagnostic exit_path: 0=tiny, 1=range-vs-safety, 2=distance, 3=full.
struct UrbanV2LimitProposal {
    float t_msc_mm{0.0F};
    float g_msc_mm{0.0F};
    // Stable delta of the (t_msc, g_msc) conversion branch (fix B4).
    float delta_msc_mm{0.0F};
    float range_mm{0.0F};
    float lambda0_mm{0.0F};
    float tlimitmin_mm{0.0F};
    float tlimit_used_mm{0.0F};
    float par1{-1.0F};
    float par2{0.0F};
    float par3{0.0F};
    bool range_power_branch{false};
    bool limited_by_msc{false};
    bool no_displacement{false};
    bool valid{false};
    // OR of UrbanV2TrueToGeom.rounding_fixup over the limiter's internal
    // conversions (fix C1 counting).
    bool conv_rounding_fixup{false};
    int exit_path{-1};
};

// One active C12 loss context supplies R(E), E(R), and restricted DEDX.
// The old E/7.2 and E/64.5 values came from an unbound G4VMscModel fallback
// and are not production data. All Urban range consumers use this same table.
inline float urban_v2_current_range_mm(const UrbanV2Material& mat,
                                       const float epre_mev,
                                       bool& supported) noexcept {
    supported = false;
    const auto& c=mat.loss_context;
    if(!c.valid())return 0;
    const float range=c.range(epre_mev);
    supported=range>0 && sycl::isfinite(range);
    return supported?range:0;
}

inline UrbanV2LimitProposal copper_urban_v2_limit_step_with_range(
    const float t_input_true_mm, const float epre_mev,
    const UrbanV2Material& mat, const float safety_start_mm,
    const bool at_geom_boundary, UrbanV2TrackState& state,
    const float pre_step_range_mm,
    const std::uint64_t seed, const std::uint64_t history_id,
    const std::uint64_t outer_step,
    const std::uint32_t segment_index) noexcept {
    UrbanV2LimitProposal out{};
    // Fix C2: the MSC-domain RNG proof requires outer_step < 2^22 and
    // segment_index < 1024, so step_index = outer*1024+segment < 2^32.
    // Rejected here, BEFORE any draw (limiter dims 58/59); the transport
    // fatal path fires on !valid.
    if (!(segment_index < 1024U) || !(outer_step < (1ULL << 22))) {
        return out;
    }
    const std::uint64_t step_index = outer_step * 1024U + segment_index;
    if (!(t_input_true_mm > 0.0F) || !(epre_mev > 0.0F)) {
        return out;
    }
    bool range_supported = false;
    const float table_range = urban_v2_current_range_mm(
        mat, epre_mev, range_supported);
    const float range = pre_step_range_mm;
    const float lambda0 = urban_v2_transport_mfp(mat, epre_mev);
    const float range_tolerance = 4.0F * urban_ulp_above(table_range);
    if (!range_supported || !(range > 0.0F) || !sycl::isfinite(range) ||
        sycl::fabs(range - table_range) > range_tolerance ||
        !(lambda0 > 0.0F)) {
        return out;  // unsupported material: fail fast, no fallback.
    }
    out.range_mm = range;
    out.lambda0_mm = lambda0;
    float t = sycl::fmin(t_input_true_mm, range);

    // Extreme small step: no displacement, convert, firstStep NOT cleared
    // (early return skips it in the reference).
    if (t < kUrbanV2TlimitMinFix) {
        const auto conv = copper_urban_v2_true_to_geom(
            t, lambda0, range, epre_mev, mat);
        out.t_msc_mm = t;
        out.g_msc_mm = conv.g_mm;
        out.delta_msc_mm = conv.delta_mm;
        out.range_power_branch = conv.range_power_branch;
        out.conv_rounding_fixup =
            out.conv_rounding_fixup || conv.rounding_fixup;
        out.no_displacement = true;
        out.valid = true;
        out.exit_path = 0;
        out.tlimitmin_mm = state.tlimitmin_mm;
        return out;
    }

    // presafety: full isotropic safety, NO cap at t (the t==range branch
    // below is reachable only if safety can exceed t; a capped presafety
    // would make it dead code, which it is not in the reference).
    const float presafety = safety_start_mm;

    // Stop here if small step or range is less than safety.
    if ((t == range && t < presafety) || t < kUrbanV2TlimitMinFix) {
        const auto conv = copper_urban_v2_true_to_geom(
            t, lambda0, range, epre_mev, mat);
        out.t_msc_mm = t;
        out.g_msc_mm = conv.g_mm;
        out.delta_msc_mm = conv.delta_mm;
        out.range_power_branch = conv.range_power_branch;
        out.conv_rounding_fixup =
            out.conv_rounding_fixup || conv.rounding_fixup;
        out.no_displacement = true;
        out.valid = true;
        out.exit_path = 1;
        out.tlimitmin_mm = state.tlimitmin_mm;
        return out;
    }

    // Far from geometry boundary: no MSC limitation needed.
    const auto cache = urban_v2_material_cache(mat.zeff);
    const float distance = range * cache.doverrb;
    if (distance < presafety) {
        const auto conv = copper_urban_v2_true_to_geom(
            t, lambda0, range, epre_mev, mat);
        out.t_msc_mm = t;
        out.g_msc_mm = conv.g_mm;
        out.delta_msc_mm = conv.delta_mm;
        out.range_power_branch = conv.range_power_branch;
        out.conv_rounding_fixup =
            out.conv_rounding_fixup || conv.rounding_fixup;
        out.no_displacement = true;
        out.valid = true;
        out.exit_path = 2;
        out.tlimitmin_mm = state.tlimitmin_mm;
        return out;
    }

    // fMinimal: persistent tlimit, recomputed ONLY at a geometry boundary
    // as facrange*max(range, lambda0), floored by the FROZEN tlimitmin.
    // (The earlier doc note claiming min() was wrong; the reference takes
    // max().) No ComputeStepmin/ComputeTlimitmin call exists on this branch
    // in the reference, so none is made here (fix B3).
    const float tlimitmin = state.tlimitmin_mm;
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
            rng::msc_unit_strict(seed, history_id, step_index, 58U), 1.0e-12F);
        const float u1 = rng::msc_unit_strict(seed, history_id, step_index, 59U);
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
        t, lambda0, range, epre_mev, mat);
    out.t_msc_mm = t;
    out.g_msc_mm = conv.g_mm;
        out.delta_msc_mm = conv.delta_mm;
        out.conv_rounding_fixup =
            out.conv_rounding_fixup || conv.rounding_fixup;
    out.par1 = conv.par1;
    out.par2 = conv.par2;
    out.par3 = conv.par3;
    out.range_power_branch = conv.range_power_branch;
    out.valid = true;
    out.exit_path = 3;
    state.first_step = false;
    return out;
}

inline UrbanV2LimitProposal copper_urban_v2_limit_step(
    const float t_input_true_mm, const float epre_mev,
    const UrbanV2Material& mat, const float safety_start_mm,
    const bool at_geom_boundary, UrbanV2TrackState& state,
    const std::uint64_t seed, const std::uint64_t history_id,
    const std::uint64_t outer_step,
    const std::uint32_t segment_index) noexcept {
    bool range_supported = false;
    const float range = urban_v2_current_range_mm(
        mat, epre_mev, range_supported);
    if (!range_supported) return {};
    return copper_urban_v2_limit_step_with_range(
        t_input_true_mm, epre_mev, mat, safety_start_mm, at_geom_boundary,
        state, range, seed, history_id, outer_step, segment_index);
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
// Production SampleScattering uses the validated RNG dimensions and order.
// Debug export of the Urban sampling weights/branches (fix C3). Production
// passes nullptr (no behavior change); validation reads the taken branch
// and the decision-relevant floats to audit them against an independent
// long-double recomputation and threshold-adjacent uniforms.
// branch: 0 identity (t>=range / degenerate inputs), 1 early identity
// (tiny t / kin), 2 TauBig isotropic, 3 SimpleScattering (relloss),
// 4 SimpleScattering (theta0), 5 SimpleScattering (x_mean gate),
// 6 mixture/isotropic trial.
struct UrbanV2SampleDebug {
    float x_mean1{0.0F};
    float x_mean_th{0.0F};
    float probability{0.0F};
    float q_probability{0.0F};
    float d_param{0.0F};
    float theta0{0.0F};
    float xsi{0.0F};
    float tau_used{0.0F};
    int branch{0};
};

CorrelatedScatteringStep copper_urban_v2_sample_full(
    const Direction3F direction, const float epre_mev,
    const float t_final_mm, const float delta_final_mm,
    const UrbanV2Material& mat, const float range_mm,
    const float lambda0_mm, const float tlimitmin_mm,
    const float safety_end_mm, const float amplitude_scale,
    const bool allow_displacement,
    const std::uint64_t seed, const std::uint64_t history_id,
    const std::uint64_t step_index,
    const std::uint32_t random_dimension,
    UrbanV2SampleDebug* debug = nullptr) noexcept {
    CorrelatedScatteringStep result{direction, Direction3F{0.0F, 0.0F, 0.0F}};
    if (epre_mev <= 0.0F || mat.projectile_z <= 0 || mat.projectile_a <= 0 ||
        t_final_mm <= 0.0F || amplitude_scale <= 0.0F) {
        return result;
    }
    constexpr float tausmall = 1.0e-16F;
    constexpr float tlimitminfix = 1.0e-8F;
    constexpr float pi = 3.14159265358979323846F;
    constexpr float two_pi = 6.2831853071795864769F;
    const float mass_mev = mat.mass_mev;
    const auto radiation_length_mm = mat.radlen_mm;
    if (!(radiation_length_mm > 0.0F && lambda0_mm > 0.0F &&
          range_mm > 0.0F)) {
        return result;
    }
    const auto coefficients = copper_urban_coefficients(mat.zeff);

    // SampleScattering: no scattering at/over range.
    if (t_final_mm >= range_mm) {
        return result;
    }
    // Internal energy prediction (:784-790). Restricted DEDX from the loss
    // table (GetDEDX with the production couple), NOT the caller's
    // condensed-loss stopping and NOT post-straggling Epost (ordering kept).
    float kin_energy = epre_mev;
    if (t_final_mm > range_mm * kUrbanV2Dtrl) {
        kin_energy = mat.loss_context.energy(range_mm - t_final_mm);
    } else if (t_final_mm > range_mm * 0.01F) {
        kin_energy -= t_final_mm * mat.loss_context.dedx(epre_mev);
    }
    result.predicted_internal_energy_mev = kin_energy;
    if (!sycl::isfinite(kin_energy) || kin_energy < 0.0F ||
        kin_energy > epre_mev) {
        result.energy_prediction_valid = false;
        return result;
    }
    if ((t_final_mm <= tlimitminfix) || (t_final_mm < tausmall * lambda0_mm) ||
        (kin_energy <= 1.0e-6F)) {
        return result;
    }

    // Effective tau with lambda1 (:831-839).
    float tau = t_final_mm / lambda0_mm;
    float lambdaeff = lambda0_mm;
    if (kin_energy != epre_mev) {
        const float lambda1 = urban_v2_transport_mfp(mat, kin_energy);
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
        // (B2 verdict: float trial; reachable-domain audit shows zero
        // double-vs-float flips. Strict draws only.)
        const auto a = (2.0F * x_mean_th + 9.0F * x2_mean_th - 3.0F) /
                       (2.0F * x_mean_th - 3.0F * x2_mean_th + 1.0F);
        const auto probability = (a + 2.0F) * x_mean_th / a;
        const auto u0 = sycl::fmax(
            rng::msc_unit_strict(seed, history_id, step_index, dimension), 1.0e-12F);
        const auto u1 = rng::msc_unit_strict(
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
        if (debug != nullptr) {
            debug->branch = 2;
            debug->tau_used = tau;
        }
        const auto u = rng::msc_unit_strict(
            seed, history_id, step_index, random_dimension);
        cth = -1.0F + 2.0F * u;
        omcth = 2.0F * (1.0F - u);
    } else if (tau >= tausmall) {
        // Too-large relative loss falls back to isotropic-ish sampling.
        const float relloss = 1.0F - kin_energy / epre_mev;
        if (relloss > 0.50F) {
            if (debug != nullptr) {
                debug->branch = 3;
                debug->tau_used = tau;
            }
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
            float theta0 = 13.6F * static_cast<float>(mat.projectile_z) *
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
                    if (debug != nullptr) {
                        debug->branch = 4;
                        debug->tau_used = tau;
                        debug->theta0 = theta0;
                    }
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
                    // Fix B2 + audit C3 (evidence urban_after_0f2c0ca_20260922,
                    // localize_C3.txt): float trials are exact GIVEN the
                    // weights (threshold-adjacent strict draws verified by
                    // search); weights match an independent long-double
                    // recomputation to ~1e-7 (d to ~6e-6); isotropic tail
                    // weight error < 1e-7; residual float-vs-double decision
                    // differences live in the 2^-24 threshold fuzzy zone
                    // (documented, rate-consistent). No "bit-identical on
                    // the whole domain" claim is made. The ~4e-4 FP32 drift
                    // some analyses flag exists only at (t=0.05, E~1 MeV),
                    // which cannot sample (t<range implies
                    // tau<=range/lambda0<3e-3 there), so no double chain is
                    // carried: unconditional double cost 3-4x wall time on
                    // sm_75 for zero reachable effect.
                    // Kept exact and cheap: strict draws (B1), float 0.999
                    // gate, omcth-based |cth| protection below, dispR-capped
                    // postSafety in the acceptance helper.
                    const auto x_mean1 =
                        1.0F - (1.0F - (1.0F + xsi) * ea) * x / eaa;
                    const auto x0 = 1.0F - xsi * x;
                    if (x_mean1 <= 0.999F * x_mean_th) {
                        if (debug != nullptr) {
                            debug->branch = 5;
                            debug->tau_used = tau;
                            debug->theta0 = theta0;
                            debug->xsi = xsi;
                            debug->x_mean1 = x_mean1;
                            debug->x_mean_th = x_mean_th;
                        }
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
                        if (debug != nullptr) {
                            debug->branch = 6;
                            debug->tau_used = tau;
                            debug->theta0 = theta0;
                            debug->xsi = xsi;
                            debug->x_mean1 = x_mean1;
                            debug->x_mean_th = x_mean_th;
                            debug->probability = probability;
                            debug->q_probability = q_probability;
                            debug->d_param = d;
                        }
                        const auto u0 = rng::msc_unit_strict(
                            seed, history_id, step_index, random_dimension);
                        const auto u1 = rng::msc_unit_strict(
                            seed, history_id, step_index, random_dimension + 1U);
                        if (u0 < q_probability) {
                            if (u1 < probability) {
                                const auto u2 = rng::msc_unit_strict(
                                    seed, history_id, step_index,
                                    random_dimension + 2U);
                                const auto log_arg = ea + u2 * eaa;
                                const auto log_val = sycl::log(log_arg);
                                cth = 1.0F + log_val * x;
                                omcth = -log_val * x;
                            } else {
                                auto var = (1.0F - d) * rng::msc_unit_strict(
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
    // B2: |cth| >= 1 on the stable pre-rounding measure returns unscattered
    // (reference SampleScattering protection). A bare float clamp would
    // either kill most small-angle updates (if applied to rounded cth==1)
    // or rotate by a clamped angle the reference never proposes.
    const auto bad_cth = !(omcth > 0.0F) || !(omcth < 2.0F);
    if (bad_cth) {
        result.one_minus_cth = 0.0F;
        result.cth_rounded_to_one = false;
        return result;
    }
    cth = sycl::clamp(cth, -1.0F, 1.0F);
    result.one_minus_cth = omcth;
    result.cth_rounded_to_one = (cth == 1.0F);

    const auto sth = sycl::sqrt(
        sycl::fmax(0.0F, omcth * (2.0F - omcth)));
    const auto phi = two_pi * rng::msc_unit_strict(
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
    // (bad_cth defined above; rejected angles already returned.)
    const auto rmax2 = delta_final_mm * (2.0F * t_final_mm - delta_final_mm);
    if (!bad_cth && allow_displacement && current_tau >= tausmall &&
        rmax2 > 0.0F) {
        constexpr float cbeta = 2.160F;
        const auto cbeta1 = 1.0F - sycl::exp(-cbeta * pi);
        const auto radius = 0.73F * sycl::sqrt(rmax2);
        const auto u0 = rng::msc_unit_strict(
            seed, history_id, step_index, random_dimension + 4U);
        const auto u1 = rng::msc_unit_strict(
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

// ComputeTrueStepLength mirror for a geometry-truncated step: invert the
// FINAL geometric length with the limiter's par set, clamped to
// [g_final, t_msc]. Untruncated steps return t_msc untouched (the
// geomStepLength == zPathLength shortcut) with the proposal delta.
// delta_out_mm is the stable (t-g) OF THE TAKEN BRANCH (fix B4/C1).
// Fix C1: t is CONSTRUCTED from the stable delta (t = g + delta), never
// from an unstable log that a t<g pseudo-clamp then deletes. The old
// t = -lambda*log(1-g/lambda) rounds to 0 for tiny g/lambda, and the old
// `if (t<g){t=g;delta=0}` discarded the recovered small quantity. The old
// `t>t_msc -> reuse delta_msc` reused a delta that belongs to g_prop, not
// to g_final; the clamp now uses delta = t_msc - g_final, which belongs to
// the returned pair BY DEFINITION. guard_tripped flags any clamp repair
// (callers count it; reachable-domain tests assert zero trips).
inline float copper_urban_v2_finalize_true(
    const float g_final_mm, const float g_prop_mm, const float t_msc_mm,
    const float delta_msc_mm, const float lambda0_mm, const float range_mm,
    const float par1, const float par3, float& delta_out_mm,
    bool& guard_tripped,
    const bool range_power_branch = false) noexcept {
    guard_tripped = false;
    if (!(g_final_mm >= g_prop_mm)) {
        if (g_final_mm < 1.0e-6F) {
            delta_out_mm = 0.0F;
            return g_final_mm;
        }
        if (g_final_mm > lambda0_mm * 1.0e-16F) {
            float t;
            double delta;
            const double gd = static_cast<double>(g_final_mm);
            const double lamd = static_cast<double>(lambda0_mm);
            if (par1 < 0.0F) {
                // Stable par1<0 inversion: same math as the tested
                // true_path_and_delta helper. Series below ratio 1e-2
                // (no transcendental, no cancellation); float log1p +
                // definitional subtraction above (delta is sizable there).
                // t is then CONSTRUCTED as g + delta, so the t<g
                // pseudo-clamp can never fire on algorithm error.
                const double ratio = gd / lamd;
                if (ratio < 1.0e-2) {
                    const double series = 0.5 +
                        ratio * (1.0 / 3.0 + ratio * (0.25 + ratio * 0.2));
                    delta = gd * ratio * series;
                } else {
                    const float t_f = -lambda0_mm *
                        sycl::log1p(-(g_final_mm / lambda0_mm));
                    delta = static_cast<double>(t_f) - gd;
                }
                t = static_cast<float>(gd + delta);
            } else {
                // Geometry-truncated inverse of the same power-law branch.
                // For C12 (E < rest mass), p1=1/R and p3=1+R/lambda;
                // retain R/lambda in double when FP32 would round p3 to 1.
                const double p1d = static_cast<double>(par1);
                const double p3d = range_power_branch
                    ? 1.0 + static_cast<double>(range_mm) / lamd
                    : static_cast<double>(par3);
                const double inverse_p1 = range_power_branch
                    ? 1.0 / static_cast<double>(range_mm)
                    : p1d;
                delta = urban_v2_inverse_power_delta_mm(
                    inverse_p1, p3d, gd,
                    static_cast<double>(range_mm),
                    range_power_branch
                        ? static_cast<double>(range_mm) / lamd
                        : 0.0);
                if (!sycl::isfinite(delta) || delta < 0.0) {
                    guard_tripped = true;
                    delta_out_mm = std::numeric_limits<float>::quiet_NaN();
                    return std::numeric_limits<float>::quiet_NaN();
                } else {
                    t = static_cast<float>(gd + delta);
                }
            }
            if (t < g_final_mm) {
                // Unreachable with the t=g+delta construction above except
                // by float narrowing; kept as a counted guard, not a silent
                // repair.
                t = g_final_mm;
                delta = 0.0;
                guard_tripped = true;
            }
            if (t > t_msc_mm) {
                // Inversion overshoot. Within a few ulp of t_msc this is
                // normal float rounding of the par>=0 formula on marginally
                // truncated steps (end-of-range): clamp to t_msc with the
                // delta that belongs to the RETURNED pair (t_msc - g_final).
                // A LARGER overshoot means branch inconsistency and is
                // counted as a numeric fault.
                const float over = t - t_msc_mm;
                const float ulp_msc = urban_ulp_above(t_msc_mm);
                t = t_msc_mm;
                delta = static_cast<double>(t_msc_mm) - gd;
                if (!(over <= 4.0F * ulp_msc)) guard_tripped = true;
            }
            if (!(delta > 0.0)) delta = 0.0;
            delta_out_mm = static_cast<float>(delta);
            return t;
        }
    } else {
        delta_out_mm = delta_msc_mm;
        return t_msc_mm;
    }
    delta_out_mm = 0.0F;
    return g_final_mm;
}

// Full per-step GPIL+DoIt orchestration (Phase 5):
//   external TRUE candidate (user ceiling) -> fMinimal limit -> true->geom
//   -> geometry truncation -> geom->final-true -> scatter.
// Binding (limit_reason): 1=user/external, 2=msc, 3=geometry, 4=range (no
// scatter sampled). NOTE: no claim is made that evaluating geometry
// reachability before the limiter equals the reference order in general —
// true/geometric conversion, re-intersection and state update are not scalar
// min(). The loop shape is valid HERE because the caller subdivides to a
// fixed ceiling and the limiter is non-binding inside it (tlimit >> ceiling
// except at range end, where the DoIt gate takes over); any change to the
// step-competition structure must re-audit this ordering (prompt §2.6).
// Generic GPIL+DoIt orchestration shared by Copper and Water (Phase 5).
// Binding: 1=user/external, 2=msc, 3=geometry, 4=range (no scatter).
CorrelatedScatteringStep urban_v2_propose_and_sample(
    const Direction3F direction, const float epre_mev,
    const float user_ceiling_mm, const float boundary_path_mm,
    const float pos_x_mm, const float pos_y_mm, const float pos_z_mm,
    const float dir_x, const float dir_y, const float dir_z,
    const UrbanV2SafetyCtx& safety, const bool at_boundary,
    UrbanV2TrackState& state, const UrbanV2Material& mat,
    const float amplitude_scale, const std::uint64_t seed,
    const std::uint64_t history_id, const std::uint64_t outer_step,
    const std::uint32_t segment_index,
    const std::uint32_t random_dimension) noexcept {
    CorrelatedScatteringStep result{direction, Direction3F{0.0F, 0.0F, 0.0F}};
    // Fix C2: same address-proof guard as the limiter (see above); invalid
    // proposals carry proposal_valid=false with NO draws consumed.
    if (!(segment_index < 1024U) || !(outer_step < (1ULL << 22))) {
        result.limit_reason = -101; return result;
    }
    const std::uint64_t step_index = outer_step * 1024U + segment_index;
    if (!(epre_mev > 0.0F) || !(user_ceiling_mm > 0.0F) ||
        !(boundary_path_mm > 0.0F) ||
        !mat.loss_context.valid()) {
        result.limit_reason = -102; return result;
    }
    const float lambda0_now = urban_v2_transport_mfp(mat, epre_mev);
    if (!(lambda0_now > 0.0F)) {
        result.limit_reason = -103; return result;
    }
    // Step-start safety for the limiter presafety (full isotropic, uncapped).
    const float safety_start =
        urban_v2_safety_at(pos_x_mm, pos_y_mm, pos_z_mm, safety);
    // External TRUE candidate from the user ceiling (par1<0 inversion;
    // ceiling 0.25 << lambda0 ~ 2e6 so the small-ratio branch is exact).
    bool range_supported = false;
    const float pre_step_range =
        urban_v2_current_range_mm(mat, epre_mev, range_supported);
    if (!range_supported) { result.limit_reason = -104; return result; }
    const float t_ext = user_ceiling_mm;
    auto limit = copper_urban_v2_limit_step_with_range(
        t_ext, epre_mev, mat, safety_start, at_boundary, state,
        pre_step_range, seed, history_id, outer_step, segment_index);
    if (!limit.valid) {
        result.limit_reason = -105; return result;
    }
    const float g_prop = limit.g_msc_mm;
    const float g_final =
        g_prop < boundary_path_mm ? g_prop : boundary_path_mm;
    const bool truncated = (g_final < g_prop);
    // Fix B4: (t, g, delta) come from ONE branch. finalize returns the
    // branch-consistent delta; the old par1<0-only re-inversion of g_final
    // (wrong on the range branch at Bragg end) is deleted. The old
    // `t_final > t_ext` clamp is deleted too: t_final <= t_msc <= t_ext by
    // construction (limiter takes min with the t_ext-derived input), so it
    // was dead code with a wrong reference citation.
    float delta_final = 0.0F;
    bool finalize_guard = false;
    float t_final = copper_urban_v2_finalize_true(
        g_final, g_prop, limit.t_msc_mm, limit.delta_msc_mm,
        limit.lambda0_mm, limit.range_mm, limit.par1, limit.par3,
        delta_final, finalize_guard, limit.range_power_branch);

    result.proposed_true_path_mm = limit.t_msc_mm;
    result.final_true_path_mm = t_final;
    result.final_geom_path_mm = g_final;
    result.stable_delta_mm = static_cast<float>(
        delta_final > 0.0 ? delta_final : 0.0);
    result.boundary_crossed = truncated;
    result.proposal_valid = true;
    result.current_range_mm = limit.range_mm;
    result.pre_step_energy_mev = epre_mev;
    result.range_terminal =
        t_ext >= limit.range_mm && t_final >= limit.range_mm;
    // Fix C2.3 counting: ONLY finalize-class trips (t<g / t>t_msc clamp
    // repairs, which indicate real inconsistency) set the numeric-guard
    // flag. The limiter's sub-ulp chord repair (z snapped to t with delta
    // kept) is normal reference-class protection firing on ~7% of
    // production segments by FP32 granularity; it is proven bounded
    // (g off by <= 1 ulp, delta exact) and accuracy-asserted per grid
    // point in C1, so it is intentionally NOT counted as a fault.
    result.numeric_guard_tripped = finalize_guard;

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
    // Navigation ended on a physical face: its post-safety is zero even
    // if FP32 endpoint arithmetic leaves a sub-ulp gap to the same face.
    const float safety_end = urban_reaches_voxel_face(
        g_final, safety.box_boundary_path_mm) ? 0.0F :
        urban_v2_safety_at(end_x, end_y, end_z, safety);
    result = copper_urban_v2_sample_full(
        direction, epre_mev, t_final,
        static_cast<float>(delta_final > 0.0 ? delta_final : 0.0), mat,
        limit.range_mm, limit.lambda0_mm, limit.tlimitmin_mm, safety_end,
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
    result.pre_step_energy_mev = epre_mev;
    result.range_terminal =
        t_ext >= limit.range_mm && t_final >= limit.range_mm;
    // Fix C2.3 counting: ONLY finalize-class trips (t<g / t>t_msc clamp
    // repairs, which indicate real inconsistency) set the numeric-guard
    // flag. The limiter's sub-ulp chord repair (z snapped to t with delta
    // kept) is normal reference-class protection firing on ~7% of
    // production segments by FP32 granularity; it is proven bounded
    // (g off by <= 1 ulp, delta exact) and accuracy-asserted per grid
    // point in C1, so it is intentionally NOT counted as a fault.
    result.numeric_guard_tripped = finalize_guard;
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

// Copper wrapper: same signature as the c53f151 production call; builds the
// Cu material (frozen elemental helper, Zeff=29) and kind-0 safety ctx, then
// delegates to the shared generic core above. No copper physics lives here,
// so the Cu path cannot drift from the Water path's core.
inline CorrelatedScatteringStep copper_urban_v2_propose_and_sample(
    const Direction3F direction, const float epre_mev,
    const int atomic_number, const int mass_number,
    const float user_ceiling_mm, const float boundary_path_mm,
    const float pos_x_mm, const float pos_y_mm, const float pos_z_mm,
    const float dir_x, const float dir_y, const float dir_z,
    const UrbanV2GeomCtx& geom, const bool at_boundary,
    UrbanV2TrackState& state, const UrbanMcsContext& loss_context,
    const float density_g_per_cm3, const float radiation_length_g_per_cm2,
    const float amplitude_scale, const std::uint64_t seed,
    const std::uint64_t history_id, const std::uint64_t outer_step,
    const std::uint32_t segment_index,
    const std::uint32_t random_dimension,
    const float rest_mass_MeV) noexcept {
    UrbanV2Material mat{};
    mat.loss_context = loss_context;
    mat.zeff = kUrbanV2CopperZeff;
    mat.radlen_mm = 10.0F * radiation_length_g_per_cm2 / density_g_per_cm3;
    mat.projectile_z = atomic_number;
    mat.projectile_a = mass_number;
    mat.mass_mev = rest_mass_MeV;
    mat.density_g_per_cm3 = density_g_per_cm3;
    UrbanV2SafetyCtx safety{};
    safety.kind = 0;
    safety.cu = geom;
    return urban_v2_propose_and_sample(
        direction, epre_mev, user_ceiling_mm, boundary_path_mm, pos_x_mm,
        pos_y_mm, pos_z_mm, dir_x, dir_y, dir_z, safety, at_boundary, state,
        mat, amplitude_scale, seed, history_id, outer_step, segment_index,
        random_dimension);
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
    const float energy_per_u) noexcept {
    // Pure-EM G4_WATER slab fits at 50/150/300 MeV/u.  Each energy used
    // independent 1 and 10 mm, 250k-history TOPAS references.  Keep C12 and
    // uncalibrated ions on the previously validated C12-water candidate.
    if (atomic_number == 1 && mass_number == 1) {
        return WaterIonFermiEygesParameters{
            interpolate_water_fe_parameter(
                energy_per_u, 12.1250440F, 11.8620445F, 11.7235806F),
            interpolate_water_fe_parameter(
                energy_per_u, 0.002426742F, 0.002246132F, 0.001998740F),
            interpolate_water_fe_parameter(
                energy_per_u, 6.7809249F, 7.8095673F, 8.8033596F)};
    }
    if (atomic_number == 1 && mass_number == 2) {
        return WaterIonFermiEygesParameters{
            interpolate_water_fe_parameter(
                energy_per_u, 10.7251611F, 10.5823527F, 10.4433029F),
            interpolate_water_fe_parameter(
                energy_per_u, 0.003069940F, 0.003104010F, 0.003273779F),
            interpolate_water_fe_parameter(
                energy_per_u, 3.2194970F, 3.0651926F, 2.7842157F)};
    }
    if (atomic_number == 1 && mass_number == 3) {
        return WaterIonFermiEygesParameters{
            interpolate_water_fe_parameter(
                energy_per_u, 10.6735838F, 10.4564336F, 10.3058709F),
            interpolate_water_fe_parameter(
                energy_per_u, 0.003033585F, 0.003151119F, 0.003381651F),
            interpolate_water_fe_parameter(
                energy_per_u, 3.2822138F, 2.9271832F, 2.6076538F)};
    }
    if (atomic_number == 2 && mass_number == 4) {
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
    const float mass_per_u = static_cast<float>(nucleon_rest_mass_MeV),
    const bool gaussian_only = false) noexcept {
    CorrelatedScatteringStep result{direction, Direction3F{0.0F, 0.0F, 0.0F}};
    if (kinetic_energy_MeV <= 0.0F || atomic_number <= 0 || mass_number <= 0 ||
        path_length_mm <= 0.0F ||
        density_g_per_cm3 <= 0.0F || radiation_length_g_per_cm2 <= 0.0F) {
        return result;
    }

    const auto energy_per_u = kinetic_energy_MeV / static_cast<float>(mass_number);
    const auto total_energy_per_u =
        energy_per_u + mass_per_u;
    const auto momentum_per_u = sycl::sqrt(
        energy_per_u *
        (energy_per_u + 2.0F * mass_per_u));
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
        atomic_number, mass_number, energy_per_u);
    constexpr float water_reference_radiation_length_mm = 360.8297746F;
    const auto tail_material_scale =
        water_reference_radiation_length_mm / radiation_length_mm;
    const auto tail_rate_per_mm =
        parameters.tail_rate_per_water_mm * tail_material_scale;
    const auto charge_over_beta_momentum =
        static_cast<float>(atomic_number) / (beta * momentum);
    const auto core_ratio =
        parameters.core_scattering_energy_MeV * charge_over_beta_momentum;
    // Match the full local second moment, including Poisson tail variance.
    // FE covariance per transverse axis: A0=T*L, A1=T*L^2/2, A2=T*L^3/3.
    const auto tail_sigma_for_moments =
        parameters.tail_scattering_energy_MeV * charge_over_beta_momentum;
    const auto core_power = core_ratio * core_ratio / radiation_length_mm +
        (gaussian_only ? tail_rate_per_mm * tail_sigma_for_moments * tail_sigma_for_moments : 0.0F);
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
    while (!gaussian_only) {
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
    const float mass_per_u = static_cast<float>(nucleon_rest_mass_MeV),
    const bool gaussian_only = false) noexcept {
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
            mass_per_u, gaussian_only);
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

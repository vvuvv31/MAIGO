#pragma once

#include <limits>

namespace carbon {
// Isolated spin-zero C12 / homogeneous-water research building blocks.
// No production transport path enables these helpers yet.
enum class DeltaDrawStatus { accepted, below_cut, form_factor_null, exhausted, invalid };

template <typename Real> struct DeltaDraw {
    Real energy_MeV{};
    DeltaDrawStatus status{DeltaDrawStatus::invalid};
};

// Geant4 11.3.2 Lindhard-Sorensen spin-zero proposal and form-factor veto.
// The macro rate must correspond to proposals, including form-factor nulls.
// Caller supplies independent uniforms in [0,1); exhaustion is a hard failure,
// never a physical null event. Process-rate integral acceptance is a separate test.
template <typename Real, typename Uniform>
DeltaDraw<Real> sample_spin_zero_delta_candidate(
    Real cut, Real tmax, Real beta2, Real form_factor, Uniform&& uniform) {
    constexpr Real finite_max = std::numeric_limits<Real>::max();
    if (!(cut > 0 && cut <= finite_max) || !(tmax >= 0 && tmax <= finite_max) ||
        !(beta2 >= 0 && beta2 < 1) ||
        !(form_factor >= 0 && form_factor <= finite_max)) return {};
    if (tmax <= cut) return {0, DeltaDrawStatus::below_cut};
    for (int trial = 0; trial < 128; ++trial) {
        const Real u = uniform(), v = uniform();
        if (!(u >= 0 && u < 1 && v >= 0 && v < 1)) return {};
        const Real transfer = cut * tmax / (cut * (1 - u) + tmax * u);
        if (v > 1 - beta2 * transfer / tmax) continue;
        const Real x = form_factor * transfer;
        if (x > Real(1e-6)) {
            const Real w = uniform();
            if (!(w >= 0 && w < 1)) return {};
            if (w > 1 / ((1 + x) * (1 + x)))
                return {0, DeltaDrawStatus::form_factor_null};
        }
        return {transfer, DeltaDrawStatus::accepted};
    }
    return {0, DeltaDrawStatus::exhausted};
}

// Two-phase clock: inspect candidate distances, choose the winning process,
// then debit ALL clocks using the FINAL travelled distance. This avoids losing
// optical depth when a different process or a voxel face shortens a step.
template <typename Real> struct DiscreteDeltaClockCandidate {
    Real remaining{};
    bool active{false};
    // Caller validates finite, non-negative rates, distances and sampled tau.
    void arm(Real sampled_optical_depth) {
        remaining = sampled_optical_depth;
        active = true;
    }
    Real distance(Real proposal_rate) const {
        return active && proposal_rate > 0
            ? remaining / proposal_rate : std::numeric_limits<Real>::infinity();
    }
    void consume(Real final_path, Real proposal_rate, bool selected) {
        if (!active) return;
        const Real next = remaining - final_path * proposal_rate;
        remaining = next > 0 ? next : 0;
        // Reset after both accepted and rejected proposals of THIS process.
        if (selected) { active = false; remaining = 0; }
    }
};
} // namespace carbon

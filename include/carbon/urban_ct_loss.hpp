#pragma once
#include "carbon/urban_mcs_package.hpp"
#include "carbon/unified_em_view.hpp"
#include "carbon/beamline_em.hpp"

namespace carbon {
// The continuous26 point belongs to the caller's active transport step.
// Use unrestricted electronic stopping and its CSDA inverse in the medium.
// Cu/air use their full response; additional recoils use unrestricted electronic
// plus nuclear stopping from ELRSP001. Native Urban supplies scattering only.
struct UrbanLossContext : UrbanMcsContext {
    const UnifiedEmPoint* ct_point{};
    BeamlineEmPoint beamline_point{};

    UrbanLossContext() = default;
    UrbanLossContext(const UrbanMcsContext& context) : UrbanMcsContext(context) {}
    bool valid() const {
        return UrbanMcsContext::valid() && (!beamline_point.nodes || beamline_point.valid()) && (!ct_point ||
            (ct_point->record && ct_point->continuous && ct_point->unrestricted_ranges));
    }
    float range(float energy) const {
        if (beamline_point.nodes) return beamline_point.range(energy);
        if (!ct_point) return NAN;
        return ct_point->unrestricted_range(energy);
    }
    float dedx(float energy) const {
        if (beamline_point.nodes) return beamline_point.dedx(energy);
        if (!ct_point) return NAN;
        return ct_point->unrestricted_electronic_stopping(energy);
    }
    float energy(float range_mm) const {
        if (beamline_point.nodes) return beamline_point.energy(range_mm);
        if (!ct_point) return NAN;
        if (range_mm <= 0.f) return 0.f;
        return ct_point->unrestricted_energy(range_mm);
    }
};
} // namespace carbon

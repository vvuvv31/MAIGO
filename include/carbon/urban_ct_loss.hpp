#pragma once
#include "carbon/urban_mcs_package.hpp"
#include "carbon/unified_em_view.hpp"

namespace carbon {
// The continuous26 point belongs to the caller's active transport step.
// Keep its charge factor fixed for all Urban subqueries in that step, as
// G4VEnergyLossProcess does. Copper/air/water keep their native Urban context.
struct UrbanLossContext : UrbanMcsContext {
    const UnifiedEmPoint* ct_point{};
    float ct_factor{};

    UrbanLossContext() = default;
    UrbanLossContext(const UrbanMcsContext& context) : UrbanMcsContext(context) {}
    bool valid() const {
        return UrbanMcsContext::valid() && (!ct_point ||
            (ct_point->record && ct_point->continuous && ct_point->continuous_range &&
             ct_factor > 0.f && std::isfinite(ct_factor)));
    }
    float range(float energy) const {
        if (!ct_point) return UrbanMcsContext::range(energy);
        const float ratio = ct_point->record->ratio;
        return ct_point->corrected_raw_range(energy * ratio) / (ct_factor * ratio);
    }
    float dedx(float energy) const {
        if (!ct_point) return UrbanMcsContext::dedx(energy);
        return ct_factor * ct_point->corrected_raw_stopping(energy * ct_point->record->ratio);
    }
    float energy(float range_mm) const {
        if (!ct_point) return UrbanMcsContext::energy(range_mm);
        if (range_mm <= 0.f) return 0.f;
        const float ratio = ct_point->record->ratio;
        return ct_point->continuous_inverse(range_mm * ct_factor * ratio) / ratio;
    }
};
} // namespace carbon

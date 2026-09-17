#pragma once
#include <algorithm>
#include <cmath>
namespace carbon {
struct NuclearCollision {
    float distance_mm;
    bool occurred;
};
// Homogeneous Poisson process: P(S<h)=1-exp(-Sigma*h), S=-log(1-U)/Sigma.
// This helper contains no projectile/target model or empirical cross-section data.
inline NuclearCollision sample_exponential_collision(float rate_per_mm,
                                                     float horizon_mm,
                                                     float uniform) noexcept {
    NuclearCollision result{horizon_mm, false};
    if (!(rate_per_mm > 0.f && horizon_mm > 0.f)) return result;
    const float survival = std::exp(-rate_per_mm * horizon_mm);
    if (!(uniform < 1.f - survival)) return result;
    const float optical_depth = -std::log(std::max(1.f - uniform, 1.e-12f));
    const float lower = std::min(horizon_mm, 1.e-5f);
    result.distance_mm = std::clamp(optical_depth / rate_per_mm, lower, horizon_mm);
    result.occurred = true;
    return result;
}
} // namespace carbon

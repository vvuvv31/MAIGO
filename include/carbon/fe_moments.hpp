#pragma once
namespace carbon {
// Projected angular variance, displacement-angle covariance, displacement variance.
// All quantities are FP32; these are transport moments, not dose scorers.
struct FeMoments {
    float a0{}, a1{}, a2{};
    void add(float power, float length, float remaining_midpoint) noexcept {
        const float variance = power * length;
        a0 += variance;
        a1 += variance * remaining_midpoint;
        a2 += variance * (remaining_midpoint * remaining_midpoint + length * length / 12.0F);
    }
};
}

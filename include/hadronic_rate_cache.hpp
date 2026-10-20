#pragma once

namespace carbon {
// Integral hadronic hazard cache with post-EM rejection sampling.
// The caller invalidates the cache at CT material/density transitions.
struct HadronicRateCache {
    float energy{};
    float rate{};
    bool valid{false};

    float update(float current_energy, float current_rate) noexcept {
        if (!valid || current_energy * 1.25F < energy) {
            energy = current_energy;
            rate = current_rate;
            valid = true;
        }
        return rate;
    }
    bool accept(float post_em_rate, float uniform) noexcept {
        // Geant4 resets its energy cache after every proposed interaction,
        // including proposals rejected by the integral cross-section test.
        valid = false;
        return post_em_rate >= rate * uniform;
    }
};
} // namespace carbon

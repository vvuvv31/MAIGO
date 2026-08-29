#pragma once

#include <cmath>
#include <cstdint>
#include <array>

namespace carbon {

inline constexpr int kMaxInelasticProducts = 12;
inline constexpr int kMaxProjectileFragments = 8;

struct SecondaryParticle {
    int16_t z{0};
    int16_t a{0};
    float energy_MeV{0.0F};
    float pos_x_mm{0.0F};
    float pos_y_mm{0.0F};
    float pos_z_mm{0.0F};
    float dir_x{0.0F};
    float dir_y{0.0F};
    float dir_z{1.0F};
    float weight{1.0F};
    uint32_t parent_history{0};
};

struct InelasticProductSet {
    uint8_t count{0};
    float local_deposit_MeV{0.0F};
    float untracked_energy_MeV{0.0F};
    float model_residual_MeV{0.0F};
    float model_unassigned_MeV{0.0F};
    SecondaryParticle products[kMaxInelasticProducts]{};
    uint8_t retries_used{0};
    uint8_t energy_scaled{0};
    uint8_t resample_failed{0};
    float q_MeV{0.0F};
    float neutron_ke_MeV{0.0F};
    float remnant_local_MeV{0.0F};
    int8_t leftover_proj_a{0};
    int8_t leftover_proj_z{0};
    int8_t leftover_tgt_a{0};
    int8_t leftover_tgt_z{0};
    uint8_t n_emitted{0};
    uint8_t emitted_idx[kMaxInelasticProducts]{};
    uint8_t product_capacity_overflow{0};
    float product_capacity_overflow_MeV{0.0F};
};

// Leftover after charged + local + sampled untracked + explicit unassigned.
// Scaled/fallback events keep unassigned=0 and put KE on charged products.
inline float inelastic_numerical_residual_MeV(float incident_MeV, float charged_MeV,
                                              float local_MeV, float untracked_MeV,
                                              float unassigned_MeV) noexcept {
    return incident_MeV - charged_MeV - local_MeV - untracked_MeV - unassigned_MeV;
}

inline float inelastic_fail_unassigned_if_no_products(uint8_t product_count,
                                                      float incident_MeV) noexcept {
    return (product_count == 0) ? incident_MeV : 0.0F;
}


}  // namespace carbon

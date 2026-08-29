#pragma once

#include <cmath>
#include <cstdint>
#include <array>

namespace carbon {

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
};

struct InelasticProductSet {
    uint8_t count{0};
    float local_deposit_MeV{0.0F};
    SecondaryParticle products[6]{};
};

}  // namespace carbon

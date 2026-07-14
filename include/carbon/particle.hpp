#pragma once

namespace carbon {

struct Particle1D {
    double position_mm{0.0};
    double kinetic_energy_MeV{0.0};
    int particle_type{6012};
    bool alive{true};
};

}  // namespace carbon


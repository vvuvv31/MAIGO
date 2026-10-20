#include "transport.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace carbon {

double TransportResult::physical_relative_energy_balance_error() const noexcept {
    if (initial_energy_MeV == 0.0) {
        return 0.0;
    }
    return std::abs(initial_energy_MeV - total_deposited_energy_MeV -
                    escaped_energy_MeV - beamline_removed_energy_MeV -
                    untracked_nuclear_energy_MeV -
                    primary_other_terminal_kinetic_MeV) /
           initial_energy_MeV;
}

}  // namespace carbon

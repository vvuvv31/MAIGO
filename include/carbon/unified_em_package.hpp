#pragma once
#include "carbon/em_continuous_density.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace carbon {
// EMCNT001: 26 compositions and continuous density; all 18 charged ions.
struct UnifiedEmMaterial { std::int32_t section; float density; };
struct UnifiedEmSpecies { std::uint32_t z,a; };
// Runtime-only data. The fixed EMCNT001 disk layout is decoded privately on
// the host; retired restricted/rate/spline fields never enter these GPU views.
struct UnifiedEmRecord {
    float mass,ratio,excitation,e0,step_fraction,final_range,linear_limit,is_ion,lowest_kinetic;
    std::uint32_t node_offset,node_count,z,a;
};
struct UnifiedEmNode {
    float energy,full,factor,dispersion,universal_dispersion,fluctuation;
};
static_assert(sizeof(UnifiedEmMaterial)==8 && sizeof(UnifiedEmSpecies)==8);
static_assert(sizeof(UnifiedEmRecord)==52 && sizeof(UnifiedEmNode)==24);
class UnifiedEmPackage {
public:
    std::vector<UnifiedEmMaterial> materials;
    std::vector<UnifiedEmSpecies> species;
    std::vector<UnifiedEmRecord> records;
    std::vector<UnifiedEmNode> nodes;
    std::vector<EmContinuousDescriptor> continuous_descriptors;
    std::vector<float> continuous_scalars;
    unsigned continuous_scalar_count{};
    // Derived once from native unrestricted electronic stopping. Same node
    // offsets as nodes; units are mm at each composition's reference density.
    // This is runtime-owned data, not a change to the EMCNT001 file schema.
    std::vector<float> unrestricted_ranges;
    static UnifiedEmPackage load(const std::filesystem::path&);
};
} // namespace carbon

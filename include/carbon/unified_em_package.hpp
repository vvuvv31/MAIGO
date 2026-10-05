#pragma once
#include "carbon/primary_em_mean_candidate.hpp"
#include "carbon/em_continuous_density.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace carbon {
// EMJOINT1: one container, water + Schneider density nodes, all 18 charged ions.
struct UnifiedEmMaterial { std::int32_t section; float density; };
struct UnifiedEmSpecies { std::uint32_t z,a; };
struct UnifiedEmRecord {
    float mass,ratio,cut,excitation,e0,spin,step_fraction,final_range,linear_limit;
    float is_ion,form_factor,magnetic_moment2,lowest_kinetic,peak;
    std::uint32_t node_offset,node_count,offsets[4],counts[4],z,a;
};
struct UnifiedEmNode {
    float energy,full,restricted,stopping,range,lambda,factor,correction;
    float dispersion,universal_dispersion,q2,model,fluctuation;
};
static_assert(sizeof(UnifiedEmMaterial)==8 && sizeof(UnifiedEmSpecies)==8);
static_assert(sizeof(UnifiedEmRecord)==104 && sizeof(UnifiedEmNode)==52);
class UnifiedEmPackage {
public:
    std::vector<UnifiedEmMaterial> materials;
    std::vector<UnifiedEmSpecies> species;
    std::vector<UnifiedEmRecord> records;
    std::vector<UnifiedEmNode> nodes;
    std::vector<EmCubicSegmentCandidate<float>> segments;
    std::vector<EmContinuousDescriptor> continuous_descriptors;
    std::vector<float> continuous_scalars, continuous_ranges;
    unsigned continuous_degree{}, continuous_scalar_count{};
    std::vector<unsigned> continuous_lengths;
    // Immutable certificate computed once when the package is parsed.
    bool shared_prepare_grid{false};
    static UnifiedEmPackage load(const std::filesystem::path&);
};
bool unified_em_has_shared_prepare_grid(const UnifiedEmPackage&);
} // namespace carbon

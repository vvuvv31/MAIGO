#pragma once
#include <cstdint>
#include <filesystem>
#include <vector>
namespace carbon {
// Explicit portable layout; all fields are 32-bit, file is little endian.
struct NeutronMaterialRecord {std::int32_t section;float density,temperature,excitation;};
struct NeutronRateRecord {
 std::uint32_t material,node,subtype,event_offset,event_count;
 float energy,rate;
};
struct NeutronEventRecord {
 std::uint32_t rate,product_offset,product_count;
 std::int32_t target_z,target_a,status,model;
 float continuation,dx,dy,dz,deposit,kinetic_gain;
};
struct NeutronProductRecord {
 std::int32_t pdg,z,a;
 float mass,energy,dx,dy,dz,time;
};
// Independent macroscopic grid; no event offsets or final-state interpolation.
// interval_status describes the interval to the next node of this process:
// 0=end, 1=pointwise bound met, 2=zero, 3=adjacent FP32,
// 4=statistics limited, 5=needs refinement.
struct NeutronMacroscopicRecord {
 std::uint32_t material,subtype,queries,interval_status;
 float energy,rate,sem;
};
static_assert(sizeof(NeutronMacroscopicRecord)==28);
static_assert(sizeof(NeutronMaterialRecord)==16 && sizeof(NeutronRateRecord)==28);
static_assert(sizeof(NeutronEventRecord)==52 && sizeof(NeutronProductRecord)==36);
class NeutronPackage {
public:
 // Audit mode can inspect unvalidated native extraction packages. Runtime
 // loading rejects them. The explicitly experimental research backend uses
 // draft loading; this does not certify physical validity.
 static NeutronPackage load(const std::filesystem::path& manifest,bool require_runtime_validation=true);
 bool runtime_validated{};
 std::vector<NeutronMaterialRecord> materials;
 std::vector<NeutronRateRecord> rates;
 std::vector<NeutronEventRecord> events;
 std::vector<NeutronProductRecord> products;
 std::vector<NeutronMacroscopicRecord> macroscopic_rates;
};
}

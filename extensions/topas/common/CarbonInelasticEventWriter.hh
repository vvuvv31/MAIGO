#ifndef CarbonInelasticEventWriter_hh
#define CarbonInelasticEventWriter_hh

#include "G4Types.hh"
#include "G4ThreeVector.hh"

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

class G4HadronicProcess;
class G4ParticleChange;
class G4Step;
class G4Track;
class G4VProcess;

// This is copied from the const G4Track/G4Step immediately before the
// delegated process call.  It intentionally owns values, not Geant4 object
// pointers: the wrapped process is allowed to mutate its particle-change
// object and the post-call track must never be mistaken for collision input.
struct CarbonCinel02InputSnapshot {
    std::uint64_t run_id{0};
    std::uint32_t thread_id{0};
    std::uint64_t event_id{0};
    std::uint32_t track_id{0};
    std::uint32_t parent_track_id{0};
    std::int32_t projectile_pdg{0};
    std::int16_t projectile_z{0};
    std::int16_t projectile_a{0};
    float projectile_charge{0.0F};
    float projectile_rest_mass{0.0F};
    float projectile_excitation{0.0F};
    float collision_energy_MeV{0.0F};
    float collision_energy_MeV_per_u{0.0F};
    G4ThreeVector collision_position{};
    G4ThreeVector collision_direction{0.0, 0.0, 1.0};
    float collision_time_ns{0.0F};
    float proper_time_ns{0.0F};
    float track_weight{1.0F};
    float step_length_mm{0.0F};
    std::int32_t material_id{-1};
    std::int32_t cuts_couple_id{-1};
    float source_initial_energy_MeV{0.0F};
    float absolute_depth_mm{0.0F};
    float audit_pre_energy_MeV{0.0F};
    G4ThreeVector audit_pre_position{};
    G4ThreeVector audit_pre_direction{0.0, 0.0, 1.0};
    float audit_whole_step_deposit_MeV{0.0F};
    std::string material_name;
};

// The following packed records are the worker-local CINEL02 raw contract.
// They intentionally duplicate the public MAIGO loader layout: TOPAS builds
// extensions outside the carbon-core CMake target, so the extension must not
// depend on the MAIGO library just to write a raw event.
#pragma pack(push, 1)
struct CarbonCinel02Interaction {
    std::uint64_t run_id{0};
    std::uint32_t thread_id{0};
    std::uint64_t event_id{0};
    std::uint32_t track_id{0};
    std::uint32_t parent_track_id{0};
    std::uint32_t interaction_sequence{0};
    std::int32_t projectile_pdg{0};
    std::int16_t projectile_z{0};
    std::int16_t projectile_a{0};
    float projectile_charge{0.0F};
    float projectile_rest_mass{0.0F};
    float projectile_excitation{0.0F};
    float collision_energy_MeV{0.0F};
    float collision_energy_MeV_per_u{0.0F};
    float collision_x_mm{0.0F};
    float collision_y_mm{0.0F};
    float collision_z_mm{0.0F};
    float collision_direction_x{0.0F};
    float collision_direction_y{0.0F};
    float collision_direction_z{1.0F};
    float collision_time_ns{0.0F};
    float proper_time_ns{0.0F};
    float track_weight{1.0F};
    float step_length_mm{0.0F};
    std::int32_t material_id{0};
    std::int32_t cuts_couple_id{0};
    std::int16_t target_z{0};
    std::int16_t target_a{0};
    std::uint32_t target_isotope_id{0};
    std::int32_t process_type{0};
    std::int32_t process_subtype{0};
    std::int32_t model_id{-1};
    std::int32_t parent_status{0};
    std::int32_t parent_pdg{0};
    std::int16_t parent_z{0};
    std::int16_t parent_a{0};
    float parent_charge{0.0F};
    float parent_rest_mass{0.0F};
    float parent_excitation{0.0F};
    float parent_energy_MeV{0.0F};
    float parent_direction_x{0.0F};
    float parent_direction_y{0.0F};
    float parent_direction_z{1.0F};
    float parent_x_mm{0.0F};
    float parent_y_mm{0.0F};
    float parent_z_mm{0.0F};
    float parent_time_ns{0.0F};
    float parent_proper_time_ns{0.0F};
    float parent_weight{1.0F};
    float process_local_deposit_MeV{0.0F};
    float nonionizing_deposit_MeV{0.0F};
    std::uint32_t direct_product_count{0};
    std::uint32_t unsupported_product_count{0};
    float unsupported_product_energy_MeV{0.0F};
    float audit_pre_energy_MeV{0.0F};
    float audit_pre_direction_x{0.0F};
    float audit_pre_direction_y{0.0F};
    float audit_pre_direction_z{1.0F};
    float audit_pre_x_mm{0.0F};
    float audit_pre_y_mm{0.0F};
    float audit_pre_z_mm{0.0F};
    float audit_whole_step_deposit_MeV{0.0F};
    float source_initial_energy_MeV{0.0F};
    float absolute_depth_mm{0.0F};
    char material_name[64]{};
    char target_isotope_name[32]{};
    char process_name[64]{};
    char model_name[64]{};
};

struct CarbonCinel02Product {
    std::int32_t pdg{0};
    std::int16_t z{0};
    std::int16_t a{0};
    float charge{0.0F};
    float rest_mass{0.0F};
    float excitation{0.0F};
    float kinetic_energy_MeV{0.0F};
    float direction_x{0.0F};
    float direction_y{0.0F};
    float direction_z{1.0F};
    float local_direction_x{0.0F};
    float local_direction_y{0.0F};
    float local_direction_z{1.0F};
    float position_x_mm{0.0F};
    float position_y_mm{0.0F};
    float position_z_mm{0.0F};
    float creation_time_ns{0.0F};
    float weight{1.0F};
    std::int32_t role{0};
};

struct CarbonCinel02RawHeader {
    char magic[8]{};
    std::uint32_t version{2};
    std::uint32_t header_size{64};
    std::uint32_t endian_marker{0x01020304U};
    std::uint32_t flags{0x0FU};
    std::uint64_t interaction_count{0};
    std::uint64_t product_count{0};
    std::uint64_t bytes_written{0};
    std::uint64_t reserved0{0};
    std::uint64_t reserved1{0};
};

struct CarbonCinel02RecordPrefix {
    char magic[4]{};
    std::uint16_t version{2};
    std::uint16_t flags{0};
    std::uint32_t record_length{0};
    std::uint32_t payload_length{0};
};
#pragma pack(pop)

static_assert(sizeof(CarbonCinel02Interaction) == 476);
static_assert(sizeof(CarbonCinel02Product) == 72);
static_assert(sizeof(CarbonCinel02RawHeader) == 64);
static_assert(sizeof(CarbonCinel02RecordPrefix) == 16);

class CarbonInelasticEventWriter {
public:
    static CarbonInelasticEventWriter& Instance();

    CarbonInelasticEventWriter(const CarbonInelasticEventWriter&) = delete;
    CarbonInelasticEventWriter& operator=(const CarbonInelasticEventWriter&) = delete;

    static CarbonCinel02InputSnapshot CaptureInput(const G4Track&, const G4Step&);

    void Append(const CarbonCinel02InputSnapshot&, const G4ParticleChange&,
                const G4VProcess&, G4HadronicProcess*);

private:
    CarbonInelasticEventWriter();
    ~CarbonInelasticEventWriter();

    void OpenIfNeeded();
    void Finalize();
    std::uint32_t NextInteractionSequence(std::uint64_t event_id, std::uint32_t track_id);
    void WriteRecord(const CarbonCinel02Interaction&, const std::vector<CarbonCinel02Product>&);
    void WriteCampaignContract() const;
    void WriteWorkerContract() const;

    std::fstream stream_;
    std::string output_directory_;
    std::string output_file_;
    std::string campaign_uuid_;
    bool overwrite_campaign_{false};
    std::uint64_t interaction_count_{0};
    std::uint64_t product_count_{0};
    std::uint64_t target_isotope_pointer_missing_count_{0};
    std::uint64_t bytes_written_{sizeof(CarbonCinel02RawHeader)};
    std::uint64_t current_event_id_{~std::uint64_t{0}};
    std::map<std::uint32_t, std::uint32_t> interaction_sequences_;
    bool opened_{false};
};

#endif

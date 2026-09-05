#pragma once
#include "TsVNtupleScorer.hh"
#include <map>
#include <tuple>
#include <utility>

// CarbonElectronDepositNtuple record schema v2.
//
// v1 (frozen): 21 columns, world-coordinate step records, no parent conditioning.
// v2 (this file): 21 v1 columns + 8 new columns carrying the parent-C12 birth
//   conditioning needed for a transportable electron response:
//
//     schema_version      == 2 on every v2 row (old 21-column files must never
//                            be parsed as v2; the analyzer rejects column-count
//                            mismatches as unknown schema).
//     parent_valid        1 when a same-event C12 state was cached before this
//                         electron's first step, else 0 (null-equivalent; the
//                         parent_* fields below are sentinel -1/0 and must not
//                         be used for projection when 0).
//     parent_ke_MeV       cached C12 pre-step kinetic energy (MeV).
//     parent_dir_x/y/z    cached C12 pre-step momentum direction (unit).
//     creator_process_id  electron creator process enum (legend below).
//     birth_density_g_cm3 material density at the track's first step.
//
// Parent-state source (explicit by design): the most recent C12
// (PDG 1000060120) pre-step state observed in the SAME event prior to the
// electron's first step, cached per event inside this scorer. This is a
// last-known-state approximation, NOT the exact creation-time parent state:
// Geant4 does not retain the parent track object at the daughter's first
// step. The projection convention is:
//
//     longitudinal = (deposit_mid - electron_birth) . parent_dir
//     radial       = |(deposit_mid - electron_birth) - longitudinal*parent_dir|
//
// Never assume the parent travels along world +z; the analyzer projects
// against the recorded per-row parent_dir and carries rotation tests.
//
// Creator-process legend (stable; unknown/future processes map to 9):
//   0 = primary (no creator process), 1 = ionIoni, 2 = eIoni, 3 = compt,
//   4 = conv, 5 = phot, 6 = eBrem, 7 = CoulombScat, 8 = msc, 9 = other.
class CarbonElectronDepositNtuple : public TsVNtupleScorer {
public:
    CarbonElectronDepositNtuple(TsParameterManager*, TsMaterialManager*,
        TsGeometryManager*, TsScoringManager*, TsExtensionManager*,
        G4String, G4String, G4String, G4bool);
    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
private:
    G4int run_, event_, track_, parent_, pdg_, step_;
    G4double birth_x_, birth_y_, birth_z_, birth_ke_;
    G4double pre_x_, pre_y_, pre_z_, post_x_, post_y_, post_z_;
    G4double edep_, pre_ke_, post_ke_, weight_, density_;
    // v2 additions (schema_version == 2).
    G4int schema_version_, parent_valid_, creator_process_id_;
    G4double parent_ke_, parent_dir_x_, parent_dir_y_, parent_dir_z_;
    G4double birth_density_;
    struct ParentState {
        G4double ke;
        G4double dir_x, dir_y, dir_z;
    };
    // Same-event last-known C12 pre-step state keyed by (run, event).
    std::map<std::pair<G4int, G4int>, ParentState> parent_cache_;
    // Per-electron-track birth conditioning keyed by (run, event, track):
    // snapshot at the track's first step, replayed verbatim on later steps.
    std::map<std::tuple<G4int, G4int, G4int>, ParentState> track_parent_cache_;
    std::map<std::tuple<G4int, G4int, G4int>, G4int> track_parent_valid_;
    // Birth density per track keyed by (run, event, track).
    std::map<std::tuple<G4int, G4int, G4int>, G4double> birth_density_cache_;
    static G4int CreatorProcessId(const G4String& name, bool is_primary);
};

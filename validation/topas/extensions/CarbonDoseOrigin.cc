// Scorer for CarbonDoseOrigin

#include "CarbonDoseOrigin.hh"

#include "G4ParticleDefinition.hh"
#include "G4Material.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"

#include <array>
#include <cmath>
#include <unordered_map>

namespace {

enum class OriginCategory : G4int {
    PrimaryC12 = 0,
    SecondaryCarbon,
    Boron,
    Beryllium,
    Lithium,
    Helium,
    Proton,
    OtherCharged,
    Neutron,
    Gamma,
    NeutralOther,
    Unresolved,
};

constexpr std::array<const char*, 12> kCategoryNames{
    "primary_c12",
    "secondary_carbon",
    "boron",
    "beryllium",
    "lithium",
    "helium",
    "proton",
    "other_charged",
    "neutron",
    "gamma",
    "neutral_other",
    "unresolved",
};

struct ThreadOriginState {
    G4int event_id = -1;
    std::unordered_map<G4int, OriginCategory> track_origins;
    G4int cached_track_id = -1;
    G4int cached_step_number = -1;
    OriginCategory cached_category = OriginCategory::Unresolved;
};

thread_local ThreadOriginState g_origin_state;

bool IsNeutralSource(OriginCategory category) {
    return category == OriginCategory::Neutron || category == OriginCategory::Gamma ||
           category == OriginCategory::NeutralOther;
}

bool IsElectron(const G4ParticleDefinition* definition) {
    const G4int pdg = definition->GetPDGEncoding();
    return pdg == 11 || pdg == -11;
}

OriginCategory ClassifyIndependentTrack(const G4Track* track, bool is_primary) {
    const G4ParticleDefinition* definition = track->GetDefinition();
    const G4String& name = definition->GetParticleName();
    const G4int atomic_number = definition->GetAtomicNumber();
    const G4int atomic_mass = definition->GetAtomicMass();

    if (name == "neutron") {
        return OriginCategory::Neutron;
    }
    if (name == "gamma") {
        return OriginCategory::Gamma;
    }

    if (is_primary && atomic_number == 6 && atomic_mass == 12) {
        return OriginCategory::PrimaryC12;
    }
    if (name == "proton" || (atomic_number == 1 && atomic_mass == 1)) {
        return OriginCategory::Proton;
    }
    switch (atomic_number) {
        case 6:
            return OriginCategory::SecondaryCarbon;
        case 5:
            return OriginCategory::Boron;
        case 4:
            return OriginCategory::Beryllium;
        case 3:
            return OriginCategory::Lithium;
        case 2:
            return OriginCategory::Helium;
        default:
            break;
    }

    // In OpenTOPAS 4.1 / Geant4 11.1, a GenericIon dynamic charge can still
    // be zero at BeginOfTrack. Nuclear identity therefore has to be resolved
    // before consulting the dynamic charge. For non-nuclear particles the
    // charge is initialized and remains the correct discriminator.
    const G4double charge = track->GetDynamicParticle()->GetCharge() / eplus;
    if (std::abs(charge) < 0.5) {
        return OriginCategory::NeutralOther;
    }
    return OriginCategory::OtherCharged;
}

OriginCategory ResolveOrigin(const G4Track* track, G4int event_id) {
    ThreadOriginState& state = g_origin_state;
    if (state.event_id != event_id) {
        state.event_id = event_id;
        state.track_origins.clear();
        state.cached_track_id = -1;
        state.cached_step_number = -1;
    }

    const G4int track_id = track->GetTrackID();
    const G4int step_number = track->GetCurrentStepNumber();
    if (state.cached_track_id == track_id && state.cached_step_number == step_number) {
        return state.cached_category;
    }

    const auto existing = state.track_origins.find(track_id);
    if (existing != state.track_origins.end()) {
        state.cached_track_id = track_id;
        state.cached_step_number = step_number;
        state.cached_category = existing->second;
        return existing->second;
    }

    OriginCategory category = OriginCategory::Unresolved;
    const G4int parent_id = track->GetParentID();
    if (parent_id == 0) {
        category = ClassifyIndependentTrack(track, true);
    } else {
        const auto parent = state.track_origins.find(parent_id);
        const bool parent_known = parent != state.track_origins.end();
        if (parent_known && IsNeutralSource(parent->second)) {
            category = parent->second;
        } else if (IsElectron(track->GetDefinition())) {
            category = parent_known ? parent->second : OriginCategory::Unresolved;
        } else {
            category = ClassifyIndependentTrack(track, false);
        }
    }

    state.track_origins.emplace(track_id, category);
    state.cached_track_id = track_id;
    state.cached_step_number = step_number;
    state.cached_category = category;
    return category;
}

G4int ParseCategory(const G4String& requested) {
    for (std::size_t index = 0; index < kCategoryNames.size(); ++index) {
        if (requested == kCategoryNames[index]) {
            return static_cast<G4int>(index);
        }
    }
    return -1;
}

}  // namespace

CarbonDoseOrigin::CarbonDoseOrigin(
    TsParameterManager* parameter_manager,
    TsMaterialManager* material_manager,
    TsGeometryManager* geometry_manager,
    TsScoringManager* scoring_manager,
    TsExtensionManager* extension_manager,
    G4String scorer_name,
    G4String quantity,
    G4String output_file,
    G4bool is_sub_scorer)
    : TsVBinnedScorer(
          parameter_manager,
          material_manager,
          geometry_manager,
          scoring_manager,
          extension_manager,
          scorer_name,
          quantity,
          output_file,
          is_sub_scorer) {
    SetUnit("Gy");
    if (!fPm->ParameterExists(GetFullParmName("OriginCategory"))) {
        G4cerr << "Missing required parameter: " << GetFullParmName("OriginCategory") << G4endl;
        fPm->AbortSession(1);
    }
    const G4String requested = fPm->GetStringParameter(GetFullParmName("OriginCategory"));
    selected_category_ = ParseCategory(requested);
    if (selected_category_ < 0) {
        G4cerr << GetFullParmName("OriginCategory") << " has unsupported value: " << requested
               << G4endl;
        fPm->AbortSession(1);
    }
}

CarbonDoseOrigin::~CarbonDoseOrigin() = default;

void CarbonDoseOrigin::UserHookForBeginOfTrack(const G4Track* track) {
    // Register every track before transport, including parents that never enter
    // the scored phantom. This is required to attribute electrons created in
    // upstream/world material to their actual charged or neutral-source lineage.
    ResolveOrigin(track, GetEventID());
}

G4bool CarbonDoseOrigin::ProcessHits(G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }

    const OriginCategory category = ResolveOrigin(step->GetTrack(), GetEventID());
    if (static_cast<G4int>(category) != selected_category_) {
        return false;
    }

    const G4double energy_deposit = step->GetTotalEnergyDeposit();
    if (energy_deposit <= 0.0) {
        return false;
    }
    const G4double density = step->GetPreStepPoint()->GetMaterial()->GetDensity();
    ResolveSolid(step);
    G4double dose = energy_deposit / (density * GetCubicVolume(step));
    dose *= step->GetPreStepPoint()->GetWeight();
    AccumulateHit(step, dose);
    return true;
}

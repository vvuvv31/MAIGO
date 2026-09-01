// Scorer for ChargedOriginDoseToMedium

#include "ChargedOriginDoseToMedium.hh"

#include "G4Exception.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4Track.hh"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

namespace {

G4String ReadRequiredString(
    TsParameterManager* parameter_manager,
    const G4String& parameter_name) {
    if (!parameter_manager->ParameterExists(parameter_name)) {
        G4ExceptionDescription description;
        description << "ChargedOriginDoseToMedium requires parameter "
                    << parameter_name << ".";
        G4Exception("ChargedOriginDoseToMedium", "MissingOriginCategory",
                    FatalException, description);
        return "unclassified";
    }
    return parameter_manager->GetStringParameter(parameter_name);
}

std::string Lower(const G4String& input) {
    std::string value(input);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    return value;
}

}  // namespace

ChargedOriginDoseToMedium::ChargedOriginDoseToMedium(
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
          parameter_manager, material_manager, geometry_manager,
          scoring_manager, extension_manager, scorer_name, quantity,
          output_file, is_sub_scorer),
      requested_category_(ParseRequestedCategory(ReadRequiredString(
          parameter_manager, GetFullParmName("OriginCategory")))) {
    SetUnit("Gy");
}

ChargedOriginDoseToMedium::~ChargedOriginDoseToMedium() = default;

ChargedOriginDoseToMedium::OriginCategory
ChargedOriginDoseToMedium::ParseRequestedCategory(const G4String& input) {
    const auto value = Lower(input);
    if (value == "primary_c") return OriginCategory::PrimaryC;
    if (value == "secondary_c") return OriginCategory::SecondaryC;
    if (value == "boron" || value == "z5") return OriginCategory::Boron;
    if (value == "beryllium" || value == "z4") return OriginCategory::Beryllium;
    if (value == "be6") return OriginCategory::Be6;
    if (value == "be7") return OriginCategory::Be7;
    if (value == "be9") return OriginCategory::Be9;
    if (value == "be10") return OriginCategory::Be10;
    if (value == "be_other") return OriginCategory::BeOther;
    if (value == "lithium" || value == "z3") return OriginCategory::Lithium;
    if (value == "helium" || value == "z2") return OriginCategory::Helium;
    if (value == "z1" || value == "hydrogen") return OriginCategory::Z1;
    if (value == "other_charged") return OriginCategory::OtherCharged;
    if (value == "neutral_origin") return OriginCategory::NeutralOrigin;
    if (value == "unclassified") return OriginCategory::Unclassified;

    G4ExceptionDescription description;
    description << "Unknown OriginCategory: " << input
                << ". Expected primary_c, secondary_c, boron, beryllium, "
                << "be6, be7, be9, be10, be_other, lithium, helium, z1, "
                << "other_charged, neutral_origin, or unclassified.";
    G4Exception("ChargedOriginDoseToMedium", "InvalidOriginCategory",
                FatalException, description);
    return OriginCategory::Unclassified;
}

ChargedOriginDoseToMedium::OriginCategory
ChargedOriginDoseToMedium::CategoryForChargedNucleus(
    const G4Track* track, const G4ParticleDefinition* definition) {
    const G4int z = definition->GetAtomicNumber();
    if (z == 6) {
        return track->GetParentID() == 0
                   ? OriginCategory::PrimaryC
                   : OriginCategory::SecondaryC;
    }
    if (z == 5) return OriginCategory::Boron;
    if (z == 4) {
        const G4int a = definition->GetAtomicMass();
        if (a == 6) return OriginCategory::Be6;
        if (a == 7) return OriginCategory::Be7;
        if (a == 9) return OriginCategory::Be9;
        if (a == 10) return OriginCategory::Be10;
        return OriginCategory::BeOther;
    }
    if (z == 3) return OriginCategory::Lithium;
    if (z == 2) return OriginCategory::Helium;
    if (z == 1) return OriginCategory::Z1;
    return OriginCategory::OtherCharged;
}

ChargedOriginDoseToMedium::OriginCategory
ChargedOriginDoseToMedium::ResolveOrigin(const G4Track* track) {
    if (track == nullptr || track->GetDefinition() == nullptr) {
        return OriginCategory::Unclassified;
    }
    const auto existing = track_origins_.find(track->GetTrackID());
    if (existing != track_origins_.end()) return existing->second;

    const auto* definition = track->GetDefinition();
    const G4int parent_id = track->GetParentID();
    const auto parent = track_origins_.find(parent_id);
    const bool parent_is_neutral_origin =
        parent != track_origins_.end() &&
        parent->second == OriginCategory::NeutralOrigin;

    OriginCategory category = OriginCategory::Unclassified;
    if (parent_is_neutral_origin) {
        category = OriginCategory::NeutralOrigin;
    } else if (definition->GetAtomicNumber() > 0) {
        category = CategoryForChargedNucleus(track, definition);
    } else if (definition->GetPDGCharge() == 0.0) {
        category = OriginCategory::NeutralOrigin;
    } else {
        // Electrons, positrons, muons, and other non-nuclear charged tracks
        // inherit a known charged ancestor.  A primary lepton or a child whose
        // parent never entered this scored component remains unclassified.
        category = parent == track_origins_.end()
                       ? OriginCategory::Unclassified
                       : parent->second;
    }

    track_origins_.emplace(track->GetTrackID(), category);
    return category;
}

G4bool ChargedOriginDoseToMedium::ProcessHits(
    G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (step == nullptr || step->GetTrack() == nullptr) return false;

    const G4int run_id = GetRunID();
    const G4int event_id = GetEventID();
    if (run_id != seen_run_id_ || event_id != seen_event_id_) {
        track_origins_.clear();
        seen_run_id_ = run_id;
        seen_event_id_ = event_id;
    }

    const auto origin = ResolveOrigin(step->GetTrack());
    const bool any_beryllium =
        origin == OriginCategory::Be6 || origin == OriginCategory::Be7 ||
        origin == OriginCategory::Be9 || origin == OriginCategory::Be10 ||
        origin == OriginCategory::BeOther;
    if (origin != requested_category_ &&
        !(requested_category_ == OriginCategory::Beryllium && any_beryllium)) {
        return false;
    }

    const G4double energy_deposit = step->GetTotalEnergyDeposit();
    if (!(energy_deposit > 0.0)) return false;
    const G4double density = step->GetPreStepPoint()->GetMaterial()->GetDensity();
    ResolveSolid(step);
    G4double dose = energy_deposit / (density * GetCubicVolume(step));
    dose *= step->GetPreStepPoint()->GetWeight();
    AccumulateHit(step, dose);
    return true;
}

void ChargedOriginDoseToMedium::UserHookForEndOfEvent() {
    track_origins_.clear();
    seen_run_id_ = -1;
    seen_event_id_ = -1;
}

void ChargedOriginDoseToMedium::Clear() {
    TsVBinnedScorer::Clear();
    track_origins_.clear();
    seen_run_id_ = -1;
    seen_event_id_ = -1;
}

// Scorer for CarbonMaterialPropertiesNtuple

#include "CarbonMaterialPropertiesNtuple.hh"

#include "G4Material.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"

CarbonMaterialPropertiesNtuple::CarbonMaterialPropertiesNtuple(
    TsParameterManager* parameter_manager,
    TsMaterialManager* material_manager,
    TsGeometryManager* geometry_manager,
    TsScoringManager* scoring_manager,
    TsExtensionManager* extension_manager,
    G4String scorer_name,
    G4String quantity,
    G4String output_file,
    G4bool is_sub_scorer)
    : TsVNtupleScorer(parameter_manager, material_manager, geometry_manager,
                      scoring_manager, extension_manager, scorer_name, quantity,
                      output_file, is_sub_scorer) {
    fNtuple->RegisterColumnS(&material_name_, "Material Name");
    fNtuple->RegisterColumnD(&density_g_per_cm3_, "Density (g/cm3)", "");
    fNtuple->RegisterColumnD(&radiation_length_mm_, "Radiation Length (mm)", "");
    fNtuple->RegisterColumnD(&mass_radiation_length_g_per_cm2_,
                             "Mass Radiation Length (g/cm2)", "");
}

CarbonMaterialPropertiesNtuple::~CarbonMaterialPropertiesNtuple() = default;

G4bool CarbonMaterialPropertiesNtuple::ProcessHits(
    G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (filled_ || step->GetTrack()->GetParentID() != 0) {
        return false;
    }
    const G4Material* material = step->GetPreStepPoint()->GetMaterial();
    material_name_ = material->GetName();
    density_g_per_cm3_ = material->GetDensity() / (g / cm3);
    radiation_length_mm_ = material->GetRadlen() / mm;
    mass_radiation_length_g_per_cm2_ =
        (material->GetRadlen() / cm) * density_g_per_cm3_;
    fNtuple->Fill();
    filled_ = true;
    return true;
}

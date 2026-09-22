// Executed-reference probe of G4UrbanMscModel conversion math.
//
// Drives the REAL Geant4 11.3.2 model (local install used by TOPAS 4.2.p3):
//   ComputeTruePathLengthLimit -> ComputeGeomPathLength -> ComputeTrueStepLength
// on a (E, t_request) grid in Water_75eV, dumping (t_req, t_lim, g, t_inv,
// lambda0, range). Validates MAIGO's true<->geom conversion branch-by-branch,
// including the t<1nm early exit and tiny-tau behavior, without re-reading
// any G4 source.
//
// Usage: g4_conv_oracle <out_csv>
// Build (after `source /software/env_geant4.sh`):
//   g++ -O2 -o g4_conv_oracle g4_conv_oracle.cc $(geant4-config --cflags --libs)
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "G4RunManager.hh"
#include "G4VModularPhysicsList.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserRunAction.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4ProductionCuts.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4Element.hh"
#include "G4IonTable.hh"
#include "G4GenericIon.hh"
#include "G4hMultipleScattering.hh"
#include "G4UrbanMscModel.hh"
#include "G4VEmModel.hh"
#include "G4ProcessManager.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4DynamicParticle.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

namespace {

class Detector : public G4VUserDetectorConstruction {
public:
    G4VPhysicalVolume* Construct() override {
        auto* nist = G4NistManager::Instance();
        auto* galactic = nist->FindOrBuildMaterial("G4_Galactic");
        auto* worldS = new G4Box("world", 500 * mm, 500 * mm, 500 * mm);
        auto* worldL = new G4LogicalVolume(worldS, galactic, "world");
        auto* phys =
            new G4PVPlacement(nullptr, G4ThreeVector(), worldL, "world",
                              nullptr, false, 0);
        auto* elH = new G4Element("Hydrogen", "H", 1., 1.008 * g / mole);
        auto* elO = new G4Element("Oxygen", "O", 8., 16.00 * g / mole);
        auto* water = new G4Material("Water75eV", 1.0 * g / cm3, 2);
        water->AddElement(elH, 2);
        water->AddElement(elO, 1);
        water->GetIonisation()->SetMeanExcitationEnergy(75.0 * eV);
        auto* slabS = new G4Box("slab", 50 * mm, 50 * mm, 50 * mm);
        auto* slabL = new G4LogicalVolume(slabS, water, "slab");
        new G4PVPlacement(nullptr, G4ThreeVector(), slabL, "slab", worldL,
                          false, 0);
        auto* cu = G4NistManager::Instance()->FindOrBuildMaterial("G4_Cu");
        auto* cuS = new G4Box("cuslab", 50 * mm, 50 * mm, 50 * mm);
        auto* cuL = new G4LogicalVolume(cuS, cu, "cuslab");
        new G4PVPlacement(nullptr, G4ThreeVector(200 * mm, 0, 0), cuL,
                          "cuslab", worldL, false, 0);
        return phys;
    }
};

class PhysList : public G4VModularPhysicsList {
public:
    PhysList() { RegisterPhysics(new G4EmStandardPhysics_option4(0)); }
};

}  // namespace

class Gun : public G4VUserPrimaryGeneratorAction {
public:
    void GeneratePrimaries(G4Event*) override {}
};

class RunAct : public G4UserRunAction {
public:
    std::string outPath;
    void BeginOfRunAction(const G4Run*) override;
};

void RunAct::BeginOfRunAction(const G4Run*) {
    auto* c12 = G4IonTable::GetIonTable()->GetIon(6, 12, 0.0);
    // NOTE: probe here (not after Initialize): the ion process model list
    // is populated only once tracking starts (found 2026-09-22: n=0 after
    // Initialize, populated in BeginOfRun like the slab oracle's RunAct).
    auto* pm = c12->GetProcessManager();
    G4hMultipleScattering* msc = nullptr;
    for (G4int i = 0; i < pm->GetProcessListLength(); ++i) {
        msc = dynamic_cast<G4hMultipleScattering*>((*pm->GetProcessList())[i]);
        if (msc) break;
    }
    if (!msc) {
        G4cout << "CONV ERROR: ionmsc not found" << G4endl;
        return;
    }
    G4cout << "CONV ionmsc models: n=" << msc->NumberOfModels() << G4endl;
    for (G4int k = 0; k < msc->NumberOfModels(); ++k) {
        auto* m = msc->GetModelByIndex(k);
        G4cout << "  [" << k << "] "
               << (m ? m->GetName() : "null") << G4endl;
    }
    G4UrbanMscModel* urban = nullptr;
    for (G4int k = 0; k < msc->NumberOfModels(); ++k) {
        urban = dynamic_cast<G4UrbanMscModel*>(msc->GetModelByIndex(k));
        if (urban) break;
    }
    if (!urban) {
        G4cout << "CONV ERROR: model is not Urban" << G4endl;
        return;
    }
    urban->InitialiseParameters(c12);
    urban->DumpParameters(G4cout);

    G4LogicalVolume* slabL = nullptr;
    for (auto* lv : *G4LogicalVolumeStore::GetInstance()) {
        if (lv->GetName() == "slab") slabL = lv;
    }
    if (!slabL) {
        G4cout << "CONV ERROR: slab volume not found" << G4endl;
        return;
    }
    const G4MaterialCutsCouple* couple = slabL->GetMaterialCutsCouple();
    G4Material* water = slabL->GetMaterial();
    urban->SetCurrentCouple(couple);
    // Cuts-dependence of GetRange: second couple with 0.05 mm production
    // cuts (TOPAS default-region value). If GetRange moves with cuts, the
    // MSC currentRange mirror must use the production-couple values.
    G4ProductionCuts* pc05 = new G4ProductionCuts();
    pc05->SetProductionCut(0.05 * mm);
    G4MaterialCutsCouple* couple05 =
        new G4MaterialCutsCouple(water, pc05);
    {
        // Copper couple for the Cu currentRange mirror.
        G4LogicalVolume* cuL = nullptr;
        for (auto* lv : *G4LogicalVolumeStore::GetInstance()) {
            if (lv->GetName() == "cuslab") cuL = lv;
        }
        if (!cuL) {
            G4cout << "CONV ERROR: cuslab not found" << G4endl;
            return;
        }
        const G4MaterialCutsCouple* cuCouple = cuL->GetMaterialCutsCouple();
        const double Eprobe[] = {0.12 * MeV, 0.6 * MeV,  2.4 * MeV,
                                 60.0 * MeV, 3000.0 * MeV, 4800.0 * MeV};
        for (double E : Eprobe) {
            const double r_def =
                urban->GetRange(c12, E, couple) / mm;
            const double r_05 = urban->GetRange(c12, E, couple05) / mm;
            const double lam_def =
                urban->GetTransportMeanFreePath(c12, E) / mm;
            urban->SetCurrentCouple(couple05);
            const double lam_05 =
                urban->GetTransportMeanFreePath(c12, E) / mm;
            urban->SetCurrentCouple(couple);
            const double r_cu = urban->GetRange(c12, E, cuCouple);
            G4cout << "CONV cuts: E=" << E / MeV
                   << " R_default=" << r_def << " R_0.05mm=" << r_05
                   << " lam_default=" << lam_def << " lam_0.05mm=" << lam_05
                   << " R_Cu=" << (r_cu / mm) << G4endl;
        }
    }

    FILE* f = std::fopen(outPath.c_str(), "w");
    std::fprintf(f, "# E_MeV_total,t_req_mm,t_lim_mm,g_mm,t_inv_mm,"
                    "lambda0_mm,range_mm\n");
    const double egrid_u[] = {0.2, 0.5, 1.0, 5.0, 20.0, 50.0, 100.0, 250.0};
    const double tgrid[] = {1.0e-7, 2.0e-7, 5.0e-7, 7.0e-7, 1.0e-6, 2.0e-6,
                            5.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 5.0e-3, 1.0e-2,
                            2.5e-2, 5.0e-2, 1.0e-1, 2.5e-1};
    for (double eu : egrid_u) {
        const double E = eu * 12.0 * MeV;
        // Fresh track per energy: StartTracking arms firstStep exactly like
        // production; interior probes then reuse persistent tlimit.
        G4ThreeVector pos(0, 0, 0), mom(0, 0, 1);
        G4DynamicParticle dp(c12, mom, E);
        G4Track track(&dp, 0.0, pos);
        G4Step step;
        step.GetPreStepPoint()->SetPosition(pos);
        step.GetPreStepPoint()->SetKineticEnergy(E);
        step.GetPreStepPoint()->SetMaterial(water);
        step.GetPreStepPoint()->SetMaterialCutsCouple(couple);
        track.SetStep(&step);
        urban->StartTracking(&track);
        const double lam = urban->GetTransportMeanFreePath(c12, E) / mm;
        const double rng = urban->GetRange(c12, E, couple) / mm;
        for (double tmm : tgrid) {
            G4double physStep = tmm * mm;
            G4double tlim = urban->ComputeTruePathLengthLimit(track, physStep);
            G4double g = urban->ComputeGeomPathLength(tlim);
            G4double tinv = urban->ComputeTrueStepLength(g);
            std::fprintf(f, "%.6f,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n", E / MeV,
                         tmm, tlim / mm, g / mm, tinv / mm, lam, rng);
        }
        // Over-range probes (t/R = 1.5, 3.0): reveals G4's internal
        // currentRange (CSDA-like vs restricted) via the t_lim cap.
        for (double frac : {1.5, 3.0}) {
            const double tmm = frac * rng;
            G4double physStep = tmm * mm;
            G4double tlim = urban->ComputeTruePathLengthLimit(track, physStep);
            G4double g = urban->ComputeGeomPathLength(tlim);
            G4double tinv = urban->ComputeTrueStepLength(g);
            std::fprintf(f, "%.6f,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n", E / MeV,
                         tmm, tlim / mm, g / mm, tinv / mm, lam, rng);
        }
        // Near-range branches (t/R = 0.5, 0.9, 1.0): range/par1>=0 domain.
        for (double frac : {0.5, 0.9, 1.0}) {
            const double tmm = frac * rng;
            G4double physStep = tmm * mm;
            G4double tlim = urban->ComputeTruePathLengthLimit(track, physStep);
            G4double g = urban->ComputeGeomPathLength(tlim);
            G4double tinv = urban->ComputeTrueStepLength(g);
            std::fprintf(f, "%.6f,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n", E / MeV,
                         tmm, tlim / mm, g / mm, tinv / mm, lam, rng);
        }
    }
    std::fclose(f);
    G4cout << "CONV wrote " << outPath << G4endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: g4_conv_oracle <out_csv>\n");
        return 2;
    }
    G4RunManager run;
    run.SetUserInitialization(new Detector());
    run.SetUserInitialization(new PhysList());
    run.SetUserAction(new Gun());
    auto* ra = new RunAct();
    ra->outPath = argv[1];
    run.SetUserAction(ra);
    run.Initialize();
    run.BeamOn(1);
    return 0;
}

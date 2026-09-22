// Direct Geant4 11.3.2 Urban oracle for the MAIGO Urban default-migration.
//
// Links against the LOCAL install (/software/geant4-11.3.2) actually used by
// the TOPAS 4.2.p3 reference runs. No physics is re-implemented here: the
// real G4UrbanMscModel (via g4em-standard_opt4, EM-only list) transports
// C12 through a Water_75eV-equivalent box and reports:
//   1. Urban model DumpParameters for GenericIon (stepping algorithm,
//      range factor, lateral displacement, skin, lambda limit);
//   2. ComputeCrossSectionPerAtom grid (validates MAIGO's per-atom tables);
//   3. exit phase space + step-length/energy-loss aggregates for thin slabs
//      (validates step competition and sampling statistics).
//
// Usage:
//   g4_urban_oracle <slab_mm> <events> <maxstep_mm> <out_prefix> [e_mev_u]
//
// Build (after `source /software/env_geant4.sh`):
//   g++ -O2 -o g4_urban_oracle g4_urban_oracle.cc \
//       $(geant4-config --cflags --libs)
//
// Reference only: this program OBSERVES Geant4; it modifies no G4 algorithm.
#include <cmath>
#include <cstdio>
#include <vector>

#include "G4RunManager.hh"
#include "G4VModularPhysicsList.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserRunAction.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4ParticleGun.hh"
#include "G4IonTable.hh"
#include "G4Electron.hh"
#include "G4GenericIon.hh"
#include "G4hMultipleScattering.hh"
#include "G4UrbanMscModel.hh"
#include "G4VEmModel.hh"
#include "G4VEmProcess.hh"
#include "G4ProcessManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4UserLimits.hh"
#include "G4StepLimiter.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

namespace {

double gSlabMm = 10.0;
double gMaxStepMm = 0.05;
double gEMeVu = 250.0;
std::string gOutPrefix = "oracle";
G4LogicalVolume* gWaterLog = nullptr;

// Per-run aggregates (sequential run manager: no races).
struct Agg {
    long events = 0;
    long exitN = 0;
    double sumExitE = 0.0, sumExitE2 = 0.0;
    double sumThx = 0.0, sumThx2 = 0.0, sumThy = 0.0, sumThy2 = 0.0;
    double sumX = 0.0, sumX2 = 0.0, sumY = 0.0, sumY2 = 0.0;
    double sumCovXT = 0.0, sumCovYT = 0.0;
    std::vector<double> thetas;  // space angles of exiters, mrad
    std::vector<double> stepLens;  // sampled: every 97th water step length
    long stepCount = 0;
    double sumStepLen = 0.0, sumStepLen2 = 0.0;
    double sumPreE = 0.0, sumPostE = 0.0;
    long stepRows = 0;
    FILE* stepFile = nullptr;
    long lastExitEvent = -1;
};
Agg gAgg;

class Detector2 : public G4VUserDetectorConstruction {
public:
    G4VPhysicalVolume* physWorld = nullptr;
    G4VPhysicalVolume* Construct() override { return physWorld; }
    void Build() {
        auto* nist = G4NistManager::Instance();
        auto* galactic = nist->FindOrBuildMaterial("G4_Galactic");
        auto* worldS = new G4Box("world", 500 * mm, 500 * mm, 500 * mm);
        auto* worldL = new G4LogicalVolume(worldS, galactic, "world");
        physWorld = new G4PVPlacement(nullptr, G4ThreeVector(), worldL,
                                      "world", nullptr, false, 0);
        auto* elH = new G4Element("Hydrogen", "H", 1., 1.008 * g / mole);
        auto* elO = new G4Element("Oxygen", "O", 8., 16.00 * g / mole);
        auto* water = new G4Material("Water75eV", 1.0 * g / cm3, 2);
        water->AddElement(elH, 2);
        water->AddElement(elO, 1);
        water->GetIonisation()->SetMeanExcitationEnergy(75.0 * eV);
        auto* slabS =
            new G4Box("slab", 50 * mm, 50 * mm, 0.5 * gSlabMm * mm);
        auto* slabL = new G4LogicalVolume(slabS, water, "slab");
        slabL->SetUserLimits(new G4UserLimits(gMaxStepMm * mm));
        new G4PVPlacement(nullptr, G4ThreeVector(0, 0, 0.5 * gSlabMm * mm),
                          slabL, "slab", worldL, false, 0);
        gWaterLog = slabL;
        G4cout << "ORACLE material: I = "
               << water->GetIonisation()->GetMeanExcitationEnergy() / eV
               << " eV, Zeff(ionis) = "
               << water->GetIonisation()->GetZeffective() << ", radlen = "
               << water->GetRadlen() / mm << " mm, density = "
               << water->GetDensity() / (g / cm3) << " g/cm3" << G4endl;
    }
};

class Gun : public G4VUserPrimaryGeneratorAction {
public:
    G4ParticleGun gun{1};
    bool armed = false;
    Gun() {
        // Ion table is unavailable until ConstructParticle runs; arm lazily.
        gun.SetParticlePosition(G4ThreeVector(0, 0, -1.0 * mm));
        gun.SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
    }
    void GeneratePrimaries(G4Event* e) override {
        if (!armed) {
            auto* ion = G4IonTable::GetIonTable()->GetIon(6, 12, 0.0);
            gun.SetParticleDefinition(ion);
            gun.SetParticleEnergy(gEMeVu * MeV * 12.0);
            armed = true;
        }
        gun.GeneratePrimaryVertex(e);
    }
};

class StepAct : public G4UserSteppingAction {
public:
    void UserSteppingAction(const G4Step* step) override {
        auto* track = step->GetTrack();
        if (track->GetParticleDefinition() !=
            G4GenericIon::GenericIonDefinition() &&
            track->GetParticleDefinition()->GetAtomicNumber() != 6)
            return;
        if (step->GetPreStepPoint()->GetPhysicalVolume() == nullptr)
            return;
        const bool inWater =
            step->GetPreStepPoint()->GetPhysicalVolume()
                ->GetLogicalVolume() == gWaterLog;
        if (!inWater) return;
        const double preE = step->GetPreStepPoint()->GetKineticEnergy() / MeV;
        const double postE =
            step->GetPostStepPoint()->GetKineticEnergy() / MeV;
        const double sl = step->GetStepLength() / mm;
        gAgg.stepCount++;
        gAgg.sumStepLen += sl;
        gAgg.sumStepLen2 += sl * sl;
        gAgg.sumPreE += preE;
        gAgg.sumPostE += postE;
        if ((gAgg.stepCount % 97) == 0 && gAgg.stepLens.size() < 200000)
            gAgg.stepLens.push_back(sl);
        // Exit plane: post-step z crosses slab exit.
        const double zPre = step->GetPreStepPoint()->GetPosition().z() / mm;
        const double zPost =
            step->GetPostStepPoint()->GetPosition().z() / mm;
        // Exit plane with FP tolerance: tracks stopping exactly on the
        // navigator boundary can land one ulp below slab_mm.
        if (zPre < gSlabMm && zPost >= gSlabMm * (1.0 - 1.0e-9) &&
            step->GetPostStepPoint()->GetPhysicalVolume() != nullptr) {
            // One record per event: navigator sub-ulp structure can emit
            // the exit plane twice (landing ~1 ulp inside, then crossing).
            const long ev = G4EventManager::GetEventManager()
                                  ->GetConstCurrentEvent()
                                  ->GetEventID();
            if (ev == gAgg.lastExitEvent) return;
            gAgg.lastExitEvent = ev;
            const auto& dir =
                step->GetPostStepPoint()->GetMomentumDirection();
            const auto& pos = step->GetPostStepPoint()->GetPosition();
            const double thx = std::atan2(dir.x(), dir.z()) * 1000.0;
            const double thy = std::atan2(dir.y(), dir.z()) * 1000.0;
            const double x = pos.x() / mm, y = pos.y() / mm;
            gAgg.exitN++;
            gAgg.sumExitE += postE;
            gAgg.sumExitE2 += postE * postE;
            gAgg.sumThx += thx;
            gAgg.sumThx2 += thx * thx;
            gAgg.sumThy += thy;
            gAgg.sumThy2 += thy * thy;
            gAgg.sumX += x;
            gAgg.sumX2 += x * x;
            gAgg.sumY += y;
            gAgg.sumY2 += y * y;
            gAgg.sumCovXT += x * thx;
            gAgg.sumCovYT += y * thy;
            if (gAgg.thetas.size() < 500000)
                gAgg.thetas.push_back(
                    std::sqrt(thx * thx + thy * thy));
        }
    }
};

class RunAct : public G4UserRunAction {
public:
    void BeginOfRunAction(const G4Run*) override {
        // 1. Model parameters for GenericIon (the executed configuration).
        // Cross sections use a REAL C12 ion: GenericIon (+1, ~1u) would feed
        // the equivalent-energy map the wrong mass (found 2026-09-22: the
        // "2.09x tag difference" was this probe bug, not a G4 patch).
        auto* ion = G4GenericIon::GenericIon();
        auto* c12 = G4IonTable::GetIonTable()->GetIon(6, 12, 0.0);
        auto* pm = ion->GetProcessManager();
        {
            G4cout << "ORACLE GenericIon processes:";
            auto* pv = pm->GetProcessList();
            for (G4int i = 0; i < pm->GetProcessListLength(); ++i)
                G4cout << " [" << (*pv)[i]->GetProcessName() << "]";
            G4cout << G4endl;
        }
        G4hMultipleScattering* msc = nullptr;
        {
            auto* pv = pm->GetProcessList();
            for (G4int i = 0; i < pm->GetProcessListLength(); ++i) {
                msc = dynamic_cast<G4hMultipleScattering*>((*pv)[i]);
                if (msc) break;
            }
        }
        if (!msc) {
            G4cout << "ORACLE ERROR: ionmsc process not found" << G4endl;
            return;
        }
        auto* base = msc->GetModelByIndex(0);
        auto* urban = dynamic_cast<G4UrbanMscModel*>(base);
        G4cout << "ORACLE model for GenericIon: "
               << (urban ? base->GetName() : "NOT-URBAN") << G4endl;
        if (urban) urban->DumpParameters(G4cout);
        // 2. Cross-section grid: C12 total E vs H(Z=1)/O(Z=8) atoms.
        if (urban) {
            std::string fn = gOutPrefix + "_xsec.csv";
            FILE* f = std::fopen(fn.c_str(), "w");
            std::fprintf(f, "# particle,E_MeV,Z,sigma_cm2\n");
            const double egrid[] = {1.2,  6.0,   12.0,  60.0,  120.0,
                                    360.0, 1200.0, 3000.0, 3600.0};
            for (double E : egrid) {
                for (int Z : {1, 8}) {
                    const double sig = urban->ComputeCrossSectionPerAtom(
                        c12, E * MeV, double(Z), 0.0, 0.0, DBL_MAX);
                    std::fprintf(f, "C12,%.4f,%d,%.8e\n", E, Z,
                                 sig / cm2);
                }
            }
            // Electron probe at mapped-equivalent energies (no ion mapping):
            // isolates the high-E branch from the ion equivalent-energy map.
            auto* elec = G4Electron::Electron();
            for (double E : {6.0, 12.0, 5363.0}) {
                for (int Z : {1, 8}) {
                    const double sig = urban->ComputeCrossSectionPerAtom(
                        elec, E * MeV, double(Z), 0.0, 0.0, DBL_MAX);
                    std::fprintf(f, "e-,%.4f,%d,%.8e\n", E, Z,
                                 sig / cm2);
                }
            }
            // Fine C12 grid: pins the port's energy dependence against the
            // EXECUTED library (tag-vs-installed differences under study).
            {
                const double fine[] = {
                    1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 9.0, 12.0, 18.0, 24.0,
                    36.0, 60.0, 90.0, 120.0, 180.0, 240.0, 360.0, 600.0,
                    900.0, 1200.0, 1800.0, 2400.0, 3000.0, 3600.0};
                for (double E : fine) {
                    for (int Z : {1, 8}) {
                        const double sig = urban->ComputeCrossSectionPerAtom(
                            c12, E * MeV, double(Z), 0.0, 0.0, DBL_MAX);
                        std::fprintf(f, "C12fine,%.4f,%d,%.8e\n", E, Z,
                                     sig / cm2);
                    }
                }
            }
            std::fclose(f);
            G4cout << "ORACLE wrote " << fn << G4endl;
        }
    }
    void EndOfRunAction(const G4Run*) override {
        const double n = double(gAgg.exitN);
        std::string fn = gOutPrefix + "_summary.csv";
        FILE* f = std::fopen(fn.c_str(), "w");
        std::fprintf(f, "# slab_mm=%.4f maxstep_mm=%.4f E_MeV_u=%.2f\n",
                     gSlabMm, gMaxStepMm, gEMeVu);
        std::fprintf(f, "# events,exitN,exitFrac\n%ld,%ld,%.6f\n",
                     gAgg.events, gAgg.exitN,
                     double(gAgg.exitN) / double(std::max(1L, gAgg.events)));
        if (n > 0) {
            const double mE = gAgg.sumExitE / n;
            const double vThx =
                (gAgg.sumThx2 - gAgg.sumThx * gAgg.sumThx / n) / n;
            const double vThy =
                (gAgg.sumThy2 - gAgg.sumThy * gAgg.sumThy / n) / n;
            const double vX =
                (gAgg.sumX2 - gAgg.sumX * gAgg.sumX / n) / n;
            const double cXT =
                (gAgg.sumCovXT - gAgg.sumX * gAgg.sumThx / n) / n;
            std::fprintf(f, "# meanExitE_MeV,rmsExitE,varThx_mrad2,varThy_"
                            "mrad2,varX_mm2,covXThx\n%.6f,%.6f,%.6e,%.6e,"
                            "%.6e,%.6e\n",
                         mE,
                         std::sqrt(std::max(0.0, gAgg.sumExitE2 / n -
                                                       mE * mE)),
                         vThx, vThy, vX, cXT);
        }
        const double ns = double(gAgg.stepCount);
        if (ns > 0)
            std::fprintf(f, "# steps,meanStep_mm,rmsStep_mm,meanPreE,"
                            "meanPostE\n%ld,%.6e,%.6e,%.4f,%.4f\n",
                         gAgg.stepCount, gAgg.sumStepLen / ns,
                         std::sqrt(std::max(0.0, gAgg.sumStepLen2 / ns -
                                      (gAgg.sumStepLen / ns) *
                                          (gAgg.sumStepLen / ns))),
                         gAgg.sumPreE / ns, gAgg.sumPostE / ns);
        std::fclose(f);
        G4cout << "ORACLE wrote " << fn << " exitN=" << gAgg.exitN
               << G4endl;
        std::string ft = gOutPrefix + "_theta.csv";
        FILE* ftf = std::fopen(ft.c_str(), "w");
        for (double t : gAgg.thetas) std::fprintf(ftf, "%.6f\n", t);
        std::fclose(ftf);
        std::string fs = gOutPrefix + "_steplen.csv";
        FILE* fsf = std::fopen(fs.c_str(), "w");
        for (double s : gAgg.stepLens) std::fprintf(fsf, "%.7f\n", s);
        std::fclose(fsf);
    }
};

class PhysList : public G4VModularPhysicsList {
public:
    PhysList() { RegisterPhysics(new G4EmStandardPhysics_option4(0)); }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::printf("usage: %s slab_mm events maxstep_mm out_prefix "
                    "[e_mev_u]\n",
                    argv[0]);
        return 2;
    }
    gSlabMm = std::atof(argv[1]);
    const int events = std::atoi(argv[2]);
    gMaxStepMm = std::atof(argv[3]);
    gOutPrefix = argv[4];
    if (argc > 5) gEMeVu = std::atof(argv[5]);

    auto* run = new G4RunManager();
    auto* det = new Detector2();
    det->Build();
    run->SetUserInitialization(det);
    run->SetUserInitialization(new PhysList());
    run->SetUserAction(new Gun());
    run->SetUserAction(new StepAct());
    run->SetUserAction(new RunAct());
    run->Initialize();
    // Enforce the volume MaxStepSize like TOPAS does (G4UserLimits alone
    // is inert without a limiter process).
    G4GenericIon::GenericIon()->GetProcessManager()->AddDiscreteProcess(
        new G4StepLimiter("ionStepLimiter"));
    gAgg.events = events;
    run->BeamOn(events);
    delete run;
    return 0;
}

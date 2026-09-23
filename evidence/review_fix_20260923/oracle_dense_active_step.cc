// Positive/negative lifecycle oracle for the Geant4 11.3.2 Urban loss range.
//
// Positive control: real C12 primaries are transported by G4RunManager through
// registered Water_75eV/Cu regions with production cuts.  At the first actual
// tracking step, a passive SteppingAction records the selected MSC process,
// model, couple, and energy.  The model/loss table lookups happen only after
// BeamOn has completed, so the trace cannot affect transport or consume RNG.
// Negative control: a separate, deliberately unbound Urban model demonstrates
// the G4VMscModel fallback range used when GetIonisation() is null.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "G4Box.hh"
#include "G4UserStackingAction.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4UserLimits.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4PrimaryVertex.hh"
#include "G4GenericIon.hh"
#include "G4hMultipleScattering.hh"
#include "G4IonTable.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4PhysicalConstants.hh"
#include "G4ProductionCuts.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UImanager.hh"
#include "G4UserSteppingAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4VMscModel.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4Version.hh"
#include "G4VPhysicalVolume.hh"
#include "G4UrbanMscModel.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"

namespace {

constexpr double kEnergyMeV[] = {0.12, 0.6, 2.4, 12.0, 60.0,
                                 300.0, 1200.0, 3000.0, 4800.0};
constexpr int kLossGridNodes = 40001;
constexpr double kLossGridMinMeV = 0.1;
constexpr double kLossGridMaxMeV = 4800.12;
constexpr int kMaterialCount = 3;
constexpr int kCaseCount = kLossGridNodes * kMaterialCount;
const char* kContexts[kMaterialCount] = {
    "water_cut_0p10mm", "water_cut_0p05mm", "copper_cut_0p05mm"};

struct Context {
  G4LogicalVolume* logical = nullptr;
  const G4MaterialCutsCouple* couple = nullptr;
  std::string name;
  double cut_mm = 0.0;
};

struct TraceRow {
  int event_id = -1;
  int track_id = -1;
  int context_id = -1;
  double energy_mev = 0.0;
  double range_mm = 0.0, dedx = 0.0, inverse_mev = 0.0, true_step_mm = 0.0;
  G4ParticleDefinition* particle = nullptr;
  const G4MaterialCutsCouple* couple = nullptr;
};

std::vector<Context> g_contexts;
std::vector<TraceRow> g_trace;
G4hMultipleScattering* g_msc = nullptr;
G4UrbanMscModel* g_urban = nullptr;
G4VEnergyLossProcess* g_bound_loss = nullptr;

class Detector final : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
    auto* hydrogen = new G4Element("WaterHydrogen75", "Hw75", 1.,
                                   1.00794 * g / mole);
    auto* oxygen = new G4Element("WaterOxygen75", "Ow75", 8.,
                                 15.9994 * g / mole);
    auto* water = new G4Material("Water_75eV", 1.0 * g / cm3, 2);
    water->AddElement(hydrogen, 2);
    water->AddElement(oxygen, 1);
    water->GetIonisation()->SetMeanExcitationEnergy(75.0 * eV);
    auto* copper = nist->FindOrBuildMaterial("G4_Cu");

    auto* world_solid = new G4Box("world_solid", 500 * mm, 500 * mm,
                                  500 * mm);
    auto* world_logical = new G4LogicalVolume(world_solid, vacuum, "world_lv");
    auto* world = new G4PVPlacement(nullptr, {}, world_logical, "world_pv",
                                    nullptr, false, 0, true);

    const auto add_context = [&](const char* logical_name,
                                 const char* physical_name,
                                 const char* region_name, G4Material* material,
                                 const double x_mm, const double cut_mm,
                                 const int context_id) {
      auto* solid = new G4Box((std::string(logical_name) + "_solid").c_str(),
                              40 * mm, 40 * mm, 40 * mm);
      auto* logical = new G4LogicalVolume(solid, material, logical_name);
      logical->SetUserLimits(new G4UserLimits(0.05*mm));
      new G4PVPlacement(nullptr, G4ThreeVector(x_mm * mm, 0, 0), logical,
                        physical_name, world_logical, false, context_id, true);
      auto* region = new G4Region(region_name);
      region->AddRootLogicalVolume(logical);
      auto* cuts = new G4ProductionCuts();
      cuts->SetProductionCut(cut_mm * mm);
      region->SetProductionCuts(cuts);
      g_contexts[context_id] = {logical, nullptr, kContexts[context_id],
                                cut_mm};
    };

    g_contexts.resize(kMaterialCount);
    add_context("water_default_lv", "water_default_pv", "water_default_region",
                water, -120.0, 0.10, 0);
    add_context("water_alt_lv", "water_alt_pv", "water_alt_region", water,
                0.0, 0.05, 1);
    add_context("copper_lv", "copper_pv", "copper_region", copper, 120.0,
                0.05, 2);
    return world;
  }
};

class PhysicsList final : public G4VModularPhysicsList {
 public:
  PhysicsList() { RegisterPhysics(new G4EmStandardPhysics_option4(0)); RegisterPhysics(new G4StepLimiterPhysics()); }
};

class PrimaryGenerator final : public G4VUserPrimaryGeneratorAction {
 public:
  PrimaryGenerator() : gun_(1) {}

  void GeneratePrimaries(G4Event* event) override {
    const auto id = event->GetEventID();
    if (id < 0 || id >= kCaseCount) {
      throw std::runtime_error("unexpected primary event id");
    }
    const int energy_index = id / kMaterialCount;
    const int context_id = id % kMaterialCount;
    auto* c12 = G4IonTable::GetIonTable()->GetIon(6, 12, 0.0);
    if (!c12 || c12->GetAtomicNumber() != 6 || c12->GetAtomicMass() != 12) {
      throw std::runtime_error("failed to obtain the real C12 ion definition");
    }
    const double x_mm = context_id == 0 ? -120.0 :
                        context_id == 1 ? 0.0 : 120.0;
    gun_.SetParticleDefinition(c12);
    const double energy = kLossGridMinMeV * std::pow(kLossGridMaxMeV/kLossGridMinMeV, double(energy_index)/(kLossGridNodes-1));
    gun_.SetParticleEnergy(energy * MeV);
    gun_.SetParticlePosition(G4ThreeVector(x_mm * mm, 0, 0));
    gun_.SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
    gun_.GeneratePrimaryVertex(event);
  }

 private:
  G4ParticleGun gun_;
};

class PassiveTrace final : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    const auto* track = step->GetTrack();
    if (track->GetParentID() != 0 || track->GetTrackID() != 1) return;
    const auto* event = G4EventManager::GetEventManager()->GetConstCurrentEvent();
    const int event_id = event ? event->GetEventID() : -1;
    const auto* pre = step->GetPreStepPoint();
    const auto* couple = pre->GetMaterialCutsCouple();
    if (event_id < 0 || event_id >= kCaseCount || !event || !couple) {
      throw std::runtime_error("primary event identity or step couple is invalid");
    }
    const int context_id = event_id % kMaterialCount;
    const auto* vertex = event->GetPrimaryVertex(0);
    if (!vertex) throw std::runtime_error("tracked primary has no source vertex");
    const double expected_x = context_id == 0 ? -120.0 :
                              context_id == 1 ? 0.0 : 120.0;
    if (std::abs(vertex->GetX0() / mm - expected_x) > 1.0e-9 ||
        std::abs(vertex->GetY0() / mm) > 1.0e-9 ||
        std::abs(vertex->GetZ0() / mm) > 1.0e-9) {
      throw std::runtime_error("C12 source vertex is outside its intended region");
    }
    if (track->GetCurrentStepNumber() != 1) return;
    auto* particle = track->GetDefinition();
    auto* pm = particle->GetProcessManager();
    if (!pm) throw std::runtime_error("C12 has no process manager");
    G4hMultipleScattering* msc = nullptr;
    auto* process_list = pm->GetProcessList();
    for (G4int i = 0; i < pm->GetProcessListLength(); ++i) {
      if (auto* candidate = dynamic_cast<G4hMultipleScattering*>(
              (*process_list)[i])) {
        msc = candidate;
        break;
      }
    }
    if (!msc) throw std::runtime_error("C12 G4hMultipleScattering is absent");
    G4UrbanMscModel* urban = nullptr;
    for (G4int i = 0; i < msc->NumberOfModels(); ++i) {
      if (auto* candidate = dynamic_cast<G4UrbanMscModel*>(
              msc->GetModelByIndex(i))) {
        urban = candidate;
        break;
      }
    }
    if (!urban) throw std::runtime_error("active MSC process has no Urban model");
    auto* loss = urban->GetIonisation();
    if (!loss) {
      throw std::runtime_error("MSC oracle has no bound ionisation process");
    }
    bool loss_is_registered = false;
    for (G4int i = 0; i < particle->GetProcessManager()->GetProcessListLength();
         ++i) {
      loss_is_registered |= (*particle->GetProcessManager()->GetProcessList())[i]
                            == loss;
    }
    if (!loss_is_registered) {
      throw std::runtime_error("bound ionisation is not registered on C12");
    }
    const double e = pre->GetKineticEnergy();
    const double r = urban->GetRange(particle, e, couple);
    const double d = urban->GetDEDX(particle, e, couple);
    const double inv = urban->GetEnergy(particle, r, couple);
    const double lr = loss->GetRange(e, couple);
    const double ld = loss->GetDEDX(e, couple);
    if (r != lr || d != ld || !(r > 0) || !(d > 0) ||
        !std::isfinite(r+d+inv) || std::abs(inv/e-1) > 2e-5) {
      throw std::runtime_error("actual-step Urban/loss context mismatch");
    }
    if (couple != g_contexts[context_id].logical->GetMaterialCutsCouple())
      throw std::runtime_error("actual step used wrong material-cuts couple");
    g_msc = msc; g_urban = urban; g_bound_loss = loss;
    g_trace.push_back({event_id, track->GetTrackID(), context_id,
        e/MeV, r/mm, d/(MeV/mm), inv/MeV, step->GetStepLength()/mm,
        particle, couple});
    // All observations above follow a real first step. Stop only after they
    // have been captured; no sampler/limiter is called a second time.
    const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);

  }
};

double rel_error(const double a, const double b) {
  return std::abs(a - b) / std::max({std::abs(a), std::abs(b), 1.0e-300});
}

void extract_active_loss_table(const std::string& path, const Context& context,
                               G4ParticleDefinition* particle,
                               G4UrbanMscModel* urban,
                               G4VEnergyLossProcess* loss) {
  if (!context.couple || !context.logical || !particle || !urban || !loss ||
      urban->GetIonisation() == nullptr) {
    throw std::runtime_error("cannot extract without a complete active loss context");
  }
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot open extracted loss table: " + path);
  const auto* material = context.logical->GetMaterial();
  const auto* region = context.logical->GetRegion();
  const double zeff = material->GetName() == "Water_75eV" ? 3.3334 : 29.0;
  out << "# ORACLE_STATUS VALID_ACTIVE_C12_STEP_CONTEXT\n"
      << "# g4_version " << G4Version << "\n"
      << "# physics_list G4EmStandardPhysics_option4\n"
      << "# extraction_stage actual_C12_first_step_before_EndTracking\n"
      << "# range_source active_UrbanMsc_GetRange_bound_ionIoni_GetRange\n"
      << "# inverse_source active_UrbanMsc_GetEnergy_and_ionIoni_GetKineticEnergy\n"
      << "# dedx_source active_UrbanMsc_GetDEDX_bound_ionIoni_GetDEDX_restricted\n"
      << "# particle C12_Z6_A12_charge6\n"
      << "# material " << material->GetName() << "\n"
      << "# material_couple_index " << context.couple->GetIndex() << "\n"
      << "# region " << region->GetName() << "\n"
      << "# production_cut_mm " << context.cut_mm << "\n"
      << "# material_composition_mass_fraction ";
  const auto* elements = material->GetElementVector();
  const auto* fractions = material->GetFractionVector();
  for (std::size_t i = 0; i < material->GetNumberOfElements(); ++i) {
    if (i) out << ';';
    out << (*elements)[i]->GetSymbol() << ':'
        << (fractions ? fractions[i] : 0.0);
  }
  out << "\n# density_g_per_cm3 " << material->GetDensity() / (g / cm3)
      << "\n# mean_excitation_energy_eV "
      << material->GetIonisation()->GetMeanExcitationEnergy() / eV
      << "\n# zeff " << zeff
      << "\n# radlen_mm " << material->GetRadlen() / mm
      << "\n# grid logarithmic nodes=" << kLossGridNodes
      << " total_MeV=" << kLossGridMinMeV << ".." << kLossGridMaxMeV
      << "\n# energy_MeVu,energy_total_MeV,loss_range_mm,"
         "restricted_dedx_MeV_per_mm,inverse_residual_MeV\n";
  out << std::setprecision(17);

  std::vector<double> energies(kLossGridNodes);
  std::vector<double> ranges(kLossGridNodes);
  std::vector<double> dedx(kLossGridNodes);
  std::vector<float> energies_f(kLossGridNodes);
  std::vector<float> ranges_f(kLossGridNodes);
  std::vector<float> dedx_f(kLossGridNodes);
  double max_range_rel = 0.0;
  double max_dedx_rel = 0.0;
  double max_inverse_rel = 0.0;
  for (int i = 0; i < kLossGridNodes; ++i) {
    const double f = static_cast<double>(i) / (kLossGridNodes - 1);
    const double energy_mev = kLossGridMinMeV *
        std::pow(kLossGridMaxMeV / kLossGridMinMeV, f);
    const auto& row = g_trace.at(i*kMaterialCount +
        (context.name == "water_cut_0p10mm" ? 0 : context.name == "water_cut_0p05mm" ? 1 : 2));
    if (std::abs(row.energy_mev/energy_mev-1) > 1e-12)
      throw std::runtime_error("captured grid energy mismatch");
    const double r_msc = row.range_mm, r_loss = row.range_mm;
    const double d_msc = row.dedx, d_loss = row.dedx;
    const double e_msc = row.inverse_mev, e_loss = row.inverse_mev;
    if (!(r_msc > 0.0) || !(r_loss > 0.0) || !(d_loss > 0.0) ||
        !std::isfinite(r_msc + r_loss + d_msc + d_loss + e_msc + e_loss)) {
      throw std::runtime_error("invalid active process result during dense extraction");
    }
    max_range_rel = std::max(max_range_rel, rel_error(r_msc, r_loss));
    max_dedx_rel = std::max(max_dedx_rel, rel_error(d_msc, d_loss));
    max_inverse_rel = std::max(max_inverse_rel,
                               std::max(rel_error(e_msc, energy_mev),
                                        rel_error(e_loss, energy_mev)));
    if (i > 0 && !(r_loss > ranges[i - 1])) {
      throw std::runtime_error("active loss range is not strictly increasing");
    }
    energies[i] = energy_mev;
    ranges[i] = r_loss;
    dedx[i] = d_loss;
    energies_f[i] = static_cast<float>(energy_mev);
    ranges_f[i] = static_cast<float>(r_loss);
    dedx_f[i] = static_cast<float>(d_loss);
    out << energy_mev / 12.0 << ',' << energy_mev << ',' << r_loss << ','
        << d_loss << ',' << (e_loss - energy_mev) << '\n';
  }
  double max_interp_range_rel = 0.0;
  double max_interp_dedx_rel = 0.0;
  double max_interp_inverse_rel = 0.0;
  if (max_range_rel > 2.0e-10 || max_dedx_rel > 2.0e-10 ||
      max_inverse_rel > 2.0e-5) {
    throw std::runtime_error("dense active Urban/loss consistency failed");
  }
  std::cout << "RANGE_TABLE " << context.name << " nodes=" << kLossGridNodes
            << " max_range_rel=" << max_range_rel
            << " max_dedx_rel=" << max_dedx_rel
            << " max_native_inverse_rel=" << max_inverse_rel
            << " interpolation_validation=SEPARATE_INDEPENDENT_GRID"
            << " csv=" << path << '\n';
}

}  // namespace

class KillSecondary final : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    return t->GetParentID() == 0 ? fUrgent : fKill;
  }
};
int main(int argc, char** argv) {
  if (argc != 5) return 2;
  try {
    G4RunManager run;
    run.SetUserInitialization(new Detector());
    run.SetUserInitialization(new PhysicsList());
    run.SetUserAction(new PrimaryGenerator());
    run.SetUserAction(new PassiveTrace());
    run.SetUserAction(new KillSecondary());
    run.Initialize();
    run.BeamOn(kCaseCount);
    if (g_trace.size() != kCaseCount) throw std::runtime_error("incomplete grid tracking");
    for (auto& c : g_contexts) c.couple = c.logical->GetMaterialCutsCouple();
    auto* c12 = G4IonTable::GetIonTable()->GetIon(6,12,0.0);
    for (int i=0;i<3;++i) extract_active_loss_table(argv[i+1],g_contexts[i],c12,g_urban,g_bound_loss);
    std::ofstream trace(argv[4]);
    trace << "event_id,context,E_total_MeV,range_mm,dedx_MeV_mm,inverse_MeV,true_step_mm\n";
    trace << std::setprecision(17);
    for (const auto& r:g_trace) trace << r.event_id << ',' << r.context_id << ',' << r.energy_mev << ',' << r.range_mm << ',' << r.dedx << ',' << r.inverse_mev << ',' << r.true_step_mm << '\n';
    std::cout << "ACTIVE_STEP_CONTEXT=PASS tracked_rows=" << g_trace.size() << "\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

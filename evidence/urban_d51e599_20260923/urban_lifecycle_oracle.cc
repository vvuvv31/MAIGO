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
constexpr double kLossGridMinMeV = 0.12;
constexpr double kLossGridMaxMeV = 4800.12;
constexpr int kMaterialCount = 3;
constexpr int kCaseCount = 9 * kMaterialCount;
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
  PhysicsList() { RegisterPhysics(new G4EmStandardPhysics_option4(0)); }
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
    gun_.SetParticleEnergy(kEnergyMeV[energy_index] * MeV);
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
    const auto duplicate = std::find_if(
        g_trace.begin(), g_trace.end(), [&](const TraceRow& row) {
          return row.event_id == event_id;
        });
    if (duplicate != g_trace.end()) return;

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
    g_msc = msc;
    g_urban = urban;
    g_bound_loss = loss;
    g_trace.push_back({event_id, track->GetTrackID(), context_id,
                       pre->GetKineticEnergy() / MeV, particle, couple});
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
  out << "# ORACLE_STATUS VALID_ACTIVE_C12_PROCESS_TRACKING\n"
      << "# g4_version " << G4Version << "\n"
      << "# physics_list G4EmStandardPhysics_option4\n"
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
    const double r_msc = urban->GetRange(particle, energy_mev * MeV,
                                         context.couple) / mm;
    const double r_loss = loss->GetRange(energy_mev * MeV, context.couple) / mm;
    const double d_msc = urban->GetDEDX(particle, energy_mev * MeV,
                                        context.couple) / (MeV / mm);
    const double d_loss = loss->GetDEDX(energy_mev * MeV, context.couple) /
                          (MeV / mm);
    const double e_msc = urban->GetEnergy(particle, r_msc * mm,
                                          context.couple) / MeV;
    const double e_loss = loss->GetKineticEnergy(r_loss * mm,
                                                 context.couple) / MeV;
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
  for (int i = 0; i + 1 < kLossGridNodes; ++i) {
    const double e_mid = 0.5 * (energies[i] + energies[i + 1]);
    const float ef = static_cast<float>(e_mid);
    const float f_e = (ef - energies_f[i]) /
                      (energies_f[i + 1] - energies_f[i]);
    const float r_interp = ranges_f[i] + f_e * (ranges_f[i + 1] - ranges_f[i]);
    const float d_interp = dedx_f[i] + f_e * (dedx_f[i + 1] - dedx_f[i]);
    const double r_ref = loss->GetRange(e_mid * MeV, context.couple) / mm;
    const double d_ref = loss->GetDEDX(e_mid * MeV, context.couple) /
                         (MeV / mm);
    max_interp_range_rel = std::max(max_interp_range_rel,
                                    rel_error(r_interp, r_ref));
    max_interp_dedx_rel = std::max(max_interp_dedx_rel,
                                   rel_error(d_interp, d_ref));
    const float r_mid = 0.5F * (ranges_f[i] + ranges_f[i + 1]);
    const float f_r = (r_mid - ranges_f[i]) / (ranges_f[i + 1] - ranges_f[i]);
    const float e_interp = energies_f[i] + f_r *
                           (energies_f[i + 1] - energies_f[i]);
    const double e_ref = loss->GetKineticEnergy(
        static_cast<double>(r_mid) * mm, context.couple) / MeV;
    max_interp_inverse_rel = std::max(max_interp_inverse_rel,
                                      rel_error(e_interp, e_ref));
  }
  if (max_range_rel > 2.0e-10 || max_dedx_rel > 2.0e-10 ||
      max_inverse_rel > 2.0e-5) {
    throw std::runtime_error("dense active Urban/loss consistency failed");
  }
  std::cout << "RANGE_TABLE " << context.name << " nodes=" << kLossGridNodes
            << " max_range_rel=" << max_range_rel
            << " max_dedx_rel=" << max_dedx_rel
            << " max_native_inverse_rel=" << max_inverse_rel
            << " max_linear_range_rel=" << max_interp_range_rel
            << " max_linear_dedx_rel=" << max_interp_dedx_rel
            << " max_linear_inverse_rel=" << max_interp_inverse_rel
            << " csv=" << path << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: urban_lifecycle_oracle <positive.csv> <negative.csv> "
                 "<water-0.10.csv> <water-0.05.csv> <copper-0.05.csv>\n";
    return 2;
  }
  try {
    G4RunManager run;
    run.SetUserInitialization(new Detector());
    run.SetUserInitialization(new PhysicsList());
    run.SetUserAction(new PrimaryGenerator());
    run.SetUserAction(new PassiveTrace());
    run.Initialize();
    run.BeamOn(kCaseCount);

    if (g_trace.size() != static_cast<std::size_t>(kCaseCount) || !g_urban ||
        !g_msc || !g_bound_loss) {
      throw std::runtime_error("incomplete positive control tracking coverage");
    }
    if (g_urban->GetIonisation() == nullptr) {
      throw std::runtime_error("MSC oracle has no bound ionisation process");
    }
    if (g_urban->GetIonisation() != g_bound_loss) {
      throw std::runtime_error("active Urban/loss binding changed after tracking");
    }

    for (auto& context : g_contexts) {
      context.couple = context.logical->GetMaterialCutsCouple();
      if (!context.couple || context.couple->GetIndex() <= 0 ||
          !context.couple->GetProductionCuts()) {
        throw std::runtime_error("material-cuts couple lacks registered cuts");
      }
      for (G4int cut_index = 0; cut_index < 4; ++cut_index) {
        const double actual_cut = context.couple->GetProductionCuts()
                                      ->GetProductionCut(cut_index) / mm;
        if (std::abs(actual_cut - context.cut_mm) > 1.0e-12) {
          throw std::runtime_error("material-cuts couple has unexpected cut value");
        }
      }
    }
    if (!g_bound_loss->TablesAreBuilt()) {
      throw std::runtime_error("active C12 loss physics tables are not built");
    }

    std::ofstream positive(argv[1]);
    if (!positive) throw std::runtime_error("cannot open positive output");
    positive << "g4_version,physics_list,msc_process,urban_model,loss_process,"
                "particle,Z,A,charge_e,mass_MeV,context,couple_index,region,"
                "cut_mm,material,composition_mass_fraction,density_g_cm3,mean_excitation_eV,"
                "first_step_material,first_step_couple_index,"
                "E_total_MeV,R_msc_mm,R_loss_mm,DEDX_msc_MeV_mm,"
                "DEDX_loss_MeV_mm,E_from_Rmsc_MeV,E_from_Rloss_MeV,"
                "range_rel_error,dedx_rel_error,inverse_rel_error,"
                "ionisation_bound,tables_built\n";
    positive << std::setprecision(17);

    std::ofstream negative(argv[2]);
    if (!negative) throw std::runtime_error("cannot open negative output");
    negative << "control,ionisation_bound,particle,material,E_total_MeV,"
                "fallback_range_mm,expected_linear_range_mm,rel_error\n";
    negative << std::setprecision(17);

    G4UrbanMscModel unbound("UrbanFallbackNegativeControl");
    auto* c12 = G4IonTable::GetIonTable()->GetIon(6, 12, 0.0);
    unbound.InitialiseParameters(c12);
    if (unbound.GetIonisation() != nullptr) {
      throw std::runtime_error("negative control unexpectedly bound ionisation");
    }

    std::set<int> seen_contexts;
    for (const auto& trace : g_trace) {
      const auto context_id = trace.context_id;
      const auto& context = g_contexts.at(context_id);
      const auto* couple = context.couple;
      auto* material = context.logical->GetMaterial();
      if (trace.particle != c12 || couple->GetMaterial() != material ||
          !trace.couple) {
        throw std::runtime_error("tracked C12/couple identity mismatch");
      }
      if (trace.event_id / kMaterialCount == 0 && trace.couple != couple) {
        throw std::runtime_error("low-energy primary did not step in its registered material-cuts couple");
      }
      if (g_urban->GetIonisation() == nullptr) {
        throw std::runtime_error("MSC oracle has no bound ionisation process");
      }
      const double E = kEnergyMeV[trace.event_id / kMaterialCount] * MeV;
      const double r_msc = g_urban->GetRange(c12, E, couple);
      const double r_loss = g_bound_loss->GetRange(E, couple);
      const double dedx_msc = g_urban->GetDEDX(c12, E, couple);
      const double dedx_loss = g_bound_loss->GetDEDX(E, couple);
      const double e_msc = g_urban->GetEnergy(c12, r_msc, couple);
      const double e_loss = g_bound_loss->GetKineticEnergy(r_loss, couple);
      if (!(r_msc > 0.0) || !(r_loss > 0.0) || !(dedx_loss > 0.0) ||
          !std::isfinite(r_msc + r_loss + dedx_msc + dedx_loss + e_msc + e_loss)) {
        throw std::runtime_error("nonfinite/nonpositive active loss lookup");
      }
      const double range_err = rel_error(r_msc, r_loss);
      const double dedx_err = rel_error(dedx_msc, dedx_loss);
      const double inverse_err = std::max(rel_error(e_msc, E),
                                          rel_error(e_loss, E));
      if (range_err > 2.0e-10 || dedx_err > 2.0e-10 || inverse_err > 2.0e-5) {
        throw std::runtime_error("active Urban and loss tables failed consistency");
      }
      const auto* ion = material->GetIonisation();
      const auto* region = context.logical->GetRegion();
      std::ostringstream composition;
      composition << std::setprecision(8);
      const auto* elements = material->GetElementVector();
      const auto* fractions = material->GetFractionVector();
      for (G4int i = 0; i < material->GetNumberOfElements(); ++i) {
        if (i) composition << ';';
        composition << (*elements)[i]->GetSymbol() << ':'
                    << (fractions ? fractions[i] : 0.0);
      }
      positive << G4Version << ",G4EmStandardPhysics_option4," << g_msc->GetProcessName()
               << ',' << g_urban->GetName() << ',' << g_bound_loss->GetProcessName()
               << ',' << c12->GetParticleName() << ',' << c12->GetAtomicNumber()
               << ',' << c12->GetAtomicMass() << ',' << c12->GetPDGCharge() / eplus
               << ',' << c12->GetPDGMass() / MeV << ',' << context.name << ','
               << couple->GetIndex() << ',' << region->GetName() << ','
               << context.cut_mm << ',' << material->GetName() << ','
               << composition.str() << ',' << material->GetDensity() / (g / cm3)
               << ',' << ion->GetMeanExcitationEnergy() / eV << ','
               << trace.couple->GetMaterial()->GetName() << ','
               << trace.couple->GetIndex() << ','
               << E / MeV << ','
               << r_msc / mm << ',' << r_loss / mm << ',' << dedx_msc / (MeV / mm)
               << ',' << dedx_loss / (MeV / mm) << ',' << e_msc / MeV << ','
               << e_loss / MeV << ',' << range_err << ',' << dedx_err << ','
               << inverse_err << ",true,true\n";
      seen_contexts.insert(context_id);

      unbound.SetCurrentCouple(couple);
      const double fallback_range = unbound.GetRange(c12, E, couple);
      const double q = c12->GetPDGCharge() / eplus;
      const double stopping_per_length =
          2.0 * MeV * cm2 / g * q * q * material->GetDensity();
      const double expected_range = E / stopping_per_length;
      const double fallback_err = rel_error(fallback_range, expected_range);
      if (!(fallback_range > 0.0) || fallback_err > 2.0e-6) {
        throw std::runtime_error("unbound-model linear fallback mismatch");
      }
      negative << "unbound_G4VMscModel_fallback,false," << c12->GetParticleName()
               << ',' << material->GetName() << ',' << E / MeV << ','
               << fallback_range / mm << ',' << expected_range / mm << ','
               << fallback_err << '\n';
    }
    if (seen_contexts.size() != kMaterialCount) {
      throw std::runtime_error("not all real material-cuts contexts were tracked");
    }
    extract_active_loss_table(argv[3], g_contexts[0], c12, g_urban, g_bound_loss);
    extract_active_loss_table(argv[4], g_contexts[1], c12, g_urban, g_bound_loss);
    extract_active_loss_table(argv[5], g_contexts[2], c12, g_urban, g_bound_loss);
    std::cout << "ORACLE_LIFECYCLE=PASS; tracked_primary_rows=" << g_trace.size()
              << "; null_ionisation=0; active_model=" << g_urban->GetName()
              << "; active_msc=" << g_msc->GetProcessName()
              << "; active_loss=" << g_bound_loss->GetProcessName() << '\n';
    std::cout << "positive_csv=" << argv[1] << " negative_csv=" << argv[2] << '\n';
  } catch (const std::exception& e) {
    std::cerr << "ORACLE_LIFECYCLE=FAIL: " << e.what() << '\n';
    return 1;
  }
  return 0;
}

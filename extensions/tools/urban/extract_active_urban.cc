// Geant4 11.3.2 Urban tables observed inside real first steps.
// One process per projectile; every energy/material has an independent track.
// This is an EM reference extraction, not a TOPAS dose rerun.
// Read-only process provenance, matching the unified EM exporter. Access
// labels change neither layout nor physics; never write process internals.
#include <bits/stdc++.h>
#define private public
#define protected public
#include "G4VEnergyLossProcess.hh"
#undef protected
#undef private
#include "G4Box.hh"
#include "G4EmParameters.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4IonTable.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicsListHelper.hh"
#include "G4PhysicsTable.hh"
#include "G4PhysicsVector.hh"
#include "G4ProcessManager.hh"
#include "G4ProductionCuts.hh"
#include "G4Region.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UrbanMscModel.hh"
#include "G4UserLimits.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4Version.hh"
#include "G4hMultipleScattering.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
#ifdef MAIGO_URBAN_COPPER_ONLY
constexpr int material_count = 1;
#else
constexpr int material_count = 5;
#endif
constexpr double cut_mm = 0.05;
constexpr double emin_mev = 0.05;
constexpr double emax_mev = 6500.0;
int projectile_z = 6, projectile_a = 12, node_count = 32769;
double grid_offset = 0.;
std::filesystem::path output;
struct MaterialSpec {
  const char* name;
  int section;
  double density, excitation;
  std::vector<int> z;
  std::vector<double> fractions;
};
// M0/M1/M8/M20 are the unchanged benchmark20260912/materials.txt inputs
// referenced by the reused B3/B4 TOPAS runs. Water uses atomic stoichiometry.
const std::array<MaterialSpec, material_count> specs{{
#ifdef MAIGO_URBAN_COPPER_ONLY
  {"G4_Cu", -2, 8.96, 322., {29}, {1.}}
#else
  {"Water_75eV", -1, 1., 75., {}, {}},
  {"M0", 0, .01131606475, 85.6829, {7,8,18}, {.755,.232,.013}},
  {"M1", 1, .4700411856, 69.693, {1,6,7,8,15,16,17,11,19},
      {.103,.105,.031,.749,.002,.003,.003,.002,.002}},
  {"M8", 8, 1.078799725, 70.0821, {1,6,7,8,16,17,11},
      {.094,.207,.062,.622,.006,.003,.006}},
  {"M20", 20, 1.821604848, 97.4852, {1,6,7,8,12,15,16,20,11},
      {.042,.194,.04,.425,.002,.092,.003,.201,.001}}
#endif
}};
std::array<G4LogicalVolume*, material_count> logicals{};
struct Row {
  double energy{},range{},dedx{},lambda{},inverse{},mass{},factor{},ratio{},minimum{};
  double query_range{},query_dedx{},query_energy{},query_lambda{};
  double mfp_energy{},mfp_value{};
  bool seen{};
};
std::vector<Row> rows;
double energy_at(int i) {
  return emin_mev*std::pow(emax_mev/emin_mev,
      (i+grid_offset)/(node_count-1));
}
double centre(int i) { return (i-2)*120.; }

class Detector final : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist=G4NistManager::Instance();
    auto* world_lv=new G4LogicalVolume(new G4Box("world",600*mm,100*mm,100*mm),
        nist->FindOrBuildMaterial("G4_Galactic"),"world");
    auto* world=new G4PVPlacement(nullptr,{},world_lv,"world",nullptr,false,0);
    for(int i=0;i<material_count;++i) {
      const auto& s=specs[i];
#ifdef MAIGO_URBAN_COPPER_ONLY
      auto* material=nist->FindOrBuildMaterial("G4_Cu");
#else
      auto* material=new G4Material(s.name,s.density*g/cm3,
          i==0?2:static_cast<int>(s.z.size()));
      if(i==0) {
        auto* h=new G4Element("WaterHydrogen75","Hw75",1.,1.00794*g/mole);
        auto* o=new G4Element("WaterOxygen75","Ow75",8.,15.9994*g/mole);
        material->AddElement(h,2); material->AddElement(o,1);
      } else {
        for(std::size_t j=0;j<s.z.size();++j)
          material->AddElement(nist->FindOrBuildElement(s.z[j]),s.fractions[j]);
      }
      material->GetIonisation()->SetMeanExcitationEnergy(s.excitation*eV);

#endif
      auto* lv=new G4LogicalVolume(new G4Box(s.name,40*mm,40*mm,40*mm),material,s.name);
      lv->SetUserLimits(new G4UserLimits(.05*mm));
      new G4PVPlacement(nullptr,{centre(i)*mm,0,0},lv,s.name,world_lv,false,i);
      auto* region=new G4Region(std::string(s.name)+"_region");
      region->AddRootLogicalVolume(lv);
      auto* cuts=new G4ProductionCuts(); cuts->SetProductionCut(cut_mm*mm);
      region->SetProductionCuts(cuts); logicals[i]=lv;
    }
    return world;
  }
};

class Physics final : public G4VModularPhysicsList {
 public:
  Physics() {
    RegisterPhysics(new G4EmStandardPhysics_option4(0));
    RegisterPhysics(new G4StepLimiterPhysics());
    G4EmParameters::Instance()->SetMaxEnergy(10*GeV);
    SetDefaultCutValue(cut_mm*mm);
  }
  void ConstructProcess() override {
    G4VModularPhysicsList::ConstructProcess();
    // SetEmModel appends, so it cannot replace opt4's proton/light-ion model.
    // Replace each distinct hadron MSC process before table initialization.
    auto* iterator=GetParticleIterator(); iterator->reset();
    std::set<G4ProcessManager*> visited;
    while((*iterator)()) {
      auto* particle=iterator->value();
      auto* pm=particle->GetProcessManager();
      if(!pm || !visited.insert(pm).second) continue;
      auto* processes=pm->GetProcessList();
      G4hMultipleScattering* old=nullptr;
      for(int j=0;j<pm->GetProcessListLength();++j)
        if(auto* m=dynamic_cast<G4hMultipleScattering*>((*processes)[j])) { old=m; break; }
      if(!old) continue;
      auto name=old->GetProcessName(); pm->RemoveProcess(old);
      auto* replacement=new G4hMultipleScattering(name);
      auto* model=new G4UrbanMscModel(); model->SetHighEnergyLimit(10*GeV);
      replacement->SetEmModel(model);
      if(!G4PhysicsListHelper::GetPhysicsListHelper()->RegisterProcess(replacement,particle))
        throw std::runtime_error("failed to register all-ion Urban process");
    }
  }
};

class Generator final : public G4VUserPrimaryGeneratorAction {
  G4ParticleGun gun{1};
 public:
  void GeneratePrimaries(G4Event* event) override {
    const int id=event->GetEventID(), m=id%material_count, i=id/material_count;
    auto* particle=G4IonTable::GetIonTable()->GetIon(projectile_z,projectile_a,0.);
    if(!particle || particle->GetAtomicNumber()!=projectile_z ||
       particle->GetAtomicMass()!=projectile_a) throw std::runtime_error("invalid projectile identity");
    gun.SetParticleDefinition(particle); gun.SetParticleEnergy(energy_at(i)*MeV);
    gun.SetParticlePosition({centre(m)*mm,0,0});
    gun.SetParticleMomentumDirection({0,0,1}); gun.GeneratePrimaryVertex(event);
  }
};

class Capture final : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    auto* track=step->GetTrack();
    if(track->GetParentID()!=0 || track->GetCurrentStepNumber()!=1) return;
    const int id=G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
    auto* couple=step->GetPreStepPoint()->GetMaterialCutsCouple();
    if(id<0 || static_cast<std::size_t>(id)>=rows.size() || rows[id].seen ||
       couple!=logicals[id%material_count]->GetMaterialCutsCouple())
      throw std::runtime_error("wrong or duplicate active material/track context");
    auto* particle=track->GetDefinition(); auto* pm=particle->GetProcessManager();
    G4hMultipleScattering* process=nullptr;
    for(int j=0;j<pm->GetProcessListLength();++j)
      if(auto* m=dynamic_cast<G4hMultipleScattering*>((*pm->GetProcessList())[j])) { process=m; break; }
    auto* model=process && process->NumberOfModels()==1
        ? dynamic_cast<G4UrbanMscModel*>(process->GetModelByIndex(0)):nullptr;
    if(!model || !model->GetIonisation()) throw std::runtime_error("active model is not bound Urban");
    auto* loss=model->GetIonisation();
    bool registered=false;
    for(int j=0;j<pm->GetProcessListLength();++j) registered|=(*pm->GetProcessList())[j]==loss;
    if(!registered) throw std::runtime_error("bound loss process is not registered on projectile");
    const double e=step->GetPreStepPoint()->GetKineticEnergy();
    const double range=model->GetRange(particle,e,couple);
    const double dedx=model->GetDEDX(particle,e,couple);
    const double inverse=model->GetEnergy(particle,range,couple);
    const double factor=loss->fFactor, ratio=loss->massRatio;
    if(!(factor>0 && ratio>0)) throw std::runtime_error("invalid active mass/charge factor");
    model->SetCurrentCouple(couple);
    const double lambda=model->GetTransportMeanFreePath(particle,e);
    if(range!=loss->GetRange(e,couple) || dedx!=loss->GetDEDX(e,couple) ||
       !(range>0 && dedx>0 && lambda>0) || !std::isfinite(range+dedx+lambda+inverse) ||
       std::abs(inverse/e-1)>5e-5 || std::abs(e/MeV/energy_at(id/material_count)-1)>1e-10)
      throw std::runtime_error("active Urban/loss range, inverse, or lambda validation failed");
    // Queries inside ONE step keep the current dynamic factor. In general,
    // the range from separate tracks is not monotone in energy for heavy ions.
    // Export native spline vectors instead of attempting to invert that curve.
    const double query_range=model->GetRange(particle,.37*e,couple);
    const double query_dedx=model->GetDEDX(particle,.37*e,couple);
    const double query_energy=model->GetEnergy(particle,.37*range,couple);
    const double query_lambda=model->GetTransportMeanFreePath(particle,.37*e);
    // Internal Urban energy prediction can be far below the transport cutoff.
    // Query its MFP down to the native 1 eV scattering gate in the same model.
    const double mfp_energy=1e-6*std::pow(emax_mev/1e-6,
        (id/material_count+grid_offset)/(node_count-1));
    const double mfp_value=model->GetTransportMeanFreePath(particle,mfp_energy*MeV)/mm;
    if(!(mfp_value>0) || !std::isfinite(mfp_value))
      throw std::runtime_error("invalid low-energy MFP query");
    auto* rv=(*loss->RangeTableForLoss())[loss->basedCoupleIndex];
    auto* dv=(*loss->DEDXTable())[loss->basedCoupleIndex];
    double raw_r=rv->Value(e*ratio)/(factor*ratio);
    double raw_d=dv->Value(e*ratio)*factor;
    if(e*ratio<loss->minKinEnergy) {
      const double low=std::sqrt(e*ratio/loss->minKinEnergy);
      raw_r*=low;raw_d*=low;
    }
    if(std::abs(raw_r/range-1)>1e-9 || std::abs(raw_d/dedx-1)>1e-9 ||
       loss->fFactor!=factor || loss->massRatio!=ratio)
      throw std::runtime_error("raw spline reconstruction or fixed-step factor failed");
    if(id<material_count) {
      // The equivalent electron energy 10 MeV branch is discontinuous for
      // ions. Preserve both sides at adjacent FP32 energies, without smoothing.
      const double me=CLHEP::electron_mass_c2/MeV, mass=particle->GetPDGMass()/MeV;
      const double tau=10./me,c=me*tau*(tau+2)/(tau+1),b=2*mass-c;
      const double edge=2*c*mass/(std::sqrt(b*b+4*c*mass)+b);
      float lo=static_cast<float>(edge);
      if(static_cast<double>(lo)>=edge)lo=std::nextafter(lo,0.f);
      const float hi=std::nextafter(lo,std::numeric_limits<float>::infinity());
      std::ofstream edges(output/(std::string(specs[id].name)+"_mfp_edges.csv"));
      edges<<std::setprecision(17)<<"energy_mev,lambda_mm\n";
      for(float point:{lo,hi})
        edges<<point<<','<<model->GetTransportMeanFreePath(particle,point*MeV)/mm<<'\n';
      edges.close();if(!edges)throw std::runtime_error("MFP branch-edge write failed");
      std::ofstream raw(output/(std::string(specs[id].name)+"_vectors.csv"));
      raw<<std::setprecision(17)<<"kind,index,x0,x1,y0,y_third,y_twothirds,y1,spline\n";
      int kind=0;
      for(auto* table:{loss->DEDXTable(),loss->RangeTableForLoss(),loss->InverseRangeTable()}) {
        auto* vector=(*table)[loss->basedCoupleIndex];
        for(std::size_t j=0;j+1<vector->GetVectorLength();++j) {
          const double x=vector->Energy(j),h=vector->Energy(j+1)-x;
          raw<<kind<<','<<j<<','<<x<<','<<x+h<<','<<vector->Value(x)<<','
             <<vector->Value(x+h/3)<<','<<vector->Value(x+2*h/3)<<','
             <<vector->Value(x+h)<<','<<vector->GetSpline()<<'\n';
        }
        ++kind;
      }
      raw.close();if(!raw)throw std::runtime_error("native spline write failed");
    }
    rows[id]={e/MeV,range/mm,dedx/(MeV/mm),lambda/mm,inverse/MeV,
        particle->GetPDGMass()/MeV,factor,ratio,loss->minKinEnergy/MeV,
        query_range/mm,query_dedx/(MeV/mm),query_energy/MeV,query_lambda/mm,
        mfp_energy,mfp_value,true};
    // Query only after a real first step; never sample MSC again for extraction.
    const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);
  }
};
class KillSecondaries final : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    return t->GetParentID()==0?fUrgent:fKill;
  }
};

void write_tables() {
  for(int m=0;m<material_count;++m) {
    const auto& s=specs[m]; auto* material=logicals[m]->GetMaterial();
    std::ofstream out(output/(std::string(s.name)+".csv"));
    if(!out) throw std::runtime_error("cannot create output table");
    out<<std::setprecision(17)
       <<"# ORACLE_STATUS VALID_ACTIVE_ION_STEP_CONTEXT\n"
       <<"# g4_version "<<G4Version<<"\n"
       <<"# physics G4EmStandardPhysics_option4_all_ion_Urban\n"
       <<"# extraction_stage actual_first_step_before_EndTracking\n"
       <<"# loss_context native_splines_fixed_step_factor_v2\n"
       <<"# particle_z "<<projectile_z<<"\n# particle_a "<<projectile_a
       <<"\n# particle_mass_mev "<<rows[m].mass
       <<"\n# material "<<s.name<<"\n# section "<<s.section
       <<"\n# density_g_cm3 "<<material->GetDensity()/(g/cm3)
       <<"\n# zeff "<<material->GetIonisation()->GetZeffective()
       <<"\n# radlen_mm "<<material->GetRadlen()/mm
       <<"\n# excitation_ev "<<material->GetIonisation()->GetMeanExcitationEnergy()/eV
       <<"\n# production_cut_mm "<<cut_mm
       <<"\n# grid_offset "<<grid_offset
       <<"\n# mass_ratio "<<rows[m].ratio
       <<"\n# minimum_scaled_energy_mev "<<rows[m].minimum
       <<"\nenergy_mev,range_mm,dedx_mev_mm,lambda_mm,inverse_mev,factor,query_range_mm,query_dedx_mev_mm,query_energy_mev,query_lambda_mm\n";
    double previous=0.;
    std::ofstream mfp(output/(std::string(s.name)+"_mfp.csv"));
    mfp<<std::setprecision(17)<<"energy_mev,lambda_mm\n";
    for(int i=0;i<node_count;++i) {
      const auto& v=rows[i*material_count+m];
      if(!v.seen || !(v.range*v.factor>previous) || v.ratio!=rows[m].ratio ||
         v.minimum!=rows[m].minimum)
        throw std::runtime_error("missing row or nonmonotonic unscaled native range");
      previous=v.range*v.factor;
      out<<v.energy<<','<<v.range<<','<<v.dedx<<','<<v.lambda<<','<<v.inverse<<','
         <<v.factor<<','<<v.query_range<<','<<v.query_dedx<<','<<v.query_energy<<','
         <<v.query_lambda<<'\n';
      mfp<<v.mfp_energy<<','<<v.mfp_value<<'\n';
    }
    out.close();mfp.close();
    if(!out || !mfp)throw std::runtime_error("reference table write failed");
  }
}
} // namespace

int main(int argc,char** argv) {
  if(argc<5 || argc>6) { std::cerr<<"usage: extract_active_urban output_dir Z A nodes [grid_offset]\n";return 2; }
  try {
    output=argv[1];projectile_z=std::stoi(argv[2]);projectile_a=std::stoi(argv[3]);
    node_count=std::stoi(argv[4]);if(argc==6)grid_offset=std::stod(argv[5]);
    if(node_count<3 || projectile_z<1 || projectile_a<projectile_z || grid_offset<0 || grid_offset>=1)
      throw std::runtime_error("invalid extraction arguments");
    if(std::filesystem::exists(output)) throw std::runtime_error("output already exists; refusing overwrite");
    std::filesystem::create_directories(output);rows.resize(node_count*material_count);
    G4RunManager run;run.SetUserInitialization(new Detector());
    run.SetUserInitialization(new Physics());run.SetUserAction(new Generator());
    run.SetUserAction(new Capture());run.SetUserAction(new KillSecondaries());
    run.Initialize();run.BeamOn(node_count*material_count);write_tables();
    std::cout<<"ACTIVE_URBAN_ION_CONTEXT=PASS Z="<<projectile_z<<" A="<<projectile_a
             <<" rows="<<rows.size()<<'\n';
  } catch(const std::exception& e) { std::cerr<<"ACTIVE_URBAN_ION_CONTEXT=FAIL "<<e.what()<<'\n';return 1; }
}

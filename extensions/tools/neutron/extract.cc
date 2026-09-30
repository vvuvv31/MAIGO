// Conditional native neutron process extraction. No modified cross sections.
#include "G4RunManager.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserStackingAction.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4Isotope.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4IonisParamMat.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4HadronPhysicsQGSP_BIC_HP.hh"
#include "G4HadronElasticPhysicsHP.hh"
#include "G4IonINCLXXPhysics.hh"
#include "G4DecayPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronicProcess.hh"
#include "G4HadronicInteraction.hh"
#include "G4CrossSectionDataStore.hh"
#include "G4ParticleChange.hh"
#include "G4ParticleGun.hh"
#include "G4Neutron.hh"
#include "G4ProcessManager.hh"
#include "G4Event.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Version.hh"
#include "Randomize.hh"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <set>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>

namespace {
std::filesystem::path output;
std::string material_name;
std::vector<double> energies;
int samples=128;
int rate_repeats=1;
long random_seed=92717001;
G4Material* material=nullptr;
std::ofstream rates,events,products,provenance;
std::uint64_t event_id=0,product_id=0;
bool extracted=false;
class Detector final:public G4VUserDetectorConstruction {
 G4VPhysicalVolume* Construct() override {
  auto* nist=G4NistManager::Instance();
  if(material_name=="Water_75eV") {
   material=new G4Material("Water_75eV",1.*g/cm3,2,kStateLiquid,293.15*kelvin);
   material->AddElement(nist->FindOrBuildElement("H"),2);
   material->AddElement(nist->FindOrBuildElement("O"),1);
   material->GetIonisation()->SetMeanExcitationEnergy(75*eV);
  } else material=nist->FindOrBuildMaterial(material_name);
  if(!material)throw std::runtime_error("unknown material");
  auto* lv=new G4LogicalVolume(new G4Box("world",1*m,1*m,1*m),material,"world");
  return new G4PVPlacement(nullptr,{},lv,"world",nullptr,false,0);
 }
};
class Physics final:public G4VModularPhysicsList {
 public:Physics() {
  RegisterPhysics(new G4EmStandardPhysics_option4(0));
  RegisterPhysics(new G4HadronPhysicsQGSP_BIC_HP(0));
  RegisterPhysics(new G4HadronElasticPhysicsHP(0));
  RegisterPhysics(new G4IonINCLXXPhysics(0));
  RegisterPhysics(new G4DecayPhysics(0));
  RegisterPhysics(new G4StoppingPhysics(0));
  SetDefaultCutValue(.05*mm);
 }
};
class Generator final:public G4VUserPrimaryGeneratorAction {
 G4ParticleGun gun{1};
 void GeneratePrimaries(G4Event* event) override {
  gun.SetParticleDefinition(G4Neutron::Definition());gun.SetParticleEnergy(1*MeV);
  gun.SetParticlePosition({0,0,0});gun.SetParticleMomentumDirection({0,0,1});gun.GeneratePrimaryVertex(event);
 }
};
void finite(double x) {if(!std::isfinite(x))throw std::runtime_error("nonfinite native value");}
class Capture final:public G4UserSteppingAction {
 void UserSteppingAction(const G4Step* real) override {
  auto* original=real->GetTrack();
  if(extracted || original->GetParentID()!=0)return;
  extracted=true;
  auto* pm=G4Neutron::Definition()->GetProcessManager();
  std::vector<G4HadronicProcess*> processes;
  for(int i=0;i<pm->GetProcessListLength();++i) {
   auto* p=dynamic_cast<G4HadronicProcess*>((*pm->GetProcessList())[i]);if(p)processes.push_back(p);
  }
  std::set<std::string> names;
  for(auto* p:processes) {
   names.insert(p->GetProcessName());provenance<<"process "<<p->GetProcessName()<<" subtype "<<p->GetProcessSubType()<<'\n';
   for(auto* model:p->GetHadronicInteractionList())provenance<<"model "<<model->GetModelName()<<" energy_MeV "<<model->GetMinEnergy()/MeV<<' '<<model->GetMaxEnergy()/MeV<<'\n';
  }
  if(!names.count("hadElastic") || !names.count("neutronInelastic") || !names.count("nCapture"))throw std::runtime_error("incomplete neutron process list");
  for(std::size_t node=0;node<energies.size();++node) {
   double energy=energies[node]*MeV;
   for(auto* p:processes) {
    G4Track track(new G4DynamicParticle(G4Neutron::Definition(),{0,0,1},energy),0,{0,0,0});
    track.SetTouchableHandle(real->GetPreStepPoint()->GetTouchableHandle());track.SetTrackID(1);track.SetWeight(1.);
    G4Step step;track.SetStep(&step);step.InitializeStep(&track);
    step.GetPreStepPoint()->SetMaterial(material);step.GetPreStepPoint()->SetMaterialCutsCouple(real->GetPreStepPoint()->GetMaterialCutsCouple());
    *step.GetPostStepPoint()=*step.GetPreStepPoint();
    double rate=0,rate_m2=0,rate_min=1e300,rate_max=0;
    for(int repeat=0;repeat<rate_repeats;++repeat) {
     // HP data cache by exact kinetic energy/material/element. Evict with a
     // different energy, then query the unchanged target energy. The evicting
     // query is discarded; it is not an interpolation or a physics correction.
     if(repeat>0) {
      G4DynamicParticle evict(G4Neutron::Definition(),{0,0,1},energy*.5);
      p->GetCrossSectionDataStore()->ComputeCrossSection(&evict,material);
     }
     double value=p->GetCrossSectionDataStore()->ComputeCrossSection(track.GetDynamicParticle(),material)*mm;
     finite(value);if(value<0)throw std::runtime_error("negative rate");
     double delta=value-rate;rate+=delta/(repeat+1);rate_m2+=delta*(value-rate);
     rate_min=std::min(rate_min,value);rate_max=std::max(rate_max,value);
    }
    double rate_sem=rate_repeats>1?std::sqrt(rate_m2/(rate_repeats-1)/rate_repeats):0;
    rates<<node<<','<<energies[node]<<','<<p->GetProcessName()<<','<<p->GetProcessSubType()<<','<<rate<<','<<event_id<<','<<(rate>0?samples:0)<<','<<rate_repeats<<','<<rate_sem<<','<<rate_min<<','<<rate_max<<'\n';
    if(rate<=0)continue;
    for(int sample=0;sample<samples;++sample) {
     p->StartTracking(&track);G4ForceCondition condition=NotForced;
     p->PostStepGetPhysicalInteractionLength(track,0,&condition);
     auto* base=p->PostStepDoIt(track,step);auto* change=dynamic_cast<G4ParticleChange*>(base);
     if(!change)throw std::runtime_error("unsupported particle-change type");
     const double edep=change->GetLocalEnergyDeposit()/MeV,remaining=change->GetEnergy()/MeV;
     const auto dir=*change->GetMomentumDirection();finite(edep);finite(remaining);
     if(edep<0 || remaining<0)throw std::runtime_error("negative native energy");
     const auto status=change->GetTrackStatus();const bool alive=status==fAlive || status==fStopButAlive;
     const auto first=product_id;double outgoing=alive?remaining:0;
     for(int i=0;i<change->GetNumberOfSecondaries();++i) {
      auto* secondary=change->GetSecondary(i);auto* def=secondary->GetDefinition();auto u=secondary->GetMomentumDirection();
      const double e=secondary->GetKineticEnergy()/MeV;finite(e);if(e<0 || secondary->GetWeight()!=1.)throw std::runtime_error("unsupported product weight/energy");
      products<<product_id++<<','<<event_id<<','<<def->GetPDGEncoding()<<','<<def->GetAtomicNumber()<<','<<def->GetAtomicMass()<<','<<def->GetPDGMass()/MeV<<','<<e<<','<<u.x()<<','<<u.y()<<','<<u.z()<<','<<secondary->GetGlobalTime()/ns<<'\n';
      outgoing+=e;delete secondary;
     }
     // This gain includes nuclear Q/thermal transfer. It is NOT forced to zero.
     events<<event_id++<<','<<node<<','<<p->GetProcessName()<<','<<p->GetTargetNucleus()->GetZ_asInt()<<','<<p->GetTargetNucleus()->GetA_asInt()<<','<<(p->GetHadronicInteraction()?p->GetHadronicInteraction()->GetModelName():G4String("not_exposed_by_process"))<<','<<int(status)<<','<<(alive?remaining:0)<<','<<dir.x()<<','<<dir.y()<<','<<dir.z()<<','<<edep<<','<<outgoing+edep-energies[node]<<','<<first<<','<<product_id-first<<'\n';
     change->Clear();
    }
   }
   if(node%16==0)std::cout<<"EXTRACT_NODE "<<node<<' '<<energies[node]<<std::endl;
  }
  original->SetTrackStatus(fStopAndKill);
 }
};
class Kill final:public G4UserStackingAction {
 G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override{return t->GetParentID()==0?fUrgent:fKill;}
};
}
int main(int argc,char** argv) {
 std::signal(SIGSEGV, +[](int){void* frames[40];int n=backtrace(frames,40);backtrace_symbols_fd(frames,n,2);_exit(139);});
 try {
  if(argc<5||argc>7)throw std::runtime_error("usage: extract_neutron OUTPUT MATERIAL ENERGY_GRID_TXT SAMPLES [RATE_REPEATS [SEED]]");
  if(argc==7)random_seed=std::stol(argv[6]);
  if(random_seed<=0||random_seed>2147483646)throw std::runtime_error("invalid random seed");
  if(argc>=6)rate_repeats=std::stoi(argv[5]);
  if(rate_repeats<1||rate_repeats>1000000)throw std::runtime_error("invalid rate repeats");
  output=argv[1];material_name=argv[2];samples=std::stoi(argv[4]);if(samples<0 || samples>1000000)throw std::runtime_error("invalid samples");
  if(rate_repeats>1&&samples!=0)throw std::runtime_error("repeated rate audit requires SAMPLES=0");
  if(material_name!="Water_75eV" && material_name!="G4_Cu" && material_name!="G4_AIR")throw std::runtime_error("unsupported material");
  std::ifstream grid(argv[3]);double e;while(grid>>e){if(!std::isfinite(e)||e<=0||(!energies.empty()&&e<=energies.back()))throw std::runtime_error("invalid grid");energies.push_back(e);}if(!grid.eof()||energies.empty())throw std::runtime_error("invalid grid file");
  if(std::filesystem::exists(output))throw std::runtime_error("output exists");std::filesystem::create_directories(output);
  rates.open(output/"rates.csv");events.open(output/"events.csv");products.open(output/"products.csv");provenance.open(output/"provenance.txt");
  for(auto* f:{&rates,&events,&products,&provenance})*f<<std::setprecision(17);
  rates<<"node,energy_MeV,process,subtype,rate_per_mm,event_offset,event_count,rate_queries,rate_sem_per_mm,rate_min_per_mm,rate_max_per_mm\n";
  events<<"event,node,process,target_z,target_a,model,status,continuation_MeV,dx,dy,dz,local_deposit_MeV,kinetic_gain_MeV,product_offset,product_count\n";
  products<<"product,event,pdg,z,a,mass_MeV,energy_MeV,dx,dy,dz,time_ns\n";
  CLHEP::HepRandom::setTheSeed(random_seed);
  auto* run=new G4RunManager;std::cerr<<"STAGE manager\n";run->SetUserInitialization(new Detector);std::cerr<<"STAGE detector\n";run->SetUserInitialization(new Physics);std::cerr<<"STAGE physics\n";run->SetUserAction(new Generator);run->SetUserAction(new Capture);run->SetUserAction(new Kill);std::cerr<<"STAGE actions\n";run->Initialize();std::cerr<<"STAGE initialized\n";
  provenance<<"geant4 "<<G4Version<<"\nmaterial "<<material_name<<"\ndensity_g_cm3 "<<material->GetDensity()/(g/cm3)<<"\ntemperature_K "<<material->GetTemperature()/kelvin<<"\nexcitation_eV "<<material->GetIonisation()->GetMeanExcitationEnergy()/eV<<"\nseed "<<random_seed<<"\nsamples_per_process_node "<<samples<<"\nrate_queries_per_node "<<rate_repeats<<'\n';
  for(std::size_t i=0;i<material->GetNumberOfElements();++i) {
   auto* element=(*material->GetElementVector())[i];
   provenance<<"element "<<element->GetName()<<" Z "<<element->GetZ()<<" A_g_mol "<<element->GetA()/(g/mole)<<" atoms_per_mm3 "<<material->GetVecNbOfAtomsPerVolume()[i]*mm3<<'\n';
   for(std::size_t j=0;j<element->GetNumberOfIsotopes();++j)provenance<<"isotope "<<(*element->GetIsotopeVector())[j]->GetName()<<" A "<<(*element->GetIsotopeVector())[j]->GetN()<<" abundance "<<element->GetRelativeAbundanceVector()[j]<<'\n';
  }
  run->BeamOn(1);if(!extracted)throw std::runtime_error("no active extraction context");
  for(auto* f:{&rates,&events,&products,&provenance}){f->flush();if(!*f)throw std::runtime_error("output write failure");}
  std::ofstream done(output/"COMPLETE");done<<event_id<<' '<<product_id<<'\n';
  delete run;return 0;
 }catch(const std::exception& e){std::cerr<<"EXTRACTION_REJECTED "<<e.what()<<'\n';return 1;}
}

// Scorer for PrimaryMaterialDump
#include "PrimaryMaterialDump.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4Isotope.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4HadronicProcess.hh"
#include "G4HadronicProcessStore.hh"
#include "G4CrossSectionDataStore.hh"
#include "G4EnergyRangeManager.hh"
#include "G4DynamicParticle.hh"
#include "G4HadProjectile.hh"
#include "G4HadFinalState.hh"
#include "G4Nucleus.hh"
#include "G4NucleiProperties.hh"
#include "G4EmCalculator.hh"
#include "G4SystemOfUnits.hh"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <set>
#include <stdexcept>
PrimaryMaterialDump::PrimaryMaterialDump(TsParameterManager* p,TsMaterialManager* m,
 TsGeometryManager* g,TsScoringManager* s,TsExtensionManager* x,G4String n,
 G4String q,G4String o,G4bool sub):TsVScorer(p,m,g,s,x,n,q,o,sub),stem_(o) {}
G4bool PrimaryMaterialDump::ProcessHits(G4Step* step,G4TouchableHistory*) {
 if(done_ || step->GetTrack()->GetParentID()!=0)return false;
 done_=true;
 auto* particle=step->GetTrack()->GetDefinition();auto* material=step->GetPreStepPoint()->GetMaterial();
 const int z=particle->GetAtomicNumber(),a=particle->GetAtomicMass();
 if(z<=0||a<z)throw std::runtime_error("Invalid source ZA");
 G4HadronicProcess* process=nullptr;int count=0;
 auto* pv=particle->GetProcessManager()->GetProcessList();
 for(std::size_t i=0;i<pv->size();++i) {
  auto* p=dynamic_cast<G4HadronicProcess*>((*pv)[i]);
  if(p&&p->GetProcessSubType()==fHadronElastic){process=p;++count;}
 }
 if(count!=1)throw std::runtime_error("Exactly one hadronic elastic process required");
 G4EnergyRangeManager manager;for(auto* model:process->GetHadronicInteractionList())manager.RegisterMe(model);
 auto* store=G4HadronicProcessStore::Instance();G4EmCalculator calc;
 std::ofstream sp(stem_+"_stopping.csv"),xs(stem_+"_inelastic_rate.csv"),el(stem_+"_elastic.csv");
 sp<<std::setprecision(17)<<"energy_MeVu,stopping_power_MeV_per_mm\n";
 xs<<std::setprecision(17)<<"energy_MeVu,macroscopic_cross_section_per_mm\n";
 el<<std::setprecision(17)<<"energy_MeV_per_u,macroscopic_cross_section_per_mm,transfer_fraction,target_a,target_mass_MeV\n";
 std::set<std::string> models;
 for(int i=0;i<=2510;++i) {
  const double eu=.01+.1*i,T=a*eu,m1=particle->GetPDGMass()/MeV;
  sp<<eu<<','<<calc.ComputeElectronicDEDX(T*MeV,particle,material)/(MeV/mm)<<'\n';
  xs<<eu<<','<<store->GetInelasticCrossSectionPerVolume(particle,T*MeV,material)*mm<<'\n';
  if(i%5!=0)continue;
  G4DynamicParticle dp(particle,{0,0,1},T*MeV);G4HadProjectile hp(dp);
  auto* data=process->GetCrossSectionDataStore();const double macro=data->ComputeCrossSection(&dp,material);
  if(!std::isfinite(macro)||macro<0)throw std::runtime_error("Invalid elastic rate");
  if(macro==0) {
   auto* element=(*material->GetElementVector())[0];
   const int ta=element->GetIsotope(0)->GetN(),tz=int(element->GetZ());
   el<<eu<<",0,0,"<<ta<<','<<G4NucleiProperties::GetNuclearMass(ta,tz)/MeV<<'\n';
   continue;
  }
  for(int j=0;j<512;++j) {
   G4Nucleus nucleus;auto* element=data->SampleZandA(&dp,material,nucleus);
   const int ta=nucleus.GetA_asInt(),tz=nucleus.GetZ_asInt();
   const double m2=G4NucleiProperties::GetNuclearMass(ta,tz)/MeV;
   auto* model=manager.GetHadronicInteraction(hp,nucleus,material,element);
   if(!model)throw std::runtime_error("No elastic model");models.insert(model->GetModelName());
   auto* fs=model->ApplyYourself(hp,nucleus);
   const double tmax=2*m2*T*(T+2*m1)/((m1+m2)*(m1+m2)+2*m2*T);
   double fraction=(T-fs->GetEnergyChange()/MeV)/tmax;
   if(!std::isfinite(fraction)||fraction< -1e-7||fraction>1+1e-5)throw std::runtime_error("Invalid elastic fraction");
   el<<eu<<','<<macro*mm<<','<<std::max(0.,std::min(1.,fraction))<<','<<ta<<','<<m2<<'\n';fs->Clear();
  }
 }
 if(!sp||!xs||!el)throw std::runtime_error("Material table write failed");
 std::ofstream meta(stem_+"_metadata.json");meta<<std::setprecision(17)<<"{\"z\":"<<z<<",\"a\":"<<a<<",\"mass_MeV\":"<<particle->GetPDGMass()/MeV<<",\"material\":\""<<material->GetName()<<"\",\"elastic_models\":[";
 bool first=true;for(auto& name:models){if(!first)meta<<',';first=false;meta<<'"'<<name<<'"';}meta<<"],\"samples_per_node\":512}\n";
 G4cout<<"PRIMARY_MATERIAL_DUMP=PASS Z="<<z<<" A="<<a<<" material="<<material->GetName()<<G4endl;
 return true;
}

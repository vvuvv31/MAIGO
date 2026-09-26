// Physics Module for AllChargedUrbanPhysics
#include "AllChargedUrbanPhysics.hh"
#include "G4PhysicsConstructorFactory.hh"
#include "G4PhysicsListHelper.hh"
#include "G4ProcessManager.hh"
#include "G4hMultipleScattering.hh"
#include "G4UrbanMscModel.hh"
#include "G4SystemOfUnits.hh"
#include <set>
#include <stdexcept>
G4_DECLARE_PHYSCONSTR_FACTORY(AllChargedUrbanPhysics);
AllChargedUrbanPhysics::AllChargedUrbanPhysics(G4int) : G4VPhysicsConstructor("AllChargedUrbanPhysics") {}
AllChargedUrbanPhysics::AllChargedUrbanPhysics(TsParameterManager*) : AllChargedUrbanPhysics(0) {}
void AllChargedUrbanPhysics::ConstructProcess() {
 auto* iterator=GetParticleIterator(); iterator->reset();
 std::set<G4ProcessManager*> visited;
 while((*iterator)()) {
  auto* particle=iterator->value(); auto* pm=particle->GetProcessManager();
  if(!pm || !visited.insert(pm).second) continue;
  auto* processes=pm->GetProcessList(); G4hMultipleScattering* old=nullptr;
  for(int j=0;j<pm->GetProcessListLength();++j)
   if(auto* m=dynamic_cast<G4hMultipleScattering*>((*processes)[j])) { old=m; break; }
  if(!old) continue;
  auto name=old->GetProcessName(); pm->RemoveProcess(old);
  auto* replacement=new G4hMultipleScattering(name);
  auto* model=new G4UrbanMscModel(); model->SetHighEnergyLimit(10*GeV);
  replacement->SetEmModel(model);
  if(!G4PhysicsListHelper::GetPhysicsListHelper()->RegisterProcess(replacement,particle))
   throw std::runtime_error("failed to register hadron Urban process");
  G4cout << "MAIGO Urban replacement: " << particle->GetParticleName() << G4endl;
 }
}

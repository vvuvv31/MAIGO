// Physics Module for PrimaryIonizationResponsePhysics
#include "PrimaryIonizationResponsePhysics.hh"
#include "G4PhysicsConstructorFactory.hh"
#include "G4ProcessManager.hh"
#include "G4ParticleDefinition.hh"
#include "G4VProcess.hh"
#include <set>
#include <vector>
G4_DECLARE_PHYSCONSTR_FACTORY(PrimaryIonizationResponsePhysics);
PrimaryIonizationResponsePhysics::PrimaryIonizationResponsePhysics(G4int)
 : G4VPhysicsConstructor("PrimaryIonizationResponsePhysics") {}
PrimaryIonizationResponsePhysics::PrimaryIonizationResponsePhysics(TsParameterManager*)
 : PrimaryIonizationResponsePhysics(0) {}
void PrimaryIonizationResponsePhysics::ConstructProcess() {
 auto* it=GetParticleIterator();it->reset();std::set<G4ProcessManager*> visited;
 while((*it)()) {
  auto* p=it->value();auto* pm=p->GetProcessManager();
  if(!pm || p->GetBaryonNumber()==0 || p->GetPDGCharge()==0 || !visited.insert(pm).second)continue;
  std::vector<G4VProcess*> remove;
  for(int j=0;j<pm->GetProcessListLength();++j) {
   auto* process=(*pm->GetProcessList())[j];const auto name=process->GetProcessName();
   if(name=="CoulombScat" || name=="nuclearStopping")remove.push_back(process);
  }
  for(auto* process:remove) {
   G4cout << "MAIGO electron-response extraction ONLY: remove " << process->GetProcessName()
          << " from " << p->GetParticleName() << G4endl;
   pm->RemoveProcess(process);
  }
 }
}

#include "DiagnosticIonLinearLoss.hh"
#include "TsParameterManager.hh"
#include "G4GenericIon.hh"
#include "G4ProcessManager.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4Exception.hh"
DiagnosticIonLinearLoss::DiagnosticIonLinearLoss(TsParameterManager* p)
 : G4VPhysicsConstructor("DiagnosticIonLinearLoss"),
 limit_(p->GetUnitlessParameter("Ph/DiagnosticLinearLossLimit")), bins_(p->GetIntegerParameter("Ph/DiagnosticIonDEDXBins")) {
 if(!(limit_>0.0 && limit_<1.0)) G4Exception("DiagnosticIonLinearLoss","BadLimit",FatalException,"Require 0 < limit < 1");
}
void DiagnosticIonLinearLoss::ConstructProcess() {
 auto* pm=G4GenericIon::GenericIon()->GetProcessManager();
 int changed=0;
 if(pm) for(int i=0;i<pm->GetProcessListLength();++i) {
  auto* p=dynamic_cast<G4VEnergyLossProcess*>((*pm->GetProcessList())[i]);
  if(p && p->GetProcessName()=="ionIoni") {p->SetLinearLossLimit(limit_); p->SetDEDXBinning(bins_); ++changed;}
 }
 if(changed!=1) G4Exception("DiagnosticIonLinearLoss","ProcessCount",FatalException,"Expected exactly one GenericIon ionIoni process");
 G4cout << "[diagnostic-linear-loss] GenericIon ionIoni limit=" << limit_ << " bins=" << bins_ << " count=" << changed << G4endl;
}

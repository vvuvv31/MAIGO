// Scorer for NeutronProtonBirthNtuple
#include "NeutronProtonBirthNtuple.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4StepPoint.hh"
#include "G4ParticleDefinition.hh"
#include "G4Material.hh"
#include "G4HadronicProcess.hh"
#include "G4SystemOfUnits.hh"
NeutronProtonBirthNtuple::NeutronProtonBirthNtuple(TsParameterManager* pm,TsMaterialManager* mm,
 TsGeometryManager* gm,TsScoringManager* sm,TsExtensionManager* em,
 G4String name,G4String quantity,G4String output,G4bool sub)
 :TsVNtupleScorer(pm,mm,gm,sm,em,name,quantity,output,sub){
 fNtuple->RegisterColumnI(&event_,"event");
 fNtuple->RegisterColumnI(&parent_pdg_,"parent_pdg");
 fNtuple->RegisterColumnI(&parent_track_,"parent_track");
 fNtuple->RegisterColumnI(&parent_step_,"parent_step");
 fNtuple->RegisterColumnI(&process_,"process_subtype");
 fNtuple->RegisterColumnI(&material_,"material_water0_copper1_air2");
 fNtuple->RegisterColumnI(&target_z_,"target_Z");
 fNtuple->RegisterColumnI(&target_a_,"target_A");
 fNtuple->RegisterColumnD(&parent_energy_,"parent_energy_MeV","");
 fNtuple->RegisterColumnD(&energy_,"energy_MeV","");
 fNtuple->RegisterColumnD(&x_,"x_mm","");fNtuple->RegisterColumnD(&y_,"y_mm","");fNtuple->RegisterColumnD(&z_,"z_mm","");
 fNtuple->RegisterColumnD(&dx_,"dx","");fNtuple->RegisterColumnD(&dy_,"dy","");fNtuple->RegisterColumnD(&dz_,"dz","");
 fNtuple->RegisterColumnD(&weight_,"weight","");
 fNtuple->RegisterColumnD(&parent_dx_,"parent_dx","");fNtuple->RegisterColumnD(&parent_dy_,"parent_dy","");fNtuple->RegisterColumnD(&parent_dz_,"parent_dz","");
}
G4bool NeutronProtonBirthNtuple::ProcessHits(G4Step* s,G4TouchableHistory*){
 if(!fIsActive){++fSkippedWhileInactive;return false;}
 const auto* children=s->GetSecondaryInCurrentStep();if(!children||children->empty())return false;
 const auto* t=s->GetTrack();const auto* pre=s->GetPreStepPoint();
 const auto* process=s->GetPostStepPoint()->GetProcessDefinedStep();
 event_=GetEventID();parent_pdg_=t->GetDefinition()->GetPDGEncoding();parent_track_=t->GetTrackID();parent_step_=t->GetCurrentStepNumber();
 process_=process?process->GetProcessSubType():-1;parent_energy_=pre->GetKineticEnergy()/MeV;
 auto pd=pre->GetMomentumDirection();parent_dx_=pd.x();parent_dy_=pd.y();parent_dz_=pd.z();
 const auto name=pre->GetMaterial()->GetName();material_=name=="Water_75eV"?0:name=="G4_Cu"?1:name=="G4_AIR"?2:-1;
 target_z_=target_a_=-1;
 if(auto* had=dynamic_cast<const G4HadronicProcess*>(process)){
  auto* nucleus=const_cast<G4HadronicProcess*>(had)->GetTargetNucleus();
  if(nucleus){target_z_=nucleus->GetZ_asInt();target_a_=nucleus->GetA_asInt();}
 }
 // TOPAS may pass a copied scoring step whose secondary-view offset is
 // reset, exposing older children from this parent. Accept only children at
 // this generating endpoint and record each pointer once per event/parent.
 if(active_event_!=event_){seen_.clear();active_event_=event_;}
 auto& seen=seen_[parent_track_];
 bool filled=false;
 for(const auto* child:*children){
  if(child->GetDefinition()->GetPDGEncoding()!=2212)continue;
  if((child->GetPosition()-s->GetPostStepPoint()->GetPosition()).mag2()>1.e-18*mm*mm)continue;
  if(!seen.insert(child).second)continue;
  auto p=child->GetPosition(),d=child->GetMomentumDirection();energy_=child->GetKineticEnergy()/MeV;
  x_=p.x()/mm;y_=p.y()/mm;z_=p.z()/mm;dx_=d.x();dy_=d.y();dz_=d.z();weight_=child->GetWeight();
  fNtuple->Fill();filled=true;
 }
 return filled;
}

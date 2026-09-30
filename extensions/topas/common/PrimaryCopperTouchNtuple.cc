// Scorer for PrimaryCopperTouchNtuple
#include "PrimaryCopperTouchNtuple.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
#include "G4Material.hh"
#include "G4SystemOfUnits.hh"
PrimaryCopperTouchNtuple::PrimaryCopperTouchNtuple(TsParameterManager* p,TsMaterialManager* m,
    TsGeometryManager* g,TsScoringManager* s,TsExtensionManager* e,G4String n,G4String q,
    G4String o,G4bool sub):TsVNtupleScorer(p,m,g,s,e,n,q,o,sub) {
    fNtuple->RegisterColumnI(&run_,"run");
    fNtuple->RegisterColumnI(&event_,"event");
    fNtuple->RegisterColumnI(&track_,"track");
    fNtuple->RegisterColumnD(&length_,"first_cu_step_mm","");
    fNtuple->RegisterColumnD(&energy_,"first_cu_pre_energy_MeV","");
}
G4bool PrimaryCopperTouchNtuple::ProcessHits(G4Step* s,G4TouchableHistory*) {
    if (!fIsActive) {++fSkippedWhileInactive;return false;}
    const auto* t=s->GetTrack();const auto* material=s->GetPreStepPoint()->GetMaterial();
    if(t->GetParentID()!=0 || s->GetStepLength()<=0 || !material ||
       material->GetNumberOfElements()!=1 || material->GetZ()!=29.)return false;
    run_=GetRunID();event_=GetEventID();track_=t->GetTrackID();
    const auto key=std::make_pair(run_,event_);
    if(key!=event_key_){seen_.clear();event_key_=key;}
    if(!seen_.insert(track_).second)return false;
    length_=s->GetStepLength()/mm;energy_=s->GetPreStepPoint()->GetKineticEnergy()/MeV;
    fNtuple->Fill();return true;
}

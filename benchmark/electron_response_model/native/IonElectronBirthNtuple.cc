#include "IonElectronBirthNtuple.hh"
#include "TsParameterManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
#include "G4ParticleDefinition.hh"
#include "G4Material.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>
#include <algorithm>
IonElectronBirthNtuple::IonElectronBirthNtuple(TsParameterManager* pm,TsMaterialManager* mm,
    TsGeometryManager* gm,TsScoringManager* sm,TsExtensionManager* em,G4String name,
    G4String quantity,G4String output,G4bool sub):TsVNtupleScorer(pm,mm,gm,sm,em,name,quantity,output,sub) {
    fNtuple->RegisterColumnI(&run_,"run");fNtuple->RegisterColumnI(&event_,"event");
    fNtuple->RegisterColumnI(&kind_,"kind");fNtuple->RegisterColumnI(&bin_,"bin");fNtuple->RegisterColumnI(&step_,"step");
    fNtuple->RegisterColumnD(&parentE_,"parent_pre_MeVu","");fNtuple->RegisterColumnD(&birthE_,"birth_MeV","");
    fNtuple->RegisterColumnD(&mu_,"cos_parent_electron","");fNtuple->RegisterColumnD(&loss_,"primary_loss_MeV","");
    fNtuple->RegisterColumnD(&local_,"primary_local_MeV","");fNtuple->RegisterColumnD(&children_,"children_MeV","");
    fNtuple->RegisterColumnD(&residual_,"max_step_residual_MeV","");fNtuple->RegisterColumnD(&maxStep_,"max_step_mm","");
    fNtuple->RegisterColumnD(&rho_,"density_g_cm3","");
}
void IonElectronBirthNtuple::Require(bool ok,const char* why) {
    if(!ok){G4cerr<<"IonElectronBirth audit run="<<run_<<" event="<<event_<<" "<<why<<G4endl;fPm->AbortSession(1);}
}
G4bool IonElectronBirthNtuple::ProcessHits(G4Step* s,G4TouchableHistory*) {
    if(!fIsActive){++fSkippedWhileInactive;return false;}
    const auto* t=s->GetTrack();if(t->GetParentID()!=0)return false;
    const auto pdg=t->GetDefinition()->GetPDGEncoding();
    Require(pdg==2212 || pdg==1000060120,"primary must be proton or C12");
    const double mass_number=pdg==2212?1.:12.;
    const auto* pre=s->GetPreStepPoint();const auto* post=s->GetPostStepPoint();
    Require(pre->GetWeight()==1,"weighted source unsupported");
    if(event_<0){event_=GetEventID();run_=GetRunID();}
    Require(event_==GetEventID()&&run_==GetRunID(),"event hook missing");
    step_=t->GetCurrentStepNumber();const double kinetic=pre->GetKineticEnergy()/MeV;parentE_=kinetic/mass_number;
    if(!root_){Require(step_==1,"missing primary start");initial_=last_=kinetic;root_=true;}
    Require(std::abs(last_-kinetic)<1e-9,"primary energy continuity");last_=post->GetKineticEnergy()/MeV;
    Require(parentE_>=0&&parentE_<=450.000001&&last_>=0&&last_<=kinetic+1e-10,"energy range");
    const int last_bin=std::min(89,int(std::ceil(initial_/(5*mass_number)))-1);
    bin_=std::min(last_bin,int(parentE_/5));rho_=pre->GetMaterial()->GetDensity()/(g/cm3);
    auto& b=sums_[bin_];const double loss=kinetic-last_,local=s->GetTotalEnergyDeposit()/MeV;double children=0;
    const auto* secondary=s->GetSecondaryInCurrentStep();
    for(const auto* c:*secondary) {
        Require(c->GetDefinition()->GetPDGEncoding()==11,"non-electron primary child");
        birthE_=c->GetKineticEnergy()/MeV;mu_=pre->GetMomentumDirection().dot(c->GetMomentumDirection());
        Require(std::isfinite(birthE_)&&birthE_>=0&&std::isfinite(mu_)&&std::abs(mu_)<=1+1e-12,"birth state");
        children+=birthE_;if(birthE_==0)continue;
        mu_=std::max(-1.,std::min(1.,mu_));kind_=1;
        loss_=local_=children_=residual_=maxStep_=0;fNtuple->Fill();
    }
    const double residual=std::abs(loss-local-children),length=s->GetStepLength()/mm;
    Require(residual<1e-8,"primary energy closure");Require(length<=.250001,"primary step exceeds0.25mm");
    b.loss+=loss;b.local+=local;b.children+=children;b.maxResidual=std::max(b.maxResidual,residual);
    b.maxStep=std::max(b.maxStep,length);++b.steps;
    return true;
}
void IonElectronBirthNtuple::UserHookForEndOfEvent() {
    if(!root_)return;
    Require(last_==0,"primary left source material before stopping");
    double total=0;
    for(bin_=0;bin_<90;++bin_) {
        const auto& b=sums_[bin_];if(!b.steps)continue;
        kind_=0;step_=b.steps;parentE_=bin_*5;birthE_=mu_=0;loss_=b.loss;local_=b.local;children_=b.children;
        residual_=b.maxResidual;maxStep_=b.maxStep;total+=loss_;fNtuple->Fill();
    }
    Require(std::abs(total-initial_)<1e-7,"event energy closure");
    sums_={};root_=false;run_=event_=-1;initial_=last_=0;
}

#include "PhotonResponseNtuple.hh"
#include "TsParameterManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4AtomicShells.hh"
#include "G4EmCalculator.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <set>
PhotonResponseNtuple::PhotonResponseNtuple(TsParameterManager* pm,TsMaterialManager* mm,
    TsGeometryManager* gm,TsScoringManager* sm,TsExtensionManager* em,G4String name,
    G4String quantity,G4String output,G4bool sub):TsVNtupleScorer(pm,mm,gm,sm,em,name,quantity,output,sub) {
    fNtuple->RegisterColumnI(&run_,"run");fNtuple->RegisterColumnI(&event_,"event");
    fNtuple->RegisterColumnI(&step_,"step");fNtuple->RegisterColumnI(&kind_,"kind");
    fNtuple->RegisterColumnI(&pdg_,"pdg");fNtuple->RegisterColumnI(&process_,"process");
    fNtuple->RegisterColumnD(&energy_,"pre_MeV","");fNtuple->RegisterColumnD(&remaining_,"post_or_child_MeV","");
    fNtuple->RegisterColumnD(&deposit_,"deposit_MeV","");fNtuple->RegisterColumnD(&children_,"children_MeV","");
    for(int i=0;i<3;++i)fNtuple->RegisterColumnD(pre_+i,"pre_direction_"+std::to_string(i),"");
    for(int i=0;i<3;++i)fNtuple->RegisterColumnD(post_+i,"post_direction_"+std::to_string(i),"");
    fNtuple->RegisterColumnD(&length_,"length_mm","");fNtuple->RegisterColumnD(&density_,"density_g_cm3","");
}
void PhotonResponseNtuple::Require(bool ok,const char* why) {
    if(!ok){G4cerr<<"PhotonResponse audit run="<<run_<<" event="<<event_<<" "<<why<<G4endl;fPm->AbortSession(1);}
}
void PhotonResponseNtuple::DumpRates(const G4Step* step) {
    if(dumped_)return;dumped_=true;
    if(!std::getenv("MAIGO_PHOTON_XS"))return;
    const auto* material=step->GetPreStepPoint()->GetMaterial();
    const auto* particle=step->GetTrack()->GetDefinition();
    std::set<double> energies;
    for(int i=0;i<=4096;++i)energies.insert(1e-5*std::pow(.65/1e-5,double(i)/4096));
    for(const auto* el:*material->GetElementVector()) {
        const int z=int(el->GetZ());
        for(int i=0;i<G4AtomicShells::GetNumberOfShells(z);++i){
            const double e=G4AtomicShells::GetBindingEnergy(z,i)/MeV;
            for(double factor:{1-1e-6,1.,1+1e-6})if(e*factor>=1e-5&&e*factor<=.65)energies.insert(e*factor);
        }
    }
    G4EmCalculator calculator;
    std::ofstream f("photon_rates.csv");f<<std::setprecision(17)<<"energy_MeV,phot_per_mm,compt_per_mm,Rayl_per_mm,conv_per_mm\n";
    for(double e:energies){f<<e;
        for(const char* process:{"phot","compt","Rayl","conv"}) {
            const double rate=calculator.GetCrossSectionPerVolume(e*MeV,particle,process,material)*mm;
            Require(std::isfinite(rate)&&rate>=0,"invalid material photon rate");f<<','<<rate;
        }f<<'\n';
    }
    Require(bool(f),"photon rate output failed");
}
G4bool PhotonResponseNtuple::ProcessHits(G4Step* s,G4TouchableHistory*) {
    if(!fIsActive){++fSkippedWhileInactive;return false;}
    const auto* t=s->GetTrack();if(t->GetParentID()!=0)return false;
    Require(t->GetDefinition()->GetPDGEncoding()==22,"primary must be gamma");
    const auto* pre=s->GetPreStepPoint();const auto* post=s->GetPostStepPoint();
    Require(pre->GetWeight()==1,"weighted source unsupported");
    if(event_<0){event_=GetEventID();run_=GetRunID();}
    Require(event_==GetEventID()&&run_==GetRunID(),"event hook missing");
    step_=t->GetCurrentStepNumber();energy_=pre->GetKineticEnergy()/MeV;
    if(!root_){Require(step_==1,"missing photon start");root_=true;initial_=last_=energy_;}
    Require(std::abs(energy_-last_)<1e-10,"primary energy discontinuity");
    last_=post->GetKineticEnergy()/MeV;density_=pre->GetMaterial()->GetDensity()/(g/cm3);
    const auto name=post->GetProcessDefinedStep()?post->GetProcessDefinedStep()->GetProcessName():"none";
    process_=name=="phot"?0:(name=="compt"?1:(name=="Rayl"?2:(name=="conv"?3:-1)));
    const auto* secondary=s->GetSecondaryInCurrentStep();double children=0;
    for(const auto* c:*secondary){Require(c->GetDefinition()->GetPDGEncoding()==11||c->GetDefinition()->GetPDGEncoding()==22,"unsupported photon child");children+=c->GetKineticEnergy()/MeV;}
    const double dep=s->GetTotalEnergyDeposit()/MeV,residual=energy_-last_-dep-children;
    Require(dep>=0&&std::abs(residual)<1e-9,"photon step energy closure");
    maxResidual_=std::max(maxResidual_,std::abs(residual));total_+=dep+children;
    if(process_>=0){
        kind_=0;pdg_=22;remaining_=last_;deposit_=dep;children_=children;length_=s->GetStepLength()/mm;
        for(int i=0;i<3;++i){pre_[i]=pre->GetMomentumDirection()[i];post_[i]=post->GetMomentumDirection()[i];}
        fNtuple->Fill();
        for(const auto* c:*secondary){
            kind_=1;pdg_=c->GetDefinition()->GetPDGEncoding();remaining_=c->GetKineticEnergy()/MeV;
            deposit_=children_=0;
            for(int i=0;i<3;++i)post_[i]=c->GetMomentumDirection()[i];
            fNtuple->Fill();
        }
    }else Require(dep==0&&children==0&&energy_==last_,"unrecorded noncollision photon energy loss");
    DumpRates(s);return true;
}
void PhotonResponseNtuple::UserHookForEndOfEvent() {
    Require(root_,"missing primary");Require(std::abs(initial_-total_-last_)<1e-8,"photon event closure");
    kind_=2;pdg_=22;process_=-1;energy_=initial_;remaining_=last_;deposit_=total_;children_=maxResidual_;
    for(int i=0;i<3;++i)pre_[i]=post_[i]=0;length_=0;fNtuple->Fill();
    root_=false;event_=run_=-1;initial_=last_=total_=maxResidual_=0;
}

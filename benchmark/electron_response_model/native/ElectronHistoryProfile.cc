// Scorer for ElectronHistoryProfile
#include "ElectronHistoryProfile.hh"
#include "TsParameterManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
#include "G4ParticleDefinition.hh"
#include "G4Material.hh"
#include "G4SystemOfUnits.hh"
#include <algorithm>
#include <cmath>
#include <string>

ElectronHistoryProfile::ElectronHistoryProfile(
    TsParameterManager* pm, TsMaterialManager* mm, TsGeometryManager* gm,
    TsScoringManager* sm, TsExtensionManager* em, G4String name,
    G4String quantity, G4String output, G4bool sub)
    : TsVNtupleScorer(pm, mm, gm, sm, em, name, quantity, output, sub) {
    fNtuple->RegisterColumnI(&run_, "run");
    fNtuple->RegisterColumnI(&event_, "event");
    fNtuple->RegisterColumnI(&roots_, "roots");
    fNtuple->RegisterColumnI(&steps_, "steps");
    fNtuple->RegisterColumnI(&tracks_, "tracks");
    fNtuple->RegisterColumnD(&incident_, "incident_MeV", "");
    fNtuple->RegisterColumnD(&charged_, "charged_MeV", "");
    fNtuple->RegisterColumnD(&sink_, "photon_birth_sink_MeV", "");
    fNtuple->RegisterColumnD(&outside_, "outside_deposit_MeV", "");
    fNtuple->RegisterColumnD(&maxResidual_, "max_step_residual_MeV", "");
    fNtuple->RegisterColumnD(&maxStep_, "max_charged_step_mm", "");
    for (int i=0; i<36; ++i)
        fNtuple->RegisterColumnD(&profile_[i], "x_bin_"+std::to_string(i)+"_MeV", "");
}

void ElectronHistoryProfile::Require(bool ok, const char* reason) {
    if (!ok) {
        G4cerr << "ElectronHistoryProfile audit failure run=" << run_
               << " event=" << event_ << ": " << reason << G4endl;
        fPm->AbortSession(1);
    }
}

G4bool ElectronHistoryProfile::ProcessHits(G4Step* s, G4TouchableHistory*) {
    Require(fIsActive, "inactive scorer");
    auto* t = s->GetTrack();
    const auto* pre = s->GetPreStepPoint();
    const auto* post = s->GetPostStepPoint();
    const int id = t->GetTrackID(), pdg = t->GetDefinition()->GetPDGEncoding();
    Require(pdg == 11 || pdg == 22, "unsupported particle");
    Require(pre->GetWeight() == 1., "weighted source unsupported");
    if (event_ < 0) { event_=GetEventID(); run_=GetRunID(); }
    Require(event_ == GetEventID() && run_ == GetRunID(), "event hook missing");
    ++steps_;
    if (t->GetCurrentStepNumber() == 1) {
        ++tracks_;
        Require(excluded_.find(id) == excluded_.end(), "duplicate first step");
        if (t->GetParentID() == 0) {
            Require(pdg == 11, "primary must be electron");
            ++roots_; incident_ += pre->GetKineticEnergy()/MeV;
            excluded_[id] = false;
        } else {
            const auto found = pending_.find(t);
            Require(found != pending_.end(), "missing generating step");
            excluded_[id] = found->second;
            pending_.erase(found);
        }
    }
    Require(excluded_.find(id) != excluded_.end(), "missing track state");
    const bool excluded = excluded_.at(id);
    double childrenEnergy=0.;
    const auto* children = s->GetSecondaryInCurrentStep();
    if (children) for (const auto* child : *children) {
        const auto childPdg = child->GetDefinition()->GetPDGEncoding();
        Require(childPdg == 11 || childPdg == 22, "unsupported child");
        Require(pending_.emplace(child, excluded || childPdg == 22).second, "duplicate pending child");
        const double kinetic=child->GetKineticEnergy()/MeV;
        childrenEnergy += kinetic;
        if (!excluded && childPdg == 22) sink_ += kinetic;
    }
    const double dep = s->GetTotalEnergyDeposit()/MeV;
    const double residual=pre->GetKineticEnergy()/MeV-post->GetKineticEnergy()/MeV-dep-childrenEnergy;
    maxResidual_=std::max(maxResidual_, std::abs(residual));
    Require(std::abs(residual) <= 1e-8+1e-10*pre->GetKineticEnergy()/MeV, "step energy closure");
    Require(dep >= 0., "negative deposit");
    // Only this isolated lung fixture is supported. Vacuum has tiny density.
    const bool inside=pre->GetMaterial()->GetDensity()/(g/cm3) > .1;
    if (!inside) {
        outside_ += dep;
        Require(excluded, "charged family escaped finite reference");
    }
    if (inside && !excluded) {
        Require(pdg == 11, "unexcluded photon");
        maxStep_=std::max(maxStep_, s->GetStepLength()/mm);
        charged_ += dep;
        if (dep > 0.) {
            const double a=std::min(pre->GetPosition().x(),post->GetPosition().x())/mm;
            const double b=std::max(pre->GetPosition().x(),post->GetPosition().x())/mm;
            const int first=static_cast<int>(std::floor((a+1.8)/.1));
            const int last=static_cast<int>(std::floor((b+1.8)/.1));
            for (int bin=first; bin<=last; ++bin) {
                const double lo=std::max(a,-1.8+bin*.1), hi=std::min(b,-1.8+(bin+1)*.1);
                const double fraction=b>a ? std::max(0.,hi-lo)/(b-a) : 1.;
                profile_[((bin%36)+36)%36] += dep*fraction;
            }
        }
    }
    return true;
}

void ElectronHistoryProfile::UserHookForEndOfEvent() {
    Require(roots_ == 1 && event_ >= 0, "missing/multiple primaries");
    Require(pending_.empty(), "unrecorded secondary tracks");
    Require(outside_ < 1e-12, "non-negligible vacuum deposit");
    Require(std::abs(incident_-charged_-sink_) < 1e-8, "history energy closure");
    double integral=0.; for (auto v : profile_) integral+=v;
    Require(std::abs(integral-charged_) < 1e-10, "profile energy closure");
    fNtuple->Fill();
    event_=run_=-1; roots_=steps_=tracks_=0;
    incident_=charged_=sink_=outside_=maxResidual_=maxStep_=0.;
    profile_.fill(0.); excluded_.clear(); pending_.clear();
}

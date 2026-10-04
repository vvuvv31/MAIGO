// Scorer for ElectronFamilyProfile
#include "ElectronFamilyProfile.hh"
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
#include <iomanip>

ElectronFamilyProfile::ElectronFamilyProfile(
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
    fNtuple->RegisterColumnD(&photonDeposit_, "photon_family_deposit_MeV", "");
    fNtuple->RegisterColumnD(&escaped_, "escaped_MeV", "");
    for (int i=0; i<36; ++i)
        fNtuple->RegisterColumnD(&profile_[i], "x_bin_"+std::to_string(i)+"_MeV", "");
    for (int i=0; i<36; ++i)
        fNtuple->RegisterColumnD(&photonProfile_[i], "photon_x_bin_"+std::to_string(i)+"_MeV", "");
    fNtuple->RegisterColumnI(&navigationSteps_, "certified_internal_null_navigation_steps");
    fNtuple->RegisterColumnD(&maxNavigationPath_, "max_internal_navigation_path_mm", "");
}

void ElectronFamilyProfile::Require(bool ok, const char* reason) {
    if (!ok) {
        G4cerr << "ElectronFamilyProfile audit failure run=" << run_
               << " event=" << event_ << ": " << reason << G4endl;
        fPm->AbortSession(1);
    }
}

G4bool ElectronFamilyProfile::ProcessHits(G4Step* s, G4TouchableHistory*) {
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
    const auto p0=pre->GetPosition(), d0=pre->GetMomentumDirection();
    const std::array<G4double,7> state{p0.x(),p0.y(),p0.z(),d0.x(),d0.y(),d0.z(),pre->GetKineticEnergy()};
    const auto pendingNavigation=pendingNavigation_.find(id);
    if (pendingNavigation!=pendingNavigation_.end()) {
        Require(state==pendingNavigation->second, "null navigation changed next pre-state");
        if (inside) pendingNavigation_.erase(pendingNavigation);
    }
    // TsImageCube can transiently report its Vacuum container at an internal
    // voxel face. Certify a tolerance-layer flight and its following state;
    // no physical transport is changed. Account its tiny Vacuum loss outside
    // the material, under the existing per-history <1e-12 MeV audit.
    if (!inside && std::abs(p0.x()/mm)<20.-1e-6 && std::abs(p0.y()/mm)<20.-1e-6 && std::abs(p0.z()/mm)<20.-1e-6) {
        bool onFace=false;
        const double coordinate[3]={p0.x()/mm,p0.y()/mm,p0.z()/mm}, spacing[3]={.1,2.,.5};
        const auto p1=post->GetPosition(),d1=post->GetMomentumDirection();
        for (int axis=0;axis<3;++axis) {
            const double face=std::round(coordinate[axis]/spacing[axis])*spacing[axis];
            onFace=onFace || (std::abs(coordinate[axis]-face)<=1e-9 && std::abs(p1[axis]/mm-face)<=1e-9);
        }
        Require(onFace && s->GetStepLength()/mm<=1e-6 && (p1-p0).mag()/mm<=1e-6
                && pre->GetMomentumDirection()==post->GetMomentumDirection()
                && pre->GetKineticEnergy()==post->GetKineticEnergy() && dep<1e-12
                && (!children || children->empty()), "uncertified internal vacuum transport");
        pendingNavigation_[id]={p1.x(),p1.y(),p1.z(),d1.x(),d1.y(),d1.z(),post->GetKineticEnergy()};
        ++navigationSteps_; outside_+=dep;
        maxNavigationPath_=std::max(maxNavigationPath_,s->GetStepLength()/mm);
        return true;
    }
    if (!inside) {
        outside_ += dep;
        if (!excluded) G4cerr << "Family outside diagnostic: id=" << id
            << " parent=" << t->GetParentID() << " pdg=" << pdg << " step=" << t->GetCurrentStepNumber()
            << " pre_mm=" << pre->GetPosition()/mm << " post_mm=" << post->GetPosition()/mm
            << " E_MeV=" << pre->GetKineticEnergy()/MeV << " remaining_MeV=" << post->GetKineticEnergy()/MeV
            << " density=" << pre->GetMaterial()->GetDensity()/(g/cm3)
            << " material=" << pre->GetMaterial()->GetName() << G4endl;
        Require(excluded, "charged family escaped finite reference");
        if (escapedTracks_.insert(id).second)
            escaped_ += pre->GetKineticEnergy()/MeV;
    }
    if (inside) {
        Require(escapedTracks_.count(id)==0, "escaped particle returned");
        if (pdg==11) maxStep_=std::max(maxStep_, s->GetStepLength()/mm);
        if (excluded) photonDeposit_ += dep;
        else { Require(pdg==11, "unexcluded photon"); charged_ += dep; }
        if (dep > 0.) {
            // Photon transfer is at the interaction endpoint, never smeared
            // along its free flight. Charged deposition follows its chord.
            const double x0=(pdg==22 ? post : pre)->GetPosition().x()/mm;
            const double x1=post->GetPosition().x()/mm;
            const double a=std::min(x0,x1), b=std::max(x0,x1);
            const int first=static_cast<int>(std::floor((a+1.8)/.1));
            const int last=static_cast<int>(std::floor((b+1.8)/.1));
            for (int bin=first; bin<=last; ++bin) {
                const double lo=std::max(a,-1.8+bin*.1), hi=std::min(b,-1.8+(bin+1)*.1);
                const double fraction=b>a ? std::max(0.,hi-lo)/(b-a) : 1.;
                (excluded ? photonProfile_ : profile_)[((bin%36)+36)%36] += dep*fraction;
            }
        }
    }
    return true;
}

void ElectronFamilyProfile::UserHookForEndOfEvent() {
    Require(roots_ == 1 && event_ >= 0, "missing/multiple primaries");
    Require(pending_.empty(), "unrecorded secondary tracks");
    Require(pendingNavigation_.empty(), "unresolved internal navigation state");
    Require(outside_ < 1e-12, "non-negligible vacuum deposit");
    Require(std::abs(incident_-charged_-sink_) < 1e-8, "history energy closure");
    Require(std::abs(sink_-photonDeposit_-escaped_) < 1e-8, "photon family closure");
    Require(std::abs(incident_-charged_-photonDeposit_-escaped_) < 1e-8, "full family closure");
    double integral=0.; for (auto v : profile_) integral+=v;
    Require(std::abs(integral-charged_) < 1e-10, "profile energy closure");
    integral=0.; for (auto v : photonProfile_) integral+=v;
    Require(std::abs(integral-photonDeposit_) < 1e-10, "photon profile energy closure");
    fNtuple->Fill();
    event_=run_=-1; roots_=steps_=tracks_=0;
    incident_=charged_=sink_=outside_=maxResidual_=maxStep_=0.;
    photonDeposit_=escaped_=0.;
    navigationSteps_=0; maxNavigationPath_=0.; pendingNavigation_.clear();
    profile_.fill(0.); photonProfile_.fill(0.); excluded_.clear(); pending_.clear(); escapedTracks_.clear();
}

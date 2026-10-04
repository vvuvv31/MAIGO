#include "PhotonConditionalDump.hh"
#include "TsParameterManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4DynamicParticle.hh"
#include "G4ProcessManager.hh"
#include "G4VEmProcess.hh"
#include "G4ParticleChangeForGamma.hh"
#include "G4Material.hh"
#include "G4SystemOfUnits.hh"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <limits>
namespace {
struct Row {
    std::int32_t run{},event{},step{1},kind{},pdg{22},process{};
    double energy{},remaining{},deposit{},children{},pre[3]{0,0,1},post[3]{},length{},density{};
};
static_assert(sizeof(Row)==120);
}
PhotonConditionalDump::PhotonConditionalDump(TsParameterManager* pm,TsMaterialManager* mm,
    TsGeometryManager* gm,TsScoringManager* sm,TsExtensionManager* em,G4String name,
    G4String quantity,G4String output,G4bool sub):TsVScorer(pm,mm,gm,sm,em,name,quantity,output,sub) {}
void PhotonConditionalDump::Require(bool ok,const char* why) {
    if(!ok){G4cerr<<"PhotonConditionalDump audit: "<<why<<G4endl;fPm->AbortSession(1);}
}
G4bool PhotonConditionalDump::ProcessHits(G4Step* source,G4TouchableHistory*) {
    if(done_ || source->GetTrack()->GetParentID()!=0)return false;
    done_=true;
    const auto* primary=source->GetTrack();const auto* particle=primary->GetDefinition();
    Require(particle->GetPDGEncoding()==22,"trigger must be a photon");
    const auto* material=source->GetPreStepPoint()->GetMaterial();
    const auto* couple=source->GetPreStepPoint()->GetMaterialCutsCouple();
    G4VEmProcess* processes[3]{};const char* names[]={"phot","compt","Rayl"};
    const auto* list=particle->GetProcessManager()->GetProcessList();
    for(std::size_t j=0;j<list->size();++j) {
        auto* em=dynamic_cast<G4VEmProcess*>((*list)[j]);if(!em)continue;
        for(int i=0;i<3;++i)if(auto* p=em->GetEmProcess(names[i])) {
            Require(!processes[i],"duplicate photon process");processes[i]=p;
        }
    }
    for(auto* p:processes)Require(p,"missing native photon process");
    std::ifstream requests("requests.txt");Require(bool(requests),"missing process requests");
    std::ofstream raw("conditional.phsp",std::ios::binary),audit("conditional_audit.csv");
    audit<<std::setprecision(17)<<"cell,process,energy_MeV,rate_per_mm,requested,accepted,null_collisions,max_energy_residual_MeV\n";
    int cell,proc,count,event=0;double energy;
    while(requests>>cell>>proc>>energy>>count) {
        Require(cell>=0&&proc>=0&&proc<3&&std::isfinite(energy)&&energy>0&&energy<=.65&&count>0,"invalid request");
        G4Step step;auto* pre=step.GetPreStepPoint();auto* post=step.GetPostStepPoint();
        pre->SetMaterial(const_cast<G4Material*>(material));post->SetMaterial(const_cast<G4Material*>(material));
        pre->SetMaterialCutsCouple(couple);post->SetMaterialCutsCouple(couple);
        pre->SetPosition({0,0,0});post->SetPosition({0,0,0});pre->SetWeight(1);post->SetWeight(1);
        G4Track track(new G4DynamicParticle(particle,{0,0,1},energy*MeV),0,{0,0,0});
        track.SetParentID(1);track.SetTrackID(1);track.SetWeight(1);
        track.SetTouchableHandle(primary->GetTouchableHandle());track.SetStep(&step);step.SetTrack(&track);
        auto* process=processes[proc];const double rate=process->GetLambda(energy*MeV,couple,std::log(energy*MeV))*mm;
        Require(std::isfinite(rate)&&rate>0,"requested process has no native support");
        double maxResidual=0;int accepted=0,nulls=0;
        for(std::uint64_t attempt=0;accepted<count && attempt<std::uint64_t(count)*10000;++attempt) {
            process->StartTracking(&track);G4ForceCondition condition=NotForced;
            const double distance=process->PostStepGetPhysicalInteractionLength(track,-1,&condition);
            Require(std::isfinite(distance)&&distance<std::numeric_limits<double>::max(),"no native process flight");
            auto* result=dynamic_cast<G4ParticleChangeForGamma*>(process->PostStepDoIt(track,step));
            Require(result,"wrong native change type");
            const double remaining=result->GetProposedKineticEnergy()/MeV,dep=result->GetLocalEnergyDeposit()/MeV;
            const auto direction=result->GetProposedMomentumDirection();const int n=result->GetNumberOfSecondaries();
            if(n==0&&dep==0&&remaining==energy&&direction==G4ThreeVector(0,0,1)) {
                ++nulls;result->Clear();continue; // native integral-method null collision
            }
            double children=0;
            for(int i=0;i<n;++i){const auto* c=result->GetSecondary(i);const int pdg=c->GetDefinition()->GetPDGEncoding();Require(pdg==11||pdg==22,"unsupported child");children+=c->GetKineticEnergy()/MeV;}
            const double residual=std::abs(energy-remaining-dep-children);
            Require(dep>=0&&remaining>=0&&residual<1e-9,"native conditional energy closure");maxResidual=std::max(maxResidual,residual);
            Row r;r.run=cell;r.event=event++;r.process=proc;r.energy=energy;r.remaining=remaining;r.deposit=dep;r.children=children;r.density=material->GetDensity()/(g/cm3);
            for(int k=0;k<3;++k)r.post[k]=direction[k];
            raw.write(reinterpret_cast<const char*>(&r),sizeof(r));
            for(int i=0;i<n;++i){auto* child=result->GetSecondary(i);r.kind=1;r.pdg=child->GetDefinition()->GetPDGEncoding();r.remaining=child->GetKineticEnergy()/MeV;r.deposit=r.children=0;
                for(int k=0;k<3;++k)r.post[k]=child->GetMomentumDirection()[k];
                raw.write(reinterpret_cast<const char*>(&r),sizeof(r));delete child;
            }
            result->Clear();++accepted;
        }
        Require(accepted==count,"conditional attempt cap");
        audit<<cell<<','<<proc<<','<<energy<<','<<rate<<','<<count<<','<<accepted<<','<<nulls<<','<<maxResidual<<'\n';
    }
    Require(requests.eof()&&event>0&&bool(raw)&&bool(audit),"conditional output failure");
    G4cout<<"PHOTON_CONDITIONAL_DUMP=PASS events="<<event<<G4endl;
    return true;
}

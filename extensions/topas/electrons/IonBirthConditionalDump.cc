#include "ElectronExtractionRange.hh"
#include "IonBirthConditionalDump.hh"
#include "TsParameterManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4DynamicParticle.hh"
#include "G4ProcessManager.hh"
#include "G4Proton.hh"
#include "G4IonTable.hh"
#include "G4ProductionCutsTable.hh"
#include "G4AtomicShells.hh"
#include "G4Element.hh"
#include "G4VEmModel.hh"
#include "G4VEmFluctuationModel.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4ParticleChangeForLoss.hh"
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
    std::int32_t run{},event{},step{1},kind{},pdg{11},process{};
    double energy{},remaining{},deposit{},children{},pre[3]{0,0,1},post[3]{},length{},density{};
};
static_assert(sizeof(Row)==120);
}
IonBirthConditionalDump::IonBirthConditionalDump(TsParameterManager* pm,TsMaterialManager* mm,
    TsGeometryManager* gm,TsScoringManager* sm,TsExtensionManager* em,G4String name,
    G4String quantity,G4String output,G4bool sub):TsVScorer(pm,mm,gm,sm,em,name,quantity,output,sub) {}
void IonBirthConditionalDump::Require(bool ok,const char* why) {
    if(!ok){G4cerr<<"IonBirthConditionalDump audit: "<<why<<G4endl;fPm->AbortSession(1);}
}
G4bool IonBirthConditionalDump::ProcessHits(G4Step* source,G4TouchableHistory*) {
    if(done_ || source->GetTrack()->GetParentID()!=0)return false;
    done_=true;
    const auto* primary=source->GetTrack();
    const int a=fPm->GetIntegerParameter(GetFullParmName("ProjectileA"));
    Require(a==1||a==12,"unsupported projectile");
    auto* particle=a==1?G4Proton::Proton():G4IonTable::GetIonTable()->GetIon(6,12,0.);
    Require(primary->GetDefinition()->GetPDGEncoding()==22,"trigger must be a photon");
    const auto* material=source->GetPreStepPoint()->GetMaterial();
    const auto* couple=source->GetPreStepPoint()->GetMaterialCutsCouple();
    G4VEnergyLossProcess* processes[1]{};const char* names[]={a==1?"hIoni":"ionIoni"};
    const auto* list=particle->GetProcessManager()->GetProcessList();
    for(std::size_t j=0;j<list->size();++j) {
        auto* em=dynamic_cast<G4VEnergyLossProcess*>((*list)[j]);if(!em)continue;
        for(int i=0;i<1;++i)if(em->GetProcessName()==names[i]) {
            auto* p=em;
            Require(!processes[i],"duplicate electron process");processes[i]=p;
        }
    }
    for(auto* p:processes)Require(p,"missing native electron process");
    std::ofstream shells("atomic_shells.csv");
    shells<<std::setprecision(17)<<"Z,electron_fraction,binding_MeV,shell_fraction\n";
    for(std::size_t i=0;i<material->GetNumberOfElements();++i){
        const auto* element=material->GetElement(i);const int z=element->GetZasInt();
        const double fraction=material->GetVecNbOfAtomsPerVolume()[i]*z/material->GetTotNbOfElectPerVolume();
        const int n=G4AtomicShells::GetNumberOfShells(z);double sum=0;
        for(int j=0;j<n;++j)sum+=G4AtomicShells::GetNumberOfElectrons(z,j)/G4AtomicShells::GetBindingEnergy(z,j);
        for(int j=0;j<n;++j)shells<<z<<','<<fraction<<','<<G4AtomicShells::GetBindingEnergy(z,j)/MeV<<','
            <<G4AtomicShells::GetNumberOfElectrons(z,j)/G4AtomicShells::GetBindingEnergy(z,j)/sum<<'\n';
    }
    Require(bool(shells),"shell metadata failure");
    std::ofstream metadata("ion_models.txt");
    metadata<<std::setprecision(17)<<"a="<<a<<" mass_MeV="<<particle->GetPDGMass()/MeV
        <<" electron_cut_MeV="<<(*G4ProductionCutsTable::GetProductionCutsTable()->GetEnergyCutsVector(1))[couple->GetIndex()]/MeV<<"\n";
    std::ifstream requests("requests.txt");Require(bool(requests),"missing process requests");
    std::ofstream raw("conditional.phsp",std::ios::binary),audit("conditional_audit.csv");
    audit<<std::setprecision(17)<<"cell,process,energy_MeV,rate_per_mm,requested,accepted,null_collisions,max_energy_residual_MeV\n";
    int cell,proc,count,event=0;double energy;
    while(requests>>cell>>proc>>energy>>count) {
        Require(cell>=0&&proc>=0&&proc<1&&std::isfinite(energy)&&energy>0&&energy<=a*450.&&count>0,"invalid request");
        G4Step step;auto* pre=step.GetPreStepPoint();auto* post=step.GetPostStepPoint();
        pre->SetMaterial(const_cast<G4Material*>(material));post->SetMaterial(const_cast<G4Material*>(material));
        pre->SetMaterialCutsCouple(couple);post->SetMaterialCutsCouple(couple);
        pre->SetPosition({0,0,0});post->SetPosition({0,0,0});pre->SetWeight(1);post->SetWeight(1);
        G4Track track(new G4DynamicParticle(particle,{0,0,1},energy*MeV),0,{0,0,0});
        track.SetParentID(1);track.SetTrackID(1);track.SetWeight(1);
        track.SetTouchableHandle(primary->GetTouchableHandle());track.SetStep(&step);step.SetTrack(&track);
        pre->SetKineticEnergy(energy*MeV);post->SetKineticEnergy(energy*MeV);
        auto* process=processes[proc];process->StartTracking(&track);
        G4ForceCondition initial=NotForced;
        process->PostStepGetPhysicalInteractionLength(track,-1,&initial);
        std::size_t modelIndex=couple->GetIndex();auto* model=process->SelectModelForMaterial(energy*MeV,modelIndex);
        Require(model,"missing native ion model");metadata<<energy/a<<" "<<model->GetName()<<"\n";
        const double rate=process->GetLambda(energy*MeV,couple,std::log(energy*MeV))*mm;
        Require(std::isfinite(rate)&&rate>0,"requested process has no native support");
        double maxResidual=0;int accepted=0,nulls=0;
        for(std::uint64_t attempt=0;accepted<count && attempt<std::uint64_t(count)*10000;++attempt) {
            process->StartTracking(&track);G4ForceCondition condition=NotForced;
            const double distance=process->PostStepGetPhysicalInteractionLength(track,-1,&condition);
            Require(std::isfinite(distance)&&distance<std::numeric_limits<double>::max(),"no native process flight");
            auto* result=dynamic_cast<G4ParticleChangeForLoss*>(process->PostStepDoIt(track,step));
            Require(result,"wrong native change type");
            const double remaining=result->GetProposedKineticEnergy()/MeV,dep=result->GetLocalEnergyDeposit()/MeV;
            const auto direction=result->GetProposedMomentumDirection();const int n=result->GetNumberOfSecondaries();
            if(n==0&&dep==0&&remaining==energy&&direction==G4ThreeVector(0,0,1)) {
                ++nulls;result->Clear();continue; // native integral-method null collision
            }
            double children=0;
            for(int i=0;i<n;++i){const auto* c=result->GetSecondary(i);const int pdg=c->GetDefinition()->GetPDGEncoding();Require(pdg==11||pdg==22,"unsupported child");children+=c->GetKineticEnergy()/MeV;}
            const double residual=std::abs(energy-remaining-dep-children);
            Require(dep>=0&&remaining>=0&&residual<1e-8,"native conditional energy closure");maxResidual=std::max(maxResidual,residual);
            Row r;r.pdg=particle->GetPDGEncoding();r.run=cell;r.event=event++;r.process=proc;r.energy=energy;r.remaining=remaining;r.deposit=dep;r.children=children;r.density=material->GetDensity()/(g/cm3);
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
    G4cout<<"ION_BIRTH_CONDITIONAL_DUMP=PASS events="<<event<<G4endl;
    return true;
}

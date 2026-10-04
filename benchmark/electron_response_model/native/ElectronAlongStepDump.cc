#include "ElectronAlongStepDump.hh"
#include "TsParameterManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Electron.hh"
#include "G4DynamicParticle.hh"
#include "G4ProcessManager.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4VEmModel.hh"
#include "G4VEmFluctuationModel.hh"
#include "G4Material.hh"
#include "G4ProductionCutsTable.hh"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"
#include "carbon/electron_charged_data.hpp"
#include "G4ParticleChangeForLoss.hh"
#include "G4SystemOfUnits.hh"
#include "G4EmParameters.hh"
#include <array>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdint>
namespace {
class TestEngine:public CLHEP::HepRandomEngine {
public:
    std::uint64_t state{};
    double flat()override{state=state*6364136223846793005ULL+1442695040888963407ULL;return (double(state>>12)+.5)*0x1p-52;}
    void flatArray(const int n,double* p)override{for(int i=0;i<n;++i)p[i]=flat();}
    void setSeed(long s,int)override{state=s;}
    void setSeeds(const long* s,int)override{state=*s;}
    void saveStatus(const char[])const override{}void restoreStatus(const char[])override{}void showStatus()const override{}
    std::string name()const override{return "IsolatedElectronLossLCG";}
};
}
ElectronAlongStepDump::ElectronAlongStepDump(TsParameterManager* pm,TsMaterialManager* mm,
    TsGeometryManager* gm,TsScoringManager* sm,TsExtensionManager* em,G4String name,
    G4String quantity,G4String output,G4bool sub):TsVScorer(pm,mm,gm,sm,em,name,quantity,output,sub) {}
void ElectronAlongStepDump::Require(bool ok,const char* why){if(!ok){G4cerr<<"ElectronAlongStepDump audit: "<<why<<G4endl;fPm->AbortSession(1);}}
G4bool ElectronAlongStepDump::ProcessHits(G4Step* source,G4TouchableHistory*) {
    if(done_||source->GetTrack()->GetParentID()!=0)return false;done_=true;
    const auto* couple=source->GetPreStepPoint()->GetMaterialCutsCouple();const auto* material=couple->GetMaterial();
    auto* electron=G4Electron::Electron();G4VEnergyLossProcess* process=nullptr;const auto* list=electron->GetProcessManager()->GetProcessList();
    for(std::size_t i=0;i<list->size();++i)if((*list)[i]->GetProcessName()=="eIoni")process=dynamic_cast<G4VEnergyLossProcess*>((*list)[i]);
    Require(process,"native ionization missing");
    std::ifstream request("charged_operator.txt");std::string path;unsigned samples;request>>path>>samples;Require(bool(request)&&samples>0,"operator request");
    carbon::ChargedResponseData data(path);const auto view=data.view();
    std::ofstream meta("alongstep_options.txt");meta<<std::setprecision(17)<<"native_density="<<material->GetDensity()/(g/cm3)<<"\noperator_density="<<view.density<<"\nlowest_electron_energy="<<G4EmParameters::Instance()->LowestElectronEnergy()<<"\n";meta.close();
    Require(float(view.density)==float(material->GetDensity()/(g/cm3)),"material density identity");
    std::ofstream output("alongstep_checks.csv");output<<std::setprecision(17)<<"cell,energy,length,samples,native_m1,native_m2,portable_m1,portable_m2,native_stops,portable_stops,max_closure\n";
    unsigned cell=0;
    for(double energy:{.00005,.0001,.0001001,.0005,.005,.05,.08,.1,.100001,.4,.65})for(double fraction:{.0001,.005,.01,.1,.5,1.01}) {
        G4Step step;auto* pre=step.GetPreStepPoint();auto* post=step.GetPostStepPoint();
        pre->SetMaterial(const_cast<G4Material*>(material));post->SetMaterial(const_cast<G4Material*>(material));pre->SetMaterialCutsCouple(couple);post->SetMaterialCutsCouple(couple);
        pre->SetWeight(1);post->SetWeight(1);pre->SetKineticEnergy(energy);post->SetKineticEnergy(energy);
        G4Track track(new G4DynamicParticle(electron,{0,0,1},energy),0,{0,0,0});track.SetWeight(1);track.SetParentID(1);track.SetTrackID(1);track.SetTouchableHandle(source->GetTrack()->GetTouchableHandle());track.SetStep(&step);step.SetTrack(&track);
        const double length=std::min(.25,fraction*process->GetRange(energy,couple));step.SetStepLength(length);
        pre->SetPosition({0,0,0});post->SetPosition({0,0,length});
        double nm=0,nm2=0,pm=0,pm2=0,closure=0;unsigned ns=0,ps=0;
        TestEngine native,portable;auto* saved=G4Random::getTheEngine();
        for(unsigned i=0;i<samples;++i){
            native.state=0x1245678abcdULL+std::uint64_t(cell)*0x123456789ULL+i*0x9e3779b97f4a7c15ULL;portable.state=native.state^0xabcdef812345ULL;
            G4Random::setTheEngine(&native);process->StartTracking(&track);G4ForceCondition condition=NotForced;process->PostStepGetPhysicalInteractionLength(track,-1,&condition);
            double safety=100;G4GPILSelection selection=NotCandidateForSelection;process->AlongStepGetPhysicalInteractionLength(track,0,length,safety,&selection);
            auto* result=dynamic_cast<G4ParticleChangeForLoss*>(process->AlongStepDoIt(track,step));Require(result,"native loss change");
            const double dep=result->GetLocalEnergyDeposit(),remaining=result->GetProposedKineticEnergy();Require(result->GetNumberOfSecondaries()==0,"PIXE/subcut unexpectedly active");result->Clear();G4Random::setTheEngine(saved);
            auto uniform=[&](){return portable.flat();};const auto draw=carbon::sample_charged_loss(view,energy,length,uniform);
            Require(draw.valid&&dep>=0&&remaining>=0,"AlongStep state invalid");
            closure=std::max(closure,std::max(std::abs(dep+remaining-energy),std::abs(draw.deposit+draw.remaining-energy)));Require(closure<1e-12,"AlongStep energy closure");
            const double n=dep/energy,p=draw.deposit/energy;nm+=n/samples;nm2+=n*n/samples;pm+=p/samples;pm2+=p*p/samples;ns+=remaining==0;ps+=draw.stopped;
        }
        output<<cell++<<','<<energy<<','<<length<<','<<samples<<','<<nm<<','<<nm2<<','<<pm<<','<<pm2<<','<<ns<<','<<ps<<','<<closure<<'\n';
    }
    Require(bool(output),"AlongStep result output");G4cout<<"ELECTRON_ALONGSTEP_DUMP=PASS"<<G4endl;return true;
}

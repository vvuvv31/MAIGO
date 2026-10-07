#include "ElectronFluctuationDump.hh"
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
#include "carbon/electron_urban_fluctuation.hpp"
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
ElectronFluctuationDump::ElectronFluctuationDump(TsParameterManager* pm,TsMaterialManager* mm,
    TsGeometryManager* gm,TsScoringManager* sm,TsExtensionManager* em,G4String name,
    G4String quantity,G4String output,G4bool sub):TsVScorer(pm,mm,gm,sm,em,name,quantity,output,sub) {}
void ElectronFluctuationDump::Require(bool ok,const char* why){if(!ok){G4cerr<<"ElectronFluctuationDump audit: "<<why<<G4endl;fPm->AbortSession(1);}}
G4bool ElectronFluctuationDump::ProcessHits(G4Step* source,G4TouchableHistory*) {
    if(done_||source->GetTrack()->GetParentID()!=0)return false;
    done_=true;
    const auto* couple=source->GetPreStepPoint()->GetMaterialCutsCouple();const auto* material=couple->GetMaterial();
    auto* electron=G4Electron::Electron();G4VEnergyLossProcess* process=nullptr;const auto* list=electron->GetProcessManager()->GetProcessList();
    for(std::size_t i=0;i<list->size();++i)if((*list)[i]->GetProcessName()=="eIoni")process=dynamic_cast<G4VEnergyLossProcess*>((*list)[i]);
    Require(process,"native ionization missing");auto* ioni=material->GetIonisation();
    const carbon::ElectronUrbanMaterial model{electron_mass_c2,ioni->GetMeanExcitationEnergy(),ioni->GetEnergy0fluct(),ioni->GetEnergy1fluct(),ioni->GetEnergy2fluct(),ioni->GetF1fluct(),ioni->GetF2fluct()};
    std::ofstream constants("urban_material.bin",std::ios::binary);constants.write(reinterpret_cast<const char*>(&model),sizeof(model));
    static_assert(sizeof(model)==56);Require(bool(constants),"material output");
    const double productionCut=(*G4ProductionCutsTable::GetProductionCutsTable()->GetEnergyCutsVector(1))[couple->GetIndex()];
    unsigned samples=50000;std::ifstream request("fluctuation_samples.txt");if(request)request>>samples;Require(samples>0&&samples<=1000000,"sample count");
    std::ofstream output("urban_checks.csv");output<<std::setprecision(17)<<"cell,energy,mean,cut,samples,native_m1,native_m2,native_m3,native_m4,portable_m1,portable_m2,portable_m3,portable_m4,ks,over_energy_native,over_energy_portable,max_draws\n";
    unsigned cell=0;
    for(double energy:{.00005,.0005,.005,.05,.08,.1,.4,.65})for(double fraction:{1e-5,.001,.01,.1,.5}) {
        const double mean=energy*fraction;G4DynamicParticle particle(electron,{0,0,1},energy);
        std::size_t index=couple->GetIndex();auto* nativeModel=process->SelectModelForMaterial(energy,index);
        auto* fluct=nativeModel->GetModelOfFluctuations();Require(fluct&&fluct->GetName()=="UrbanFluc","requires actual native UrbanFluc");
        const double tmax=nativeModel->MaxSecondaryKinEnergy(&particle),cut=std::min(productionCut,tmax);
        Require(cut>0,"invalid native cut");std::vector<double> n(samples),p(samples);std::array<double,4> nm{},pm{};
        unsigned overN=0,overP=0,maxDraws=0;TestEngine native,portable;auto* saved=G4Random::getTheEngine();
        for(unsigned i=0;i<samples;++i){
            native.state=0x435674235aULL+std::uint64_t(cell)*0x123465789ULL+i*0x9e3779b97f4a7c15ULL;
            portable.state=native.state^0x81742345123ULL;
            G4Random::setTheEngine(&native);
            const double nv=fluct->SampleFluctuations(couple,&particle,cut,tmax,.0001,mean);
            G4Random::setTheEngine(saved);auto uniform=[&](){return portable.flat();};
            const auto draw=carbon::sample_electron_urban_loss(model,energy,mean,cut,uniform);
            Require(draw.valid&&std::isfinite(nv)&&nv>=0,"fluctuation sampling state");
            n[i]=nv/mean;p[i]=draw.loss/mean;overN+=nv>energy;overP+=draw.loss>energy;maxDraws=std::max(maxDraws,draw.random_draws);
            double nx=1,px=1;for(unsigned k=0;k<4;++k){nx*=n[i];px*=p[i];nm[k]+=nx/samples;pm[k]+=px/samples;}
        }
        std::sort(n.begin(),n.end());std::sort(p.begin(),p.end());std::size_t a=0,b=0;double ks=0;
        while(a<n.size()||b<p.size()){
            const double next=a==n.size()?p[b]:(b==p.size()?n[a]:std::min(n[a],p[b]));
            while(a<n.size()&&n[a]<=next)++a;while(b<p.size()&&p[b]<=next)++b;
            ks=std::max(ks,std::abs(double(a)-double(b))/samples);
        }
        output<<cell++<<','<<energy<<','<<mean<<','<<cut<<','<<samples;
        for(double x:nm)output<<','<<x;for(double x:pm)output<<','<<x;
        output<<','<<ks<<','<<overN<<','<<overP<<','<<maxDraws<<'\n';
    }
    Require(bool(output),"fluctuation result output");G4cout<<"ELECTRON_FLUCTUATION_DUMP=PASS"<<G4endl;return true;
}

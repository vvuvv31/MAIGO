#include "ElectronElasticDump.hh"
#include "TsParameterManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Electron.hh"
#include "G4ProcessManager.hh"
#include "G4VMultipleScattering.hh"
#include "G4GoudsmitSaundersonMscModel.hh"
#include "G4GoudsmitSaundersonTable.hh"
#include "G4GSMottCorrection.hh"
#include "G4Material.hh"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"
#include "carbon/electron_elastic.hpp"
#include <array>
#include <fstream>
#include <iomanip>
#include <cstdint>
namespace {
class TestEngine:public CLHEP::HepRandomEngine {
public:
    std::uint64_t state{},calls{};
    double flat()override {++calls;state=state*6364136223846793005ULL+1442695040888963407ULL;return (double(state>>12)+.5)*0x1p-52;}
    void flatArray(const int n,double* p)override{for(int i=0;i<n;++i)p[i]=flat();}
    void setSeed(long s,int)override{state=s;calls=0;}
    void setSeeds(const long* s,int)override{setSeed(*s,0);}
    void saveStatus(const char[])const override{}
    void restoreStatus(const char[])override{}
    void showStatus()const override{}
    std::string name()const override{return "IsolatedElasticTestLCG";}
};
}
ElectronElasticDump::ElectronElasticDump(TsParameterManager* pm,TsMaterialManager* mm,
    TsGeometryManager* gm,TsScoringManager* sm,TsExtensionManager* em,G4String name,
    G4String quantity,G4String output,G4bool sub):TsVScorer(pm,mm,gm,sm,em,name,quantity,output,sub) {}
void ElectronElasticDump::Require(bool ok,const char* why) {
    if(!ok){G4cerr<<"ElectronElasticDump audit: "<<why<<G4endl;fPm->AbortSession(1);}
}
G4bool ElectronElasticDump::ProcessHits(G4Step* source,G4TouchableHistory*) {
    if(done_||source->GetTrack()->GetParentID()!=0)return false;done_=true;
    const auto* material=source->GetPreStepPoint()->GetMaterial();const int index=material->GetIndex();
    auto* electron=G4Electron::Electron();G4GoudsmitSaundersonMscModel* gs=nullptr;
    const auto* list=electron->GetProcessManager()->GetProcessList();
    for(std::size_t i=0;i<list->size();++i)if(auto* m=dynamic_cast<G4VMultipleScattering*>((*list)[i]))
        gs=dynamic_cast<G4GoudsmitSaundersonMscModel*>(m->GetModelByIndex(0));
    Require(gs&&gs->GetOptionMottCorrection()&&!gs->GetOptionPWACorrection(),"requires actual native GS/Mott");
    auto* table=gs->GetGSTable();G4GSMottCorrection mott(true);mott.Initialise();
    std::array<double,31*38> rows{};
    const double mass=electron_mass_c2,midp=.1*(.1+2*mass),midbeta=midp/(midp+mass*mass);
    auto reject=[&](unsigned e,double y){int ei=e,di=0;return mott.GetMottRejectionValue(std::log(.1),midbeta,0,1-2*y*y,index,ei,di);};
    for(unsigned i=0;i<31;++i){
        const double beta=midbeta+(i>15?double(i-15):0)*(.9999-midbeta)/15;
        const double e=i<=15?.001*std::pow(100.,double(i)/15):mass*(1/std::sqrt(1-beta)-1);
        const double pt2=e*(e+2*mass),b2=pt2/(pt2+mass*mass);
        double a,b,c;table->GetMottCorrectionFactors(std::log(e),b2,index,a,b,c);
        auto* row=rows.data()+38*i;row[0]=a;row[1]=b;row[2]=c;
        for(unsigned j=0;j<30;++j)row[3+j]=reject(i,double(j)/31);
        // Native tabulated linear knot30 and the final fitted cubic are not
        // exactly continuous. Recover each side independently; querying the
        // shared boundary would alias the left endpoint to the cubic value.
        row[33]=2*reject(i,29.5/31)-row[32];
        double inner[4];for(unsigned j=0;j<4;++j)inner[j]=reject(i,(30+(j+.5)/4)/31);
        for(unsigned j=0;j<4;++j){const double t=double(j)/3;double value=0;
            for(unsigned k=0;k<4;++k){double weight=1;for(unsigned l=0;l<4;++l)if(k!=l)weight*=(t-(l+.5)/4)/(double(k)-l)*4;value+=weight*inner[k];}
            row[34+j]=value;
        }
    }
    carbon::ElectronElasticView model{rows.data(),mass,table->GetMoliereBc(index),table->GetMoliereXc2(index)};
    double maxFunctionError=0;unsigned worstEnergy=0;double worstY=0,worstNative=0,worstPortable=0;
    for(unsigned i=0;i<31;++i)for(unsigned j=0;j<4096;++j){
        const double y=(j+.5)/4096,cosine=1-2*y*y;
        const double native=reject(i,y),portable=carbon::elastic_mott_rejection(model,i,cosine),error=std::abs(portable-native);
        if(error>maxFunctionError){maxFunctionError=error;worstEnergy=i;worstY=y;worstNative=native;worstPortable=portable;}
    }
    G4cout<<std::setprecision(17)<<"MOTT_FUNCTION_CHECK max_error="<<maxFunctionError<<" index="<<worstEnergy<<" y="<<worstY<<" native="<<worstNative<<" portable="<<worstPortable<<G4endl;
    Require(maxFunctionError<1e-10,"compact Mott function mismatch");
    unsigned samples=100000;std::ifstream request("elastic_samples.txt");if(request)request>>samples;
    Require(samples>0&&samples<=10000000,"invalid sample count");
    TestEngine native,portable,sourceEngine;sourceEngine.state=714015ULL;
    auto* saved=G4Random::getTheEngine();double maxCosineError=0,maxScreenError=0;std::uint64_t randomCalls=0;unsigned maxAttempts=0;
    std::ofstream checks("elastic_checks.bin",std::ios::binary);
    for(unsigned i=0;i<samples;++i){
        const double energy=std::exp(std::log(1e-5)+sourceEngine.flat()*std::log(.65/1e-5));
        const std::uint64_t seed=sourceEngine.state;native.state=portable.state=seed;native.calls=portable.calls=0;
        const double p2=energy*(energy+2*mass),beta=p2/(p2+mass*mass);
        double a,b,c;table->GetMottCorrectionFactors(std::log(energy),beta,index,a,b,c);
        const double screening=model.xc2/(4*p2*model.bc)*a;
        maxScreenError=std::max(maxScreenError,std::abs(carbon::elastic_screening(model,energy)/screening-1));
        G4Random::setTheEngine(&native);
        const double cost=table->SingleScattering(1,screening,std::log(energy),beta,index);
        G4Random::setTheEngine(saved);
        auto uniform=[&](){return portable.flat();};
        const auto result=carbon::sample_electron_elastic(model,energy,uniform);
        Require(result.valid&&native.calls==portable.calls&&native.state==portable.state,"paired native elastic random stream mismatch");
        maxCosineError=std::max(maxCosineError,std::abs(cost-result.cosine));maxAttempts=std::max(maxAttempts,result.attempts);randomCalls+=native.calls;
        // Native reference values and seed can be independently replayed on GPU.
        checks.write(reinterpret_cast<const char*>(&seed),8);checks.write(reinterpret_cast<const char*>(&energy),8);
        checks.write(reinterpret_cast<const char*>(&cost),8);checks.write(reinterpret_cast<const char*>(&native.calls),8);
    }
    Require(maxCosineError<1e-10&&maxScreenError<1e-10,"paired native elastic state mismatch");
    std::ofstream binary("elastic_kernel.bin",std::ios::binary);const std::uint32_t version=1,count=31;
    binary.write("MAIELS01",8);binary.write(reinterpret_cast<const char*>(&version),4);binary.write(reinterpret_cast<const char*>(&count),4);
    for(double value:{model.mass,model.bc,model.xc2})binary.write(reinterpret_cast<const char*>(&value),8);
    binary.write(reinterpret_cast<const char*>(rows.data()),rows.size()*8);
    Require(bool(binary)&&bool(checks),"elastic output failure");
    std::ofstream audit("elastic_audit.json");
    audit<<std::setprecision(17)<<"{\"samples\":"<<samples<<",\"random_calls\":"<<randomCalls<<",\"max_attempts\":"<<maxAttempts
        <<",\"maximum_function_error\":"<<maxFunctionError<<",\"maximum_screening_relative_error\":"<<maxScreenError
        <<",\"maximum_cosine_error\":"<<maxCosineError<<",\"paired_random_streams_equal\":true}\n";
    G4cout<<"ELECTRON_ELASTIC_DUMP=PASS"<<G4endl;return true;
}

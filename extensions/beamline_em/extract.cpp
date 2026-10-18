// Offline Geant4 11.3.2 full electronic response. No production cut enters
// either the mean loss or the variance (tcut=Tmax). MSC tables are separate.
#include <bits/stdc++.h>
#define private public
#include "G4EmCalculator.hh"
#undef private
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4RunManager.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4EmParameters.hh"
#include "G4NistManager.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4Material.hh"
#include "G4IonisParamMat.hh"
#include "G4DynamicParticle.hh"
#include "G4IonFluctuations.hh"
#include "G4UniversalFluctuation.hh"
#include "G4VEmModel.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4Version.hh"
struct Material {float density,excitation,e0;};
struct Species {uint32_t z,a;float mass;};
struct Record {uint32_t offset,count;};
struct Node {float energy,stopping,ion_variance,universal_variance;};
class Detector final:public G4VUserDetectorConstruction {
 G4VPhysicalVolume* Construct() override {
  auto* n=G4NistManager::Instance();n->FindOrBuildMaterial("G4_AIR");
  auto* l=new G4LogicalVolume(new G4Box("world",m,m,m),n->FindOrBuildMaterial("G4_Cu"),"world");
  return new G4PVPlacement(nullptr,{},l,"world",nullptr,false,0);
 }
};
class Physics final:public G4VModularPhysicsList {
 public:Physics(){RegisterPhysics(new G4EmStandardPhysics_option4(0));
  G4EmParameters::Instance()->SetMaxEnergy(1*TeV);SetDefaultCutValue(.05*mm);}
};
int main(int argc,char**argv){try {
 if(argc!=5)throw std::runtime_error("extract species.txt full.bin heldout.csv nodes");
 if(G4VERSION_NUMBER!=1132)throw std::runtime_error("Requires Geant4 11.3.2");
 std::ifstream in(argv[1]);std::vector<Species> species;unsigned z,a;
 while(in>>z>>a)species.push_back({z,a,0});if(species.empty())throw std::runtime_error("Empty species");
 const int count=std::stoi(argv[4]);if(count<4097)throw std::runtime_error("At least 4097 nodes required");
 auto rm=std::make_unique<G4RunManager>();rm->SetUserInitialization(new Detector);rm->SetUserInitialization(new Physics);rm->Initialize();rm->BeamOn(0);
 auto* table=G4ParticleTable::GetParticleTable();std::vector<G4ParticleDefinition*> particles;
 for(auto& s:species){const char* light=s.z==1&&s.a==1?"proton":s.z==1&&s.a==2?"deuteron":s.z==1&&s.a==3?"triton":s.z==2&&s.a==3?"He3":s.z==2&&s.a==4?"alpha":nullptr;
  auto* p=light?table->FindParticle(light):table->GetIonTable()->GetIon(s.z,s.a,0.);
  if(!p)throw std::runtime_error("Missing particle");s.mass=p->GetPDGMass()/MeV;particles.push_back(p);}
 std::vector<Material> materials;std::vector<Record> records;std::vector<Node> nodes;
 std::ofstream held(argv[3]);held<<std::setprecision(17)<<"material,projectile,energy,stopping,ion_variance,universal_variance\n";
 G4EmCalculator calc;
 for(auto name:{"G4_Cu","G4_AIR"}){
  auto* mat=G4NistManager::Instance()->FindOrBuildMaterial(name);
  const unsigned mi=materials.size();materials.push_back({float(mat->GetDensity()/(g/cm3)),float(mat->GetIonisation()->GetMeanExcitationEnergy()/MeV),float(mat->GetIonisation()->GetEnergy0fluct()/MeV)});
  for(unsigned pi=0;pi<particles.size();++pi){auto*p=particles[pi];G4IonFluctuations ion;G4UniversalFluctuation uni;
   ion.InitialiseMe(p);uni.InitialiseMe(p);
   auto evaluate=[&](double energy){
    double sp=calc.ComputeElectronicDEDX(energy*MeV,p,mat)/(MeV/mm);
    if(!calc.currentModel)throw std::runtime_error("Missing native EM model");
    double q2=calc.currentModel->GetChargeSquareRatio(p,mat,energy*MeV);
    if(species[pi].z==1 && species[pi].a==1)q2=1;
    ion.SetParticleAndCharge(p,q2);uni.SetParticleAndCharge(p,q2);
    G4DynamicParticle dp(p,{0,0,1},energy*MeV);
    const double tau=energy/species[pi].mass,re=CLHEP::electron_mass_c2/p->GetPDGMass();
    double tmax=2*CLHEP::electron_mass_c2*tau*(tau+2)/(1+2*(tau+1)*re+re*re);
    double vi=ion.Dispersion(mat,&dp,tmax,tmax,mm)/(MeV*MeV),vu=uni.Dispersion(mat,&dp,tmax,tmax,mm)/(MeV*MeV);
    if(!(sp>0&&vi>=0&&vu>=0&&std::isfinite(sp+vi+vu)))throw std::runtime_error("Invalid full response Z="+std::to_string(species[pi].z)+" A="+std::to_string(species[pi].a)+" E="+std::to_string(energy));
    return std::array<double,3>{sp,vi,vu};};
   std::vector<double> grid;const double lo=1e-6*species[pi].a,hi=6500.*species[pi].a;
   for(int i=0;i<count;++i)grid.push_back(std::exp(std::log(lo)+std::log(hi/lo)*i/(count-1)));
   for(double eu:{.001,.01,.1,1.,2.,10.,50.,100.,200.,500.,1000.})grid.push_back(eu*species[pi].a);
   // Resolve native ion correction/model boundaries on each side.
   for(double boundary:{2.*p->GetPDGMass()/CLHEP::proton_mass_c2,10.*p->GetPDGMass()/CLHEP::proton_mass_c2})
    for(double f:{1.-1e-5,1.,1.+1e-5})grid.push_back(boundary*f);
   for(auto& e:grid)e=float(e);
   std::sort(grid.begin(),grid.end());grid.erase(std::unique(grid.begin(),grid.end()),grid.end());
   const unsigned start=nodes.size();
   auto store=[&](double e,const auto& r){nodes.push_back({float(e),float(r[0]),float(r[1]),float(r[2])});};
   std::function<void(double,std::array<double,3>,double,std::array<double,3>,int)> refine;
   refine=[&](double l,std::array<double,3> vl,double r,std::array<double,3> vr,int depth){
    const double e=float(std::sqrt(l*r));
    if(e<=l||e>=r){store(l,vl);return;}
    const auto vm=evaluate(e);const double w=(e-l)/(r-l);double err=0;
    for(int k=0;k<3;++k)err=std::max(err,std::abs((vl[k]+w*(vr[k]-vl[k]))/std::max(vm[k],1e-30)-1));
    if(err>1e-4 && depth<24){refine(l,vl,e,vm,depth+1);refine(e,vm,r,vr,depth+1);}
    else store(l,vl);
   };
   auto left=evaluate(grid[0]);
   for(unsigned j=1;j<grid.size();++j){auto right=evaluate(grid[j]);refine(grid[j-1],left,grid[j],right,0);left=right;}
   store(grid.back(),left);records.push_back({start,unsigned(nodes.size()-start)});
   for(int i=0;i+1<count;i+=8){double e=std::exp(std::log(lo)+std::log(hi/lo)*(i+.37)/(count-1));auto r=evaluate(e);held<<mi<<','<<pi<<','<<e<<','<<r[0]<<','<<r[1]<<','<<r[2]<<'\n';}
   std::cout<<"FULL_RESPONSE "<<name<<" Z="<<species[pi].z<<" A="<<species[pi].a<<std::endl;
  }
 }
 std::ofstream out(argv[2],std::ios::binary);out.write("BLEM0001",8);
 uint32_t h[]={1,unsigned(materials.size()),unsigned(species.size()),unsigned(nodes.size())};out.write((char*)h,sizeof(h));
 auto write=[&](auto& v){out.write((char*)v.data(),v.size()*sizeof(v[0]));};write(materials);write(species);write(records);write(nodes);
 if(!out||!held)throw std::runtime_error("Write failed");
 std::cout<<"PASS full electronic response, species="<<species.size()<<" nodes="<<nodes.size()<<'\n';
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

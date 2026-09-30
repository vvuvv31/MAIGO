// Reuse the exact neutron extraction material/physics initialization.
#define main neutron_collision_extractor_main
#include "extract.cc"
#undef main
#include "G4EmCalculator.hh"
#include "G4IonTable.hh"
#include "G4Proton.hh"
#include "G4Deuteron.hh"
#include "G4Triton.hh"
#include "G4He3.hh"
#include "G4Alpha.hh"
int main(int argc,char** argv){
 std::signal(SIGSEGV, +[](int){void* frames[40];int n=backtrace(frames,40);backtrace_symbols_fd(frames,n,2);_exit(139);});
 std::signal(SIGABRT, +[](int){void* frames[40];int n=backtrace(frames,40);backtrace_symbols_fd(frames,n,2);_exit(134);});
 try{
  if(argc!=4)throw std::runtime_error("usage: extract_neutron_recoil OUTPUT MATERIAL SPECIES");
  std::filesystem::path out=argv[1];material_name=argv[2];
  if(material_name!="Water_75eV"&&material_name!="G4_Cu"&&material_name!="G4_AIR")throw std::runtime_error("unknown material");
  if(std::filesystem::exists(out))throw std::runtime_error("output exists");
  std::ifstream spec(argv[3]);std::vector<std::pair<int,int>> ions;int z,a;
  while(spec>>z>>a){if(z<1||a<z)throw std::runtime_error("invalid ZA");ions.emplace_back(z,a);}
  if(!spec.eof()||ions.empty())throw std::runtime_error("invalid species input");
  CLHEP::HepRandom::setTheSeed(92717001);
  std::cerr<<"STAGE manager\n";
  auto* run=new G4RunManager;run->SetUserInitialization(new Detector);run->SetUserInitialization(new Physics);run->SetUserAction(new Generator);run->SetUserAction(new Kill);run->Initialize();
  std::cerr<<"STAGE initialized\n";
  // Initialize native process tables/model couples before calculator queries.
  run->BeamOn(1);
  std::cerr<<"STAGE physics tables ready\n";
  std::filesystem::create_directories(out);
  std::ofstream table(out/"stopping.csv"),meta(out/"provenance.txt");table<<std::setprecision(17);meta<<std::setprecision(17);
  meta<<"geant4 "<<G4Version<<"\nmaterial "<<material_name<<"\ndensity_g_cm3 "<<material->GetDensity()/(g/cm3)<<"\nexcitation_eV "<<material->GetIonisation()->GetMeanExcitationEnergy()/eV<<"\nmethod ComputeElectronicDEDX_plus_ComputeNuclearDEDX\n";
  table<<"atomic_number,mass_number,energy_MeVu,stopping_power_MeV_per_mm,restricted_stopping_power_MeV_per_mm,nuclear_stopping_power_MeV_per_mm\n";
  std::size_t rows=0;
  { // Destroy calculator-owned model caches before the native run manager.
  G4EmCalculator calculator;
  for(auto [Z,A]:ions){
   std::cerr<<"ION "<<Z<<' '<<A<<'\n';
   G4ParticleDefinition* ion=nullptr;
   if(Z==1&&A==1)ion=G4Proton::Definition();else if(Z==1&&A==2)ion=G4Deuteron::Definition();else if(Z==1&&A==3)ion=G4Triton::Definition();else if(Z==2&&A==3)ion=G4He3::Definition();else if(Z==2&&A==4)ion=G4Alpha::Definition();else ion=G4IonTable::GetIonTable()->GetIon(Z,A,0.);
   if(!ion)throw std::runtime_error("missing native ion");
   meta<<"ion "<<Z<<' '<<A<<" mass_MeV "<<ion->GetPDGMass()/MeV<<'\n';
   for(int i=0;i<=768;++i){
    double eu=std::exp(std::log(1.e-9)+i*(std::log(400.)-std::log(1.e-9))/768.);
    double electronic=calculator.ComputeElectronicDEDX(eu*A*MeV,ion,material)/(MeV/mm);
    double nuclear=calculator.ComputeNuclearDEDX(eu*A*MeV,ion,material)/(MeV/mm);
    if(!std::isfinite(electronic+nuclear)||electronic<0||nuclear<0)throw std::runtime_error("invalid native stopping");
    table<<Z<<','<<A<<','<<eu<<','<<electronic<<','<<electronic<<','<<nuclear<<'\n';++rows;
   }
  }
  }
  table.flush();meta.flush();if(!table||!meta)throw std::runtime_error("write failure");
  std::ofstream(out/"COMPLETE")<<rows<<'\n';delete run;return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

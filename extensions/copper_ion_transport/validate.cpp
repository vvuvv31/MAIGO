// Offline comparison against native Geant4; never linked into the GPU runtime.
#include "G4NistManager.hh"
#include "G4IonisParamMat.hh"
#include "G4IonTable.hh"
#include "G4ParticleTable.hh"
#include "G4IonFluctuations.hh"
#include "G4UniversalFluctuation.hh"
#include "G4DynamicParticle.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ProductionCuts.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "carbon/copper_ion_fluctuation.hpp"
#include "carbon/urban_mcs_package.hpp"
#include "Randomize.hh"
#include "native_setup.hpp"
#include <random>
#include <iostream>
#include <stdexcept>
#include <array>
#include <iomanip>
struct Moments {
 double sum{},square{};unsigned n{};
 void add(double x){sum+=x;square+=x*x;++n;}
 double mean()const{return sum/n;}
 double variance()const{return square/n-mean()*mean();}
};
int main(int argc,char** argv) {
 if(argc!=2)throw std::runtime_error("validate urban_package.bin");
 auto* rm=make_copper_native_setup();
 auto* table=G4ParticleTable::GetParticleTable();
 auto* material=G4NistManager::Instance()->FindOrBuildMaterial("G4_Cu");
 G4ProductionCuts cuts;cuts.SetProductionCut(.05*mm);
 G4MaterialCutsCouple couple(material,&cuts);
 const auto bank=carbon::UrbanMcsPackage::load(argv[1]);const auto view=bank.view();
 std::mt19937 generator(70917);auto uniform=[&](){return (generator()>>8)*0x1p-24f;};
 CLHEP::HepRandom::setTheSeed(70918);
 std::cout<<std::setprecision(10);
 for(auto za:std::array<std::array<int,2>,7>{{{1,1},{1,2},{2,4},{3,7},{6,12},{8,16},{22,48}}}) {
  const int z=za[0],a=za[1];
  auto* ion=z==1?table->FindParticle(a==1?"proton":"deuteron"):z==2?table->FindParticle("alpha"):table->GetIonTable()->GetIon(z,a,0);
  if(!ion)throw std::runtime_error("Missing native ion");
  for(float eu:{1.f,10.f,50.f,100.f,250.f})for(float length:{.001f,.1f}) {
   const float energy=eu*a,mass=ion->GetPDGMass()/MeV;
   const auto context=view.at(view.material(-2,8.96f),view.projectile(z,a),energy);
   if(!context.valid()) { std::cerr<<"OUTSIDE_BANK z="<<z<<" a="<<a<<" E/u="<<eu<<"\n"; continue; }
   const float path=std::min(length,.1f*context.range(energy));
   const float mean=context.dedx(energy)*path;
   const float tau=energy/mass,ratio=.51099891f/mass;
   const float beta2=tau*(tau+2)/((tau+1)*(tau+1));
   const float tmax=2*.51099891f*tau*(tau+2)/(1+2*(tau+1)*ratio+ratio*ratio);
   const float cut=std::min(tmax,carbon::copper_cut_MeV);
   const float coefficient=CLHEP::twopi_mc2_rcl2/(MeV*MeV*mm*mm)*
       carbon::copper_electron_density*path*context.factor;
   const float universal=(tmax/beta2-.5f*cut)*coefficient;
   const float factor=carbon::copper_ion_dispersion_factor(energy,mass,z,context.factor,beta2);
   const float correction=1+(factor-1)*2*.51099891f*beta2/(tmax*(1-beta2));
   const float variance=universal*((correction>.01f&&factor>.01f)?correction:1);
   const float compare_variance=(z==1&&a==1)?universal:variance;
   G4DynamicParticle particle(ion,{0,0,1},energy*MeV);
   G4IonFluctuations ion_native;G4UniversalFluctuation proton_native;
   G4VEmFluctuationModel& native=(z==1&&a==1)?static_cast<G4VEmFluctuationModel&>(proton_native):static_cast<G4VEmFluctuationModel&>(ion_native);
   native.SetParticleAndCharge(ion,context.factor);
   const double native_var=native.Dispersion(material,&particle,cut*MeV,tmax*MeV,path*mm)/(MeV*MeV);
   if(std::abs(compare_variance/native_var-1)>5e-4)throw std::runtime_error("Native dispersion mismatch");
   carbon::RestrictedFluctuationInput<float> input{energy,mass,mean,variance,universal,cut,tmax,
       carbon::copper_excitation_MeV,carbon::copper_e0_MeV};
   carbon::RestrictedFluctuationSampler<float,decltype(uniform)> sampler(uniform);
   Moments reference,candidate,total;
   constexpr unsigned draws=100000;
   for(unsigned i=0;i<draws;++i) {
    reference.add(native.SampleFluctuations(&couple,&particle,cut*MeV,tmax*MeV,path*mm,mean*MeV)/MeV);
    const auto draw=sampler.sample(input,!(z==1&&a==1),float(z));
    const auto full=carbon::copper_ion_total_fluctuation(energy,mass,z,a,context.factor,mean,path,8.96f,uniform);
    if(!draw.valid||!full.valid||!std::isfinite(full.loss))throw std::runtime_error("Invalid draw");
    candidate.add(draw.loss);total.add(full.loss);
   }
   // Independently integrate the native delta differential cross section
   // in log-transfer coordinates, including its form-factor null collisions.
   double delta_mean=0,delta_second=0,delta_proposals=0;
   if(tmax>cut) {
    const auto identity=std::find_if(std::begin(carbon::copper_ion_identities),
        std::end(carbon::copper_ion_identities),[&](auto i){return i.z==z&&i.a==a;});
    const double ff=2*.51099891*std::pow(double(z),4./7.)/(842.6*842.6);
    const double total2=double(energy+mass)*(energy+mass);
    const double h=std::log(double(tmax)/cut)/2000;
    for(unsigned j=0;j<=2000;++j) {
     const double e=cut*std::exp(j*h),spin_term=identity->spin>0?.5*e*e/total2:0;
     const double f=1-beta2*e/tmax+spin_term;
     double accept=1;
     if(ff*e>1e-6) {
      accept=1/((1+ff*e)*(1+ff*e));
      if(identity->spin>0) {
       const double x2=.5*.51099891*e/(double(mass)*mass);
       accept*=1+identity->magnetic_moment2*(x2-spin_term/f)/(1+x2);
      }
     }
     const double weight=(j==0||j==2000)?1:(j%2?4:2);
     const double rate=coefficient/beta2*f;
     delta_mean+=weight*rate*accept;
     delta_second+=weight*rate*accept*e;
     delta_proposals+=weight*rate/e;
    }
    delta_mean*=h/3;delta_second*=h/3;delta_proposals*=h/3;
   }
   const double total_error=std::abs(total.mean()-reference.mean()-delta_mean);
   const double total_se=std::sqrt((total.variance()+reference.variance())/draws);
   if(total_error>8*total_se+1e-6)throw std::runtime_error("Native integrated total mean mismatch");
   const double expected_total_var=reference.variance()+delta_second;
   if(std::abs(total.variance()/expected_total_var-1)>.15)
       throw std::runtime_error("Native integrated total variance mismatch");
   const double mean_error=std::abs(reference.mean()-candidate.mean());
   const double standard_error=std::sqrt((reference.variance()+candidate.variance())/draws);
   const double var_error=std::abs(candidate.variance()/reference.variance()-1);
   if(mean_error>8*standard_error+1e-6 || var_error>.12)throw std::runtime_error("Native sample moment mismatch");
   if(total.mean()+8*std::sqrt(total.variance()/draws)<candidate.mean())throw std::runtime_error("Negative delta mean");
   std::cout<<"{\"z\":"<<z<<",\"a\":"<<a<<",\"energy_per_u\":"<<eu<<",\"path_mm\":"<<path
       <<",\"dispersion_relative_error\":"<<compare_variance/native_var-1<<",\"restricted_mean_native\":"<<reference.mean()
       <<",\"restricted_mean_port\":"<<candidate.mean()<<",\"variance_relative_error\":"<<candidate.variance()/reference.variance()-1
       <<",\"delta_proposals\":"<<delta_proposals<<",\"expected_delta_mean\":"<<delta_mean
       <<",\"total_mean\":"<<total.mean()<<",\"draws\":"<<draws<<"}\n";
  }
 }
 delete rm;
 std::cerr<<"PASS: native dispersion/moments, ion delta draws\n";
}

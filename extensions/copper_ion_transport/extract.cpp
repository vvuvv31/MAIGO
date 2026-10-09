// Extract species-specific Cu elastic events and immutable Geant4 constants.
#include "G4RunManager.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Material.hh"
#include "G4IonisParamMat.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4HadronicProcess.hh"
#include "G4CrossSectionDataStore.hh"
#include "G4EnergyRangeManager.hh"
#include "G4HadProjectile.hh"
#include "G4Nucleus.hh"
#include "G4NucleiProperties.hh"
#include "G4DynamicParticle.hh"
#include "G4RToEConvForElectron.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4IonINCLXXPhysics.hh"
#include "QGSP_BIC_HP.hh"
#include "Randomize.hh"
#include <fstream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include "native_setup.hpp"

int main(int argc,char** argv) {
 if(argc!=4)throw std::runtime_error("extract species.txt elastic.csv constants.hpp");
 std::ifstream input(argv[1]);std::vector<std::pair<int,int>> species;
 int z,a;while(input>>z>>a)species.emplace_back(z,a);
 if(species.empty())throw std::runtime_error("Empty species list");
 auto* rm=make_copper_native_setup();
 auto* mat=G4NistManager::Instance()->FindOrBuildMaterial("G4_Cu");
 G4RToEConvForElectron converter;
 std::ofstream constants(argv[3]);constants<<std::setprecision(17)
  <<"#pragma once\n// Generated from Geant4 11.3.2 G4_Cu; see extensions/copper_ion_transport.\nnamespace carbon {\n"
  <<"inline constexpr float copper_cut_MeV="<<converter.Convert(.05*mm,mat)/MeV<<"f;\n"
  <<"inline constexpr float copper_excitation_MeV="<<mat->GetIonisation()->GetMeanExcitationEnergy()/MeV<<"f;\n"
  <<"inline constexpr float copper_e0_MeV="<<mat->GetIonisation()->GetEnergy0fluct()/MeV<<"f;\n"
  <<"inline constexpr float copper_fermi_MeV="<<mat->GetIonisation()->GetFermiEnergy()/MeV<<"f;\n"
  <<"inline constexpr float copper_electron_density="<<mat->GetElectronDensity()*mm*mm*mm<<"f;\n"
  <<"struct CopperIonIdentity {int z,a;float spin,magnetic_moment2;};\n"
  <<"inline constexpr CopperIonIdentity copper_ion_identities[]={\n";
 std::ofstream csv(argv[2]);csv<<std::setprecision(17)
  <<"atomic_number,mass_number,energy_MeV_per_u,macroscopic_cross_section_per_mm,transfer_fraction,target_a,target_mass_MeV\n";
 CLHEP::HepRandom::setTheSeed(10070917);
 for(auto [pz,pa]:species) {
  const char* light=pz==1&&pa==1?"proton":pz==1&&pa==2?"deuteron":pz==1&&pa==3?"triton":pz==2&&pa==3?"He3":pz==2&&pa==4?"alpha":nullptr;
  auto* table=G4ParticleTable::GetParticleTable();
  auto* ion=light?table->FindParticle(light):table->GetIonTable()->GetIon(pz,pa,0.);
  if(!ion||!ion->GetProcessManager())throw std::runtime_error("Missing ion process manager");
  double magnetic=ion->GetPDGMagneticMoment()*ion->GetPDGMass()/(.5*eplus*hbar_Planck*c_squared);
  constants<<"{"<<pz<<","<<pa<<","<<std::scientific<<ion->GetPDGSpin()<<"f,"<<magnetic*magnetic-1<<"f},\n";
  G4HadronicProcess* process=nullptr;int count=0;
  auto* pv=ion->GetProcessManager()->GetProcessList();
  for(std::size_t i=0;i<pv->size();++i)
   if(auto* had=dynamic_cast<G4HadronicProcess*>((*pv)[i]);had&&had->GetProcessSubType()==fHadronElastic){process=had;++count;}
  if(count!=1)throw std::runtime_error("Exactly one elastic process required for each ion");
  G4EnergyRangeManager manager;for(auto* model:process->GetHadronicInteractionList())manager.RegisterMe(model);
  auto* xs=process->GetCrossSectionDataStore();
  for(int k=0;k<=100;++k) {
   const double energy=.001*std::pow(10.,k/15.); // through 4.64 GeV/u
   G4DynamicParticle dp(ion,{0,0,1},energy*pa*MeV);G4HadProjectile hp(dp);
   const double macro=xs->ComputeCrossSection(&dp,mat)*mm;
   if(!std::isfinite(macro)||macro<0)throw std::runtime_error("Invalid elastic rate");
   for(int n=0;n<128;++n) {
    double fraction=0,m2=G4NucleiProperties::GetNuclearMass(63,29)/MeV;int ta=63;
    if(macro>0) {
     G4Nucleus nucleus;auto* element=xs->SampleZandA(&dp,mat,nucleus);ta=nucleus.GetA_asInt();
     m2=G4NucleiProperties::GetNuclearMass(ta,29)/MeV;
     auto* model=manager.GetHadronicInteraction(hp,nucleus,mat,element);
     if(!model)throw std::runtime_error("Missing applicable Cu elastic model");
     auto* fs=model->ApplyYourself(hp,nucleus);
     const double T=energy*pa,m1=ion->GetPDGMass()/MeV;
     const double maximum=2*m2*T*(T+2*m1)/((m1+m2)*(m1+m2)+2*m2*T);
     fraction=(T-fs->GetEnergyChange()/MeV)/maximum;
     if(!std::isfinite(fraction)||fraction < -1e-7||fraction>1+1e-5)throw std::runtime_error("Non two-body elastic final state");
     fraction=std::clamp(fraction,0.,1.);fs->Clear();
    }
    csv<<pz<<','<<pa<<','<<energy<<','<<macro<<','<<fraction<<','<<ta<<','<<m2<<'\n';
   }
  }
 }
 constants<<"};\n}\n";
 if(!csv||!constants)throw std::runtime_error("Output write failed");
 delete rm;
}

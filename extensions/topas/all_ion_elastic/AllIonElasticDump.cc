// Scorer for AllIonElasticDump
// Extracts Geant4/TOPAS C12 ELASTIC cross sections across all 25 Schneider sections
// with strict process attachment verification and complete provenance recording.

#include "AllIonElasticDump.hh"

#include "G4Element.hh"
#include "G4ElementVector.hh"
#include "G4Exception.hh"
#include "G4HadronicProcess.hh"
#include "G4HadronicProcessStore.hh"
#include "G4IonTable.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4TouchableHistory.hh"
#include "G4UIcommand.hh"
#include "G4VProcess.hh"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::array<const char*, 13> kCanonicalElements = {
    "Hydrogen", "Carbon", "Nitrogen", "Oxygen",
    "Magnesium", "Phosphorus", "Sulfur", "Chlorine",
    "Argon", "Calcium", "Sodium", "Potassium", "Titanium"
};

constexpr std::array<int, 13> kCanonicalZ = {
    1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22
};

struct SectionProbe {
    int section_id;
    int hu_min;
    int hu_max;
    int rep_hu;
};

constexpr std::array<SectionProbe, 25> kSectionProbes = {{
    {0,  -1000, -950, -975},
    {1,  -950,  -120, -535},
    {2,  -120,  -83,  -102},
    {3,  -83,   -53,  -68},
    {4,  -53,   -23,  -38},
    {5,  -23,   7,    -8},
    {6,  7,     18,   12},
    {7,  18,    80,   49},
    {8,  80,    120,  100},
    {9,  120,   200,  160},
    {10, 200,   300,  250},
    {11, 300,   400,  350},
    {12, 400,   500,  450},
    {13, 500,   600,  550},
    {14, 600,   700,  650},
    {15, 700,   800,  750},
    {16, 800,   900,  850},
    {17, 900,   1000, 950},
    {18, 1000,  1100, 1050},
    {19, 1100,  1200, 1150},
    {20, 1200,  1300, 1250},
    {21, 1300,  1400, 1350},
    {22, 1400,  1500, 1450},
    {23, 1500,  2995, 2247},
    {24, 2995,  2996, 2995}
}};

G4String MaterialNameFromHU(int hu) {
    if (hu < 0) {
        return "PatientTissueFromHUNegative" + G4UIcommand::ConvertToString(-hu);
    }
    return "PatientTissueFromHU" + G4UIcommand::ConvertToString(hu);
}

}  // namespace

AllIonElasticDump::AllIonElasticDump(
    TsParameterManager* parameter_manager,
    TsMaterialManager* material_manager,
    TsGeometryManager* geometry_manager,
    TsScoringManager* scoring_manager,
    TsExtensionManager* extension_manager,
    G4String scorer_name,
    G4String quantity,
    G4String output_file,
    G4bool is_sub_scorer)
    : TsVScorer(parameter_manager, material_manager, geometry_manager,
                scoring_manager, extension_manager, scorer_name, quantity,
                output_file, is_sub_scorer) {
    output_json_path_ = output_file;
    if (output_json_path_.empty()) {
        output_json_path_ = "topas-c12-schneider-elastic-xs.json";
    }

    if (fPm->ParameterExists(GetFullParmName("OutputCsvFile"))) {
        output_csv_path_ = fPm->GetStringParameter(GetFullParmName("OutputCsvFile"));
    } else {
        output_csv_path_ = "c12_schneider_elastic_cross_sections_geant4_11_3_2.csv";
    }

    if (fPm->ParameterExists(GetFullParmName("MinEnergyMeVu"))) {
        min_energy_mevu_ = fPm->GetDoubleParameter(GetFullParmName("MinEnergyMeVu"), "Energy") / MeV;
    }
    if (fPm->ParameterExists(GetFullParmName("MaxEnergyMeVu"))) {
        max_energy_mevu_ = fPm->GetDoubleParameter(GetFullParmName("MaxEnergyMeVu"), "Energy") / MeV;
    }
    if (fPm->ParameterExists(GetFullParmName("EnergyStepMeVu"))) {
        energy_step_mevu_ = fPm->GetDoubleParameter(GetFullParmName("EnergyStepMeVu"), "Energy") / MeV;
    }

    if (min_energy_mevu_ <= 0.0 || max_energy_mevu_ < min_energy_mevu_ || energy_step_mevu_ <= 0.0) {
        G4Exception("AllIonElasticDump", "InvalidEnergyGridParameters", FatalException,
                    "Invalid energy grid parameters: Min <= 0 or Max < Min or Step <= 0!");
    }
}

AllIonElasticDump::~AllIonElasticDump() = default;

G4bool AllIonElasticDump::ProcessHits(G4Step*, G4TouchableHistory*) {
    if (!dumped_) {
        DumpCrossSections();
        dumped_ = true;
    }
    return true;
}

void AllIonElasticDump::Output() {
    if (!dumped_) {
        DumpCrossSections();
        dumped_ = true;
    }
}

#include "G4DynamicParticle.hh"
#include "G4CrossSectionDataStore.hh"
#include "G4EnergyRangeManager.hh"
#include "G4HadProjectile.hh"
#include "G4HadFinalState.hh"
#include "G4Nucleus.hh"
#include "G4NucleiProperties.hh"
#include "G4NistManager.hh"
#include "G4EmCalculator.hh"
#include "G4Isotope.hh"
#include <algorithm>
#include <set>
#include <cstdint>

void AllIonElasticDump::DumpCrossSections() {
    const int z=fPm->GetIntegerParameter(GetFullParmName("ProjectileZ"));
    const int a=fPm->GetIntegerParameter(GetFullParmName("ProjectileA"));
    const int samples=fPm->GetIntegerParameter(GetFullParmName("SamplesPerNode"));
    if(samples<32) throw std::runtime_error("Insufficient elastic samples");
    auto* table=G4ParticleTable::GetParticleTable();
    const char* light=z==1&&a==1?"proton":z==1&&a==2?"deuteron":z==1&&a==3?"triton":z==2&&a==3?"He3":z==2&&a==4?"alpha":nullptr;
    auto* projectile=light?table->FindParticle(light):table->GetIonTable()->GetIon(z,a,0.);
    if(!projectile||!projectile->GetProcessManager()) throw std::runtime_error("Missing projectile process manager");
    G4HadronicProcess* process=nullptr;int count=0;
    auto* pv=projectile->GetProcessManager()->GetProcessList();
    for(std::size_t i=0;i<pv->size();++i){auto* p=dynamic_cast<G4HadronicProcess*>((*pv)[i]);if(p&&p->GetProcessSubType()==fHadronElastic){process=p;++count;}}
    if(count!=1) throw std::runtime_error("Exactly one elastic process required");
    G4EnergyRangeManager manager;
    for(auto* model:process->GetHadronicInteractionList()) manager.RegisterMe(model);
    std::vector<double> energies;
    for(int i=0;i<=160;++i) energies.push_back(.001*std::pow(10.,i/20.));
    energies.erase(std::remove_if(energies.begin(),energies.end(),[](double e){return e>6000.;}),energies.end());
    energies.push_back(6000.11);
    std::vector<const G4Material*> mats;
    for(auto s:kSectionProbes){auto* m=G4Material::GetMaterial(MaterialNameFromHU(s.rep_hu),false);if(!m)throw std::runtime_error("Missing Schneider material");mats.push_back(m);}
    mats.push_back(G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER"));
    std::array<G4Material*,13> pure{};
    for(int t=0;t<13;++t){G4Element* el=nullptr;for(auto* m:mats)for(auto* e:*m->GetElementVector())if(int(e->GetZ())==kCanonicalZ[t])el=const_cast<G4Element*>(e);
      if(!el)throw std::runtime_error("Missing target element");
      pure[t]=new G4Material("ElasticProbe"+std::to_string(z)+"_"+std::to_string(t),1*g/cm3,1);pure[t]->AddElement(el,1.);
    }
    if(fPm->ParameterExists(GetFullParmName("RecoilStoppingOnly")) &&
       fPm->GetBooleanParameter(GetFullParmName("RecoilStoppingOnly"))) {
        std::set<std::pair<int,int>> isotopes;
        for(auto* m:pure) {
            auto* el=(*m->GetElementVector())[0];
            for(auto* isotope:*el->GetIsotopeVector())isotopes.insert({isotope->GetZ(),isotope->GetN()});
        }
        std::ofstream f(output_json_path_+".bin",std::ios::binary);
        auto write=[&](auto x){f.write(reinterpret_cast<const char*>(&x),sizeof(x));};
        f.write("ELRSP001",8);write(std::uint32_t(isotopes.size()));write(std::uint32_t(26));write(std::uint32_t(energies.size()));
        for(auto za:isotopes){write(std::int32_t(za.first));write(std::int32_t(za.second));}
        for(double e:energies)write(float(e));
        G4EmCalculator calculator;
        for(auto za:isotopes){
            auto* ion=za.first==1&&za.second==1?table->FindParticle("proton"):
                za.first==1&&za.second==2?table->FindParticle("deuteron"):table->GetIonTable()->GetIon(za.first,za.second,0.);
            for(auto* mat:mats)for(double e:energies){
                const double value=calculator.ComputeTotalDEDX(e*za.second*MeV,ion,mat)/(MeV/mm)/(mat->GetDensity()/(g/cm3));
                if(!std::isfinite(value)||value<=0)throw std::runtime_error("Invalid recoil stopping");
                write(float(value));
            }
        }
        if(!f)throw std::runtime_error("Recoil stopping write failed");
        G4cout<<"Recoil stopping complete isotopes="<<isotopes.size()<<G4endl;
        return;
    }
    auto* store=G4HadronicProcessStore::Instance();
    std::vector<double> rates;
    for(auto* m:mats)for(int tz:kCanonicalZ)for(double e:energies){double rate=0;
      for(std::size_t j=0;j<m->GetNumberOfElements();++j){auto* el=(*m->GetElementVector())[j];if(int(el->GetZ())==tz)
        rate+=m->GetVecNbOfAtomsPerVolume()[j]*store->GetElasticCrossSectionPerAtom(projectile,e*a*MeV,el,m)*mm/(m->GetDensity()/(g/cm3));}
      if(!std::isfinite(rate)||rate<0)throw std::runtime_error("Invalid elastic rate");
      rates.push_back(rate);
    }
    const std::string stem=output_json_path_;
    std::ofstream out(stem+".bin",std::ios::binary);if(!out)throw std::runtime_error("Cannot open output");
    auto write=[&](auto value){out.write(reinterpret_cast<const char*>(&value),sizeof(value));};
    out.write("ELRAW001",8);for(int v:{z,a,26,13,int(energies.size()),samples})write(std::uint32_t(v));
    write(double(projectile->GetPDGMass()/MeV));
    for(double e:energies)write(e);
    for(double r:rates)write(r);
    std::set<std::string> models;
    for(int t=0;t<13;++t)for(double e:energies){
      G4DynamicParticle dp(projectile,G4ThreeVector(0,0,1),e*a*MeV);G4HadProjectile hp(dp);
      auto* xs=process->GetCrossSectionDataStore();const double macro=xs->ComputeCrossSection(&dp,pure[t]);
      for(int n=0;n<samples;++n){
        G4Nucleus nucleus;
        if(!(macro>0)){write(float(0));write(float(1));write(std::uint32_t(0));continue;}
        auto* element=xs->SampleZandA(&dp,pure[t],nucleus);
        const int target_a=nucleus.GetA_asInt();const double m2=G4NucleiProperties::GetNuclearMass(target_a,kCanonicalZ[t])/MeV;
        auto* model=manager.GetHadronicInteraction(hp,nucleus,pure[t],element);
        if(!model)throw std::runtime_error("No applicable elastic model");
        models.insert(model->GetModelName());
        auto* fs=model->ApplyYourself(hp,nucleus);
        const double T=e*a,m1=projectile->GetPDGMass()/MeV;
        const double tmax=2*m2*T*(T+2*m1)/((m1+m2)*(m1+m2)+2*m2*T);
        const double recoil=T-fs->GetEnergyChange()/MeV;
        double fraction=recoil/tmax;
        if(!std::isfinite(fraction)||fraction < -1e-7||fraction>1+1e-5)throw std::runtime_error("Non two-body elastic final state");
        fraction=std::max(0.,std::min(1.,fraction));
        write(float(fraction));write(float(m2));write(std::uint32_t(target_a));fs->Clear();
      }
    }
    if(!out)throw std::runtime_error("Elastic binary write failed");
    out.close();
    std::ofstream meta(stem+".json");meta<<"{\"projectile_z\":"<<z<<",\"projectile_a\":"<<a<<",\"process\":\""<<process->GetProcessName()<<"\",\"models\":[";
    bool first=true;for(auto& name:models){if(!first)meta<<',';first=false;meta<<'"'<<name<<'"';}meta<<"],\"samples_per_node\":"<<samples<<",\"energy_nodes\":"<<energies.size()<<"}\n";
    G4cout<<"AllIonElasticDump complete Z="<<z<<" A="<<a<<G4endl;
}

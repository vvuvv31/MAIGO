#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4IonisParamMat.hh"
#include "G4SystemOfUnits.hh"
#include "G4Version.hh"
#include <fstream>
#include <iomanip>
#include <stdexcept>

// Exact NIST material from the same G4 installation as TOPAS. No physics list,
// irradiation, scorer, or patient fit is needed to extract these properties.
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    auto* material=G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");
    if(!material || material->GetNumberOfElements()!=2) return 3;
    // Caller creates a fresh output directory; never overwrite a frozen probe.
    if(std::ifstream(argv[1]).good()) return 4;
    std::ofstream out(argv[1]);
    out<<std::setprecision(17)<<"{\n  \"schema_version\": 1,\n  \"material\": \"G4_WATER\",\n"
       <<"  \"geant4_version_number\": "<<G4VERSION_NUMBER<<",\n"
       <<"  \"density_g_cm3\": "<<material->GetDensity()/(g/cm3)<<",\n"
       <<"  \"radiation_length_g_cm2\": "<<material->GetRadlen()*material->GetDensity()/(g/cm2)<<",\n"
       <<"  \"mean_excitation_energy_eV\": "<<material->GetIonisation()->GetMeanExcitationEnergy()/eV<<",\n"
       <<"  \"elements\": [\n";
    for(std::size_t i=0;i<2;++i) {
        const auto* element=(*material->GetElementVector())[i];
        out<<"    {\"z\": "<<element->GetZ()<<", \"atomic_mass_g_mol\": "<<element->GetA()/(g/mole)
           <<", \"mass_fraction\": "<<material->GetFractionVector()[i]<<"}"<<(i==0?",":"")<<"\n";
    }
    out<<"  ]\n}\n";
    return out.good()?0:5;
}

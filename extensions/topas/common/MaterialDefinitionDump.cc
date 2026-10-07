// Scorer for MaterialDefinitionDump
#include "MaterialDefinitionDump.hh"
#include "G4Element.hh"
#include "G4IonisParamMat.hh"
#include "G4Material.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4Version.hh"
#include <fstream>
#include <iomanip>
#include <stdexcept>

MaterialDefinitionDump::MaterialDefinitionDump(TsParameterManager* p, TsMaterialManager* m,
    TsGeometryManager* g, TsScoringManager* s, TsExtensionManager* x, G4String n,
    G4String q, G4String output, G4bool sub)
    : TsVScorer(p,m,g,s,x,n,q,output,sub), output_(output) {}

G4bool MaterialDefinitionDump::ProcessHits(G4Step* step, G4TouchableHistory*) {
    if(done_ || step->GetTrack()->GetParentID()!=0) return false;
    const auto* material=step->GetPreStepPoint()->GetMaterial();
    const double density=material->GetDensity()/(g/cm3);
    std::ofstream out(output_+".json");
    out<<std::setprecision(17)<<"{\n  \"schema_version\": 1,\n  \"material\": \""
       <<material->GetName()<<"\",\n  \"geant4_version_number\": "<<G4VERSION_NUMBER
       <<",\n  \"density_g_cm3\": "<<density
       <<",\n  \"radiation_length_g_cm2\": "<<material->GetRadlen()/cm*density
       <<",\n  \"mean_excitation_energy_eV\": "<<material->GetIonisation()->GetMeanExcitationEnergy()/eV
       <<",\n  \"elements\": [\n";
    for(std::size_t i=0;i<material->GetNumberOfElements();++i) {
        const auto* element=(*material->GetElementVector())[i];
        if(i) out<<",\n";
        out<<"    {\"z\": "<<int(element->GetZ())<<", \"atomic_mass_g_mol\": "
           <<element->GetA()/(g/mole)<<", \"mass_fraction\": "<<material->GetFractionVector()[i]<<'}';
    }
    out<<"\n  ]\n}\n";
    if(!out) throw std::runtime_error("MaterialDefinitionDump: output failed");
    done_=true;
    G4cout<<"MATERIAL_DEFINITION_DUMP=PASS "<<material->GetName()<<G4endl;
    return true;
}

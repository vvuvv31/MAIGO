// Offline formula inputs only: no transport, stopping tables, or target-node fits.
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4IonisParamMat.hh"
#include "G4RToEConvForElectron.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4Electron.hh"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if(argc!=3)throw std::runtime_error("usage: material_inputs.txt metadata.txt");
    std::ifstream in(argv[1]);std::ofstream out(argv[2]);
    if(!in || !out)throw std::runtime_error("Cannot open formula metadata files");
    auto* nist=G4NistManager::Instance();
    G4Electron::ElectronDefinition();
    G4RToEConvForElectron converter;converter.SetVerboseLevel(0);
    G4VRangeToEnergyConverter::SetEnergyRange(.00099*MeV,10000*MeV);
    out<<std::setprecision(17);
    int ordinal, section, count;double rho, excitation;
    while(in>>ordinal>>section>>rho>>excitation>>count) {
        auto* material=new G4Material("ANALYTIC_S"+std::to_string(section)+"D"+std::to_string(ordinal),
                                      rho*g/cm3,count);
        for(int i=0;i<count;++i){std::string element;double weight;in>>element>>weight;
            material->AddElement(nist->FindOrBuildElement(element),weight);}
        material->GetIonisation()->SetMeanExcitationEnergy(excitation*eV);
        auto* ion=material->GetIonisation();
        out<<ordinal<<' '<<section<<' '<<rho<<' '
           <<converter.Convert(.05*mm,material)/MeV<<' '
           <<CLHEP::twopi_mc2_rcl2*material->GetElectronDensity()/(MeV/mm)<<' '
           <<ion->GetCdensity()<<' '<<ion->GetMdensity()<<' '<<ion->GetAdensity()<<' '
           <<ion->GetX0density()<<' '<<ion->GetX1density()<<' '<<ion->GetD0density()<<' '
           <<ion->GetFermiEnergy()/MeV<<' '<<ion->GetZeffective()<<'\n';
    }
}

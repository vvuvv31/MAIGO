#include "SelectedElectronTransportNtuple.hh"
#include "TsParameterManager.hh"
SelectedElectronTransportNtuple::SelectedElectronTransportNtuple(
    TsParameterManager* pm,TsMaterialManager* mm,TsGeometryManager* gm,TsScoringManager* sm,
    TsExtensionManager* em,G4String name,G4String quantity,G4String output,G4bool sub)
    : PrimaryElectronTransportNtuple(pm,mm,gm,sm,em,name,quantity,output,sub) {
    selected_=pm->GetIntegerParameter(GetFullParmName("OnlyHistory"));
}
G4bool SelectedElectronTransportNtuple::ProcessHits(G4Step* step,G4TouchableHistory* history) {
    return GetEventID()==selected_ ? PrimaryElectronTransportNtuple::ProcessHits(step,history) : false;
}

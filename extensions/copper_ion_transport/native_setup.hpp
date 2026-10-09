#pragma once
#include "G4RunManager.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4IonINCLXXPhysics.hh"
#include "QGSP_BIC_HP.hh"
class Detector : public G4VUserDetectorConstruction {
 G4VPhysicalVolume* Construct() override {
  auto* solid=new G4Box("Cu",1*m,1*m,1*m);
  auto* logical=new G4LogicalVolume(solid,G4NistManager::Instance()->FindOrBuildMaterial("G4_Cu"),"Cu");
  return new G4PVPlacement(nullptr,{},logical,"Cu",nullptr,false,0);
 }
};

inline G4RunManager* make_copper_native_setup() {
 auto* rm=new G4RunManager;
 auto* physics=new QGSP_BIC_HP(0);
 physics->ReplacePhysics(new G4EmStandardPhysics_option4(0));
 physics->ReplacePhysics(new G4IonINCLXXPhysics(0));
 rm->SetUserInitialization(new Detector);rm->SetUserInitialization(physics);rm->Initialize();
 return rm;
}

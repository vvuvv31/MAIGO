#include "IonLETStepPhysics.hh"
#include "TsParameterManager.hh"
#include "G4EmParameters.hh"
#include "G4Threading.hh"
#include <cmath>
#include <stdexcept>

IonLETStepPhysics::IonLETStepPhysics(TsParameterManager* pm)
    : G4VPhysicsConstructor("IonLETStepPhysics"),
      fraction_(pm->GetUnitlessParameter("Ph/IonLETStepPhysics/RangeFraction")),
      finalRange_(pm->GetDoubleParameter("Ph/IonLETStepPhysics/FinalRange", "Length")),
      includeProtons_(pm->GetBooleanParameter("Ph/IonLETStepPhysics/IncludeProtons")) {
    if (!std::isfinite(fraction_) || fraction_ <= 0 || fraction_ > 1 ||
        !std::isfinite(finalRange_) || finalRange_ <= 0)
        throw std::invalid_argument("IonLETStepPhysics requires 0 < RangeFraction <= 1 and FinalRange > 0");
}

void IonLETStepPhysics::ConstructProcess() {
    if (!G4Threading::IsMasterThread()) return;
    auto* parameters = G4EmParameters::Instance();
    // Use native initialization controls, not live per-track process edits.
    // LightIons includes deuteron/triton as well as He3/alpha. GenericIon has
    // its own control. MuHad also includes other hadrons, so it is opt-in.
    parameters->SetStepFunctionLightIons(fraction_, finalRange_);
    parameters->SetStepFunctionIons(fraction_, finalRange_);
    if (includeProtons_) parameters->SetStepFunctionMuHad(fraction_, finalRange_);
}

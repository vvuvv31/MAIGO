#ifndef CarbonInelasticTargetIdentity_hh
#define CarbonInelasticTargetIdentity_hh

#include "G4HadronicProcess.hh"
#include "G4Isotope.hh"
#include "G4Nucleus.hh"

#include <sstream>

// Some Geant4 hadronic processes leave the isotope pointer unset while the
// selected G4Nucleus still contains authoritative integer Z/A. Keep those
// cases distinguishable: Z/A can be matched to an explicit material, while
// the missing isotope pointer remains an auditable extraction condition.
struct CarbonCinel02TargetIdentity {
    G4int z{0};
    G4int a{0};
    G4int isotope_id{-1};
    G4String name;
    G4bool isotope_pointer_present{false};
};

inline CarbonCinel02TargetIdentity ResolveCarbonCinel02Target(
    G4HadronicProcess* process) {
    CarbonCinel02TargetIdentity result;
    if (process == nullptr) {
        return result;
    }

    const auto* isotope = process->GetTargetIsotope();
    if (isotope != nullptr) {
        result.z = isotope->GetZ();
        result.a = isotope->GetN();
        result.isotope_id = static_cast<G4int>(isotope->GetIndex());
        result.name = isotope->GetName();
        result.isotope_pointer_present = true;
        return result;
    }

    const auto* nucleus = process->GetTargetNucleus();
    if (nucleus == nullptr) {
        return result;
    }
    result.z = nucleus->GetZ_asInt();
    result.a = nucleus->GetA_asInt();
    if (result.z > 0 && result.a >= result.z) {
        std::ostringstream name;
        name << "G4Nucleus(Z=" << result.z << ",A=" << result.a << ")";
        result.name = name.str();
    }
    return result;
}

#endif

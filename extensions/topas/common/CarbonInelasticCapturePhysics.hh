#ifndef CarbonInelasticCapturePhysics_hh
#define CarbonInelasticCapturePhysics_hh

#include "G4VPhysicsConstructor.hh"

class TsParameterManager;

class CarbonInelasticCapturePhysics final : public G4VPhysicsConstructor {
public:
    explicit CarbonInelasticCapturePhysics(G4int verbose = 0);
    explicit CarbonInelasticCapturePhysics(TsParameterManager*);
    ~CarbonInelasticCapturePhysics() override = default;

    void ConstructParticle() override;
    void ConstructProcess() override;

private:
    G4int verbose_{0};
};

#endif

#ifndef PrimaryCrossingCount_hh
#define PrimaryCrossingCount_hh

#include "TsVBinnedScorer.hh"

#include <set>

// Counts primary history crossings in component bins.  A history contributes
// at most one unit to each bin, including when it takes multiple steps or
// re-enters a bin after scattering.
class PrimaryCrossingCount : public TsVBinnedScorer {
public:
    PrimaryCrossingCount(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String,
        G4String, G4bool);

    ~PrimaryCrossingCount() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    void UserHookForEndOfEvent() override;

protected:
    void Clear() override;

private:
    G4int projectile_z_{0};
    G4int projectile_a_{0};
    G4int seen_run_id_{-1};
    G4int seen_event_id_{-1};
    std::set<G4int> seen_bin_indices_;
};

#endif

#ifndef ChargedOriginDoseToMedium_hh
#define ChargedOriginDoseToMedium_hh

#include "TsVBinnedScorer.hh"

#include <unordered_map>

class G4ParticleDefinition;
class G4Track;

class ChargedOriginDoseToMedium : public TsVBinnedScorer {
public:
    ChargedOriginDoseToMedium(
        TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String,
        G4String, G4bool);
    ~ChargedOriginDoseToMedium() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    void UserHookForEndOfEvent() override;

protected:
    void Clear() override;

private:
    enum class OriginCategory : G4int {
        PrimaryC = 0,
        SecondaryC = 1,
        Boron = 2,
        Beryllium = 3,
        Lithium = 4,
        Helium = 5,
        Z1 = 6,
        OtherCharged = 7,
        NeutralOrigin = 8,
        Unclassified = 9,
        Be6 = 10,
        Be7 = 11,
        Be9 = 12,
        Be10 = 13,
        BeOther = 14,
        He3 = 15,
        He4 = 16,
        HeOther = 17
    };

    OriginCategory ResolveOrigin(const G4Track*);
    static OriginCategory CategoryForChargedNucleus(
        const G4Track*, const G4ParticleDefinition*);
    static OriginCategory ParseRequestedCategory(const G4String&);

    OriginCategory requested_category_{OriginCategory::Unclassified};
    G4int seen_run_id_{-1};
    G4int seen_event_id_{-1};
    std::unordered_map<G4int, OriginCategory> track_origins_;
};

#endif

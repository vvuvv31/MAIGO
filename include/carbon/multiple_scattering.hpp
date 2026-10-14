#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace carbon {

constexpr double nucleon_rest_mass_MeV = 931.49410242;
constexpr double water_radiation_length_g_per_cm2 = 36.08;
constexpr double highland_energy_constant_MeV = 13.6;
// Geant4 material mass radiation lengths (g/cm2), measured with the
// CarbonMaterialPropertiesNtuple TOPAS extension.  Keep these in mass units:
// the local CT density below converts the transported step to areal density.
// 25 exact Schneider section mass radiation lengths (g/cm^2), derived from
// the TOPAS/Geant4 11.03.p02 material truth product:
// X0_mass = X0_length [cm] * density [g/cm^3].
inline constexpr double kSchneiderSectionRadiationLengthGPerCm2[25] = {
    36.608976, // sec 0  (HU -975, Air)
    36.527496, // sec 1  (HU -535, Lung)
    42.079618, // sec 2  (HU -102, Adipose tissue)
    40.909878, // sec 3  (HU -68,  Breast tissue)
    39.745431, // sec 4  (HU -38,  Water-like/Tissue)
    38.834958, // sec 5  (HU -8,   Muscle-like)
    38.152914, // sec 6  (HU 12,   Liver-like)
    36.809410, // sec 7  (HU 49,   Brain/Soft tissue)
    37.288668, // sec 8  (HU 100,  Soft tissue standard)
    36.812305, // sec 9  (HU 160,  Connective tissue)
    35.451101, // sec 10 (HU 250,  Cartilage)
    34.171385, // sec 11 (HU 350,  Trabecular bone 1)
    33.055455, // sec 12 (HU 450,  Trabecular bone 2)
    32.164282, // sec 13 (HU 550,  Trabecular bone 3)
    31.332632, // sec 14 (HU 650,  Spongiosa)
    30.603797, // sec 15 (HU 750,  Cortical bone transition)
    29.910616, // sec 16 (HU 850,  Cortical bone 1)
    29.374398, // sec 17 (HU 950,  Cortical bone 2)
    28.868185, // sec 18 (HU 1050, Cortical bone 3)
    28.377696, // sec 19 (HU 1150, Dense bone 1)
    27.983677, // sec 20 (HU 1250, Dense bone 2)
    27.604859, // sec 21 (HU 1350, Dense bone 3)
    27.240453, // sec 22 (HU 1450, Dense bone 4)
    26.994240, // sec 23 (HU 2247, High-density bone)
    16.163298  // sec 24 (HU 2995, Titanium implant)
};

// Shared X0 selector for primary AND secondary MCS in Schneider CT.
// Selection only: no scale, no scattering-formula change. Both transport
// paths must call this (never a private copy) so they cannot drift apart.
// section_id >= 25 falls back to water defensively; upstream launch-time
// validation rejects Schneider voxel material_id >= 25, so the fallback
// must never trigger in a valid Schneider run.
constexpr double select_transport_radiation_length_g_per_cm2(
    const bool ct_sections_are_schneider, const bool in_ct,
    const unsigned section_id, const bool enable_ct_material_mcs,
    const double fallback_water_x0) noexcept {
    if (enable_ct_material_mcs && in_ct && ct_sections_are_schneider) {
        if (section_id < 25U) {
            return kSchneiderSectionRadiationLengthGPerCm2[section_id];
        }
        return fallback_water_x0;
    }
    return fallback_water_x0;
}

}  // namespace carbon

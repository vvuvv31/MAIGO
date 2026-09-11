#pragma once

// Secondary-ion stopping power on the Schneider 25-section grid (scheme 2).
//
// Format SCHNIOSP v1 binary:
//   header: magic[8]="SCHNIOSP", version u32=1, num_sections u32=25,
//           num_species u32=18, num_energies u32=4302,
//           energy_min/energy_max/energy_step f64 = 0.01/430.11/0.1 (MeV/u)
//   body:   reference_densities[25] f64 (audit only, g/cm^3),
//           values[25][18][4302] f64, UNRESTRICTED electronic stopping
//           power as LINEAR stopping at unit density (MeV/mm at 1 g/cm^3),
//           same convention as the ion water CSV. Runtime scales by the
//           local voxel density: S(T,rho) = table * rho.
//
// Data must come from TOPAS/Geant4 G4EmCalculator extraction per Schneider
// material (see tools/compile_schneider_ion_stopping.py); never synthesize.
// Missing file = current water x material-factor path (fail-closed default).
// Electron contract: unrestricted totals only; the electron packet takes its
// share from deposited energy, no restricted/unrestricted double counting.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace carbon {

constexpr std::size_t kSchneiderIonSections = 25;
constexpr std::size_t kSchneiderIonSpecies = 18;
constexpr std::size_t kSchneiderIonEnergies = 4302;
constexpr double kSchneiderIonEnergyMin = 0.01;
constexpr double kSchneiderIonEnergyMax = 430.11;
constexpr double kSchneiderIonEnergyStep = 0.1;

#pragma pack(push, 1)
struct SchneiderIonStoppingHeader {
    char magic[8];             // "SCHNIOSP"
    uint32_t version;          // 1
    uint32_t num_sections;     // 25
    uint32_t num_species;      // 18
    uint32_t num_energies;     // 4302
    double energy_min_mevu;    // 0.01
    double energy_max_mevu;    // 400.01
    double energy_step_mevu;   // 0.1
};
#pragma pack(pop)

class SchneiderIonStoppingTable {
public:
    SchneiderIonStoppingTable() = default;

    static SchneiderIonStoppingTable from_binary(
        const std::filesystem::path& binary_path,
        const std::filesystem::path& metadata_path = {});

    [[nodiscard]] std::size_t num_sections() const noexcept { return kSchneiderIonSections; }
    [[nodiscard]] std::size_t num_species() const noexcept { return kSchneiderIonSpecies; }
    [[nodiscard]] std::size_t num_energies() const noexcept { return kSchneiderIonEnergies; }

    // Linear stopping at unit density (MeV/mm); -1 when out of domain.
    [[nodiscard]] double linear_stopping_at_unit_density(
        std::size_t species_idx, std::size_t section_id, double energy_mevu) const noexcept;

    [[nodiscard]] const std::vector<double>& densities() const noexcept { return densities_; }
    [[nodiscard]] const std::vector<double>& values() const noexcept { return values_; }

    [[nodiscard]] std::vector<float> to_flat_float() const;

private:
    std::vector<double> densities_;
    // Flattened [section][species][energy].
    std::vector<double> values_;
};

}  // namespace carbon

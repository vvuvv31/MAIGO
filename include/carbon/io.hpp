#pragma once

#include "carbon/transport.hpp"

#include <filesystem>

namespace carbon {

// MeV energy-deposition scorer (absolute MeV accumulated, written as MeV/primary).
// Existing combined CSV also includes a derived dose_Gy_per_primary column for
// backward compatibility; prefer write_depth_dose_Gy_csv for a pure dose scorer.
void write_depth_dose_csv(const std::filesystem::path& path,
                          const TransportConfig& config,
                          const TransportResult& result);

// Dose scorer: Dose-to-water (Gy/primary) using scorer_area_mm2 × depth_bin × density.
// Independent of the MeV scorer; same energy tally, different quantity / units.
void write_depth_dose_Gy_csv(const std::filesystem::path& path,
                             const TransportConfig& config,
                             const TransportResult& result);

void write_fragment_species_csv(const std::filesystem::path& path,
                                const TransportConfig& config,
                                const TransportResult& result);

// Fragment-species dose scorer (Gy/primary), same bin mass as the depth dose scorer.
void write_fragment_species_dose_Gy_csv(const std::filesystem::path& path,
                                        const TransportConfig& config,
                                        const TransportResult& result);

// MeV voxel energy scorer (also writes derived Gy column for compatibility).
void write_sparse_voxel_dose_csv(const std::filesystem::path& path,
                                 const TransportConfig& config,
                                 const TransportResult& result);

// Pure voxel dose scorer (Gy/primary); mass = voxel_dx×dy×dz × water density.
void write_sparse_voxel_dose_Gy_csv(const std::filesystem::path& path,
                                    const TransportConfig& config,
                                    const TransportResult& result);

void write_sparse_charged_origin_voxel_dose_csv(
    const std::filesystem::path& path,
    const TransportConfig& config,
    const TransportResult& result);

// Charged-origin categories as Gy/primary (same mass as voxel dose scorer).
void write_sparse_charged_origin_voxel_dose_Gy_csv(
    const std::filesystem::path& path,
    const TransportConfig& config,
    const TransportResult& result);

}  // namespace carbon

#pragma once

#include "carbon/transport.hpp"

#include <filesystem>

namespace carbon {

// MeV energy-deposition scorer: absolute MeV summed over all histories.
// Also writes a derived total dose_Gy column for convenience.
void write_depth_dose_csv(const std::filesystem::path& path,
                          const TransportConfig& config,
                          const TransportResult& result);

// Dose scorer: total dose-to-water (Gy) using scorer_area_mm2 × depth_bin × density.
// Values are NOT divided by history count.
void write_depth_dose_Gy_csv(const std::filesystem::path& path,
                             const TransportConfig& config,
                             const TransportResult& result);

void write_fragment_species_csv(const std::filesystem::path& path,
                                const TransportConfig& config,
                                const TransportResult& result);

// Fragment-species total dose (Gy), same bin mass as the depth dose scorer.
void write_fragment_species_dose_Gy_csv(const std::filesystem::path& path,
                                        const TransportConfig& config,
                                        const TransportResult& result);

// Sparse voxel MeV energy scorer (total MeV) with derived total dose_Gy column.
void write_sparse_voxel_dose_csv(const std::filesystem::path& path,
                                 const TransportConfig& config,
                                 const TransportResult& result);

// Pure voxel total dose scorer (Gy); mass = voxel_dx×dy×dz × density.
void write_sparse_voxel_dose_Gy_csv(const std::filesystem::path& path,
                                    const TransportConfig& config,
                                    const TransportResult& result);

void write_sparse_charged_origin_voxel_dose_csv(
    const std::filesystem::path& path,
    const TransportConfig& config,
    const TransportResult& result);

// Charged-origin categories as total Gy (same mass as voxel dose scorer).
void write_sparse_charged_origin_voxel_dose_Gy_csv(
    const std::filesystem::path& path,
    const TransportConfig& config,
    const TransportResult& result);

// Dense MetaImage MHD/RAW (total Gy) written directly from the in-memory dose
// grid. Prefer this over sparse CSV + postprocess for production CT outputs.
// Path may end in .mhd or any name; the .raw sibling is always derived.
void write_dense_voxel_dose_mhd(const std::filesystem::path& mhd_path,
                                const TransportConfig& config,
                                const TransportResult& result);

}  // namespace carbon

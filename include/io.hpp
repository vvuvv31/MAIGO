#pragma once
#include "transport.hpp"
#include <filesystem>
namespace carbon {
void write_dense_voxel_let_mhd(const std::filesystem::path& prefix,
                              const TransportConfig& config,
                              const TransportResult& result);
void write_dense_voxel_dose_mhd(const std::filesystem::path& mhd_path,
                                const TransportConfig& config,
                                const TransportResult& result);
void write_energy_ledger_json(const std::filesystem::path& path,
                              const TransportConfig& config,
                              const TransportResult& result);
void write_schneider_miss_log_json(const std::filesystem::path& path,
                                   const TransportResult& result);
void write_schneider_unsupported_tracks_json(const std::filesystem::path& path,
                                             const TransportResult& result);
}

#pragma once

#include "carbon/transport.hpp"

#include <filesystem>

namespace carbon {

void write_depth_dose_csv(const std::filesystem::path& path,
                          const TransportConfig& config,
                          const TransportResult& result);

void write_fragment_species_csv(const std::filesystem::path& path,
                                const TransportConfig& config,
                                const TransportResult& result);

}  // namespace carbon

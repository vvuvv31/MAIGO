#pragma once

#include "carbon/transport.hpp"
#include "carbon/transport_config.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace carbon {

struct QualityIssue {
    std::string code;
    std::string message;
    double value{};
    double limit{};
};

struct RunQualityReport {
    RunMode mode{RunMode::research};
    bool accepted{false};
    double absolute_energy_residual_MeV{};
    double relative_energy_residual{};
    std::uint64_t queue_overflow_count{};
    double queue_overflow_energy_MeV{};
    std::uint64_t cascade_selection_exact{};
    std::uint64_t cascade_selection_expanded{};
    std::uint64_t cascade_selection_nearest{};
    std::uint64_t cascade_selection_no_coverage{};
    double cascade_selection_energy_distance_sum_MeVu{};
    double cascade_selection_energy_distance_max_MeVu{};
    std::vector<QualityIssue> failures;
    std::vector<QualityIssue> approximations;

    [[nodiscard]] std::string status() const;
    [[nodiscard]] std::string summary() const;
};

[[nodiscard]] RunQualityReport evaluate_run_quality(
    const TransportConfig& config, const TransportResult& result);

void write_run_quality_report_json(const std::filesystem::path& path,
                                   const RunQualityReport& report);

}  // namespace carbon

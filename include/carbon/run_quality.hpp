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
    // Physical closure excludes explicit TOPAS compatibility sinks.  It is
    // intentionally reported separately so a reference-model defect cannot
    // be mistaken for deposited or escaped energy.
    double absolute_physical_energy_residual_MeV{};
    double physical_relative_energy_residual{};
    // Accounting closure includes every explicitly classified sink, including
    // topas_compat_discarded_kinetic_MeV.
    double absolute_accounting_energy_residual_MeV{};
    double accounting_relative_energy_residual{};
    // Legacy aliases retained for consumers of schema v1 reports.  They are
    // exactly the accounting values above and are emitted alongside the new
    // names for one compatibility cycle.
    double absolute_energy_residual_MeV{};
    double relative_energy_residual{};
    bool topas_reference_energy_sink_active{false};
    std::uint64_t queue_overflow_count{};
    double queue_overflow_energy_MeV{};
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

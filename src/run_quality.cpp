#include "carbon/run_quality.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace carbon {
namespace {

void append_json_string(std::ostream& output, const std::string& value) {
    output << '"';
    for (const auto character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default: output << character; break;
        }
    }
    output << '"';
}

void write_json_number(std::ostream& output, const double value) {
    if (std::isfinite(value)) {
        output << value;
    } else {
        output << "null";
    }
}

void write_issues(std::ostream& output,
                  const std::vector<QualityIssue>& issues) {
    output << "[";
    for (std::size_t index = 0; index < issues.size(); ++index) {
        const auto& issue = issues[index];
        output << (index == 0 ? "\n" : ",\n") << "    {\"code\": ";
        append_json_string(output, issue.code);
        output << ", \"message\": ";
        append_json_string(output, issue.message);
        output << ", \"value\": ";
        write_json_number(output, issue.value);
        output << ", \"limit\": ";
        write_json_number(output, issue.limit);
        output << "}";
    }
    if (!issues.empty()) {
        output << '\n';
    }
    output << "  ]";
}

}  // namespace

std::string RunQualityReport::status() const {
    if (!accepted) {
        return "fail";
    }
    return mode == RunMode::production ? "pass" : "non_production";
}

std::string RunQualityReport::summary() const {
    std::ostringstream output;
    output << "run quality " << status() << " (mode=" << run_mode_name(mode)
           << ", failures=" << failures.size()
           << ", approximations=" << approximations.size() << ')';
    if (!failures.empty()) {
        output << ": " << failures.front().message;
    }
    return output.str();
}

RunQualityReport evaluate_run_quality(const TransportConfig& config,
                                      const TransportResult& result) {
    RunQualityReport report;
    report.mode = config.run_mode;

    const auto raw_residual =
        result.initial_energy_MeV - result.total_deposited_energy_MeV -
        result.escaped_energy_MeV - result.beamline_removed_energy_MeV -
        result.untracked_nuclear_energy_MeV;
    report.absolute_energy_residual_MeV = std::abs(raw_residual);
    report.relative_energy_residual = result.relative_energy_balance_error();

    const auto add_issue = [&](const QualityIssue& issue,
                               const bool production_failure) {
        if (production_failure && config.run_mode == RunMode::production) {
            report.failures.push_back(issue);
        } else {
            report.approximations.push_back(issue);
        }
    };

    const auto add_overflow = [&](const char* code, const char* label,
                                  const std::uint64_t count,
                                  const double energy_MeV) {
        report.queue_overflow_count += count;
        report.queue_overflow_energy_MeV += energy_MeV;
        if (count == 0 && energy_MeV == 0.0) {
            return;
        }
        const QualityIssue issue{
            code, std::string(label) + " queue overflow",
            static_cast<double>(count), 0.0};
        if (config.quality_reject_any_queue_overflow &&
            (config.run_mode == RunMode::production ||
             config.validation_scorers())) {
            report.failures.push_back(issue);
        } else {
            report.approximations.push_back(issue);
        }
    };
    add_overflow("elastic_queue_overflow", "elastic",
                 result.elastic_queue_overflow,
                 result.elastic_queue_overflow_energy_MeV);
    add_overflow("secondary_queue_overflow", "secondary",
                 result.secondary_queue_overflow,
                 result.secondary_queue_overflow_energy_MeV);
    add_overflow("cascade_queue_overflow", "cascade",
                 result.cascade_queue_overflow, 0.0);
    add_overflow("neutral_queue_overflow", "neutral",
                 result.neutral_queue_overflow,
                 result.neutral_queue_overflow_energy_MeV);
    add_overflow("electron_queue_overflow", "electron",
                 result.electron_queue_overflow,
                 result.electron_queue_overflow_energy_MeV);
    add_overflow("electron_gamma_queue_overflow", "electron gamma",
                 result.electron_gamma_queue_overflow,
                 result.electron_gamma_queue_overflow_energy_MeV);

    const auto finite_issue = [&](const std::string& label) {
        report.failures.push_back(
            {"non_finite_result", label + " contains NaN or infinity", 1.0, 0.0});
    };
    if (config.quality_reject_nan_or_inf) {
        const double scalars[]{
            result.initial_energy_MeV,
            result.total_deposited_energy_MeV,
            result.escaped_energy_MeV,
            result.beamline_removed_energy_MeV,
            result.untracked_nuclear_energy_MeV,
            result.generated_direct_secondary_energy_MeV,
            result.queued_secondary_energy_MeV,
            result.secondary_queue_overflow_energy_MeV,
            result.untransported_neutral_energy_MeV,
            result.untransported_unsupported_charged_energy_MeV,
            result.nuclear_energy_not_in_direct_secondaries_MeV,
            result.elastic_local_deposited_energy_MeV,
            result.elastic_queued_charged_energy_MeV,
            result.elastic_queued_neutral_energy_MeV,
            result.elastic_queue_overflow_energy_MeV,
            result.secondary_deposited_energy_MeV,
            result.secondary_escaped_energy_MeV,
            result.queued_cascade_energy_MeV,
            result.cascade_nuclear_energy_MeV,
            result.queued_neutral_energy_MeV,
            result.neutral_queue_overflow_energy_MeV,
            result.neutral_deposited_energy_MeV,
            result.neutral_escaped_energy_MeV,
            result.residual_neutral_energy_MeV,
            result.charged_from_neutral_energy_MeV,
            result.neutral_unsupported_product_energy_MeV,
            result.neutral_package_closure_residual_MeV,
            result.queued_electron_energy_MeV,
            result.electron_queue_overflow_energy_MeV,
            result.electron_deposited_energy_MeV,
            result.electron_escaped_energy_MeV,
            result.electron_radiative_energy_MeV,
            result.positron_annihilation_reserve_MeV,
            result.electron_brems_gamma_energy_MeV,
            result.positron_annihilation_gamma_energy_MeV,
            result.electron_gamma_queue_overflow_energy_MeV,
            result.electromagnetic_generation_residual_MeV,
            result.cascade_selection_energy_distance_sum_MeVu,
            result.cascade_selection_energy_distance_max_MeVu,
            result.elapsed_seconds,
            result.primary_kernel_seconds,
            result.secondary_kernel_seconds,
            result.neutral_kernel_seconds,
            result.electron_kernel_seconds,
            result.charged_after_neutral_kernel_seconds,
            report.absolute_energy_residual_MeV,
            report.relative_energy_residual,
        };
        if (!std::all_of(std::begin(scalars), std::end(scalars),
                         [](const double value) { return std::isfinite(value); })) {
            finite_issue("energy ledger");
        }
        const auto finite_vector = [](const std::vector<double>& values) {
            return std::all_of(values.begin(), values.end(), [](const double value) {
                return std::isfinite(value);
            });
        };
        for (const auto& [label, values] :
             std::initializer_list<std::pair<const char*, const std::vector<double>*>>{
                 {"depth dose", &result.deposited_energy_MeV},
                 {"voxel dose", &result.voxel_deposited_energy_MeV},
                 {"charged-origin dose", &result.charged_origin_voxel_deposited_energy_MeV},
                 {"neutral-origin dose", &result.neutral_origin_voxel_deposited_energy_MeV},
                 {"primary dose", &result.primary_deposited_energy_MeV},
                 {"secondary carbon dose", &result.secondary_carbon_deposited_energy_MeV},
                 {"secondary boron dose", &result.secondary_boron_deposited_energy_MeV},
                 {"secondary beryllium dose", &result.secondary_beryllium_deposited_energy_MeV},
                 {"secondary lithium dose", &result.secondary_lithium_deposited_energy_MeV},
                 {"secondary helium dose", &result.secondary_helium_deposited_energy_MeV},
                 {"secondary proton dose", &result.secondary_proton_deposited_energy_MeV},
                 {"secondary other dose", &result.secondary_other_charged_deposited_energy_MeV},
                 {"primary fluence", &result.primary_fluence_mm},
                 {"secondary carbon fluence", &result.secondary_carbon_fluence_mm},
                 {"secondary boron fluence", &result.secondary_boron_fluence_mm},
                 {"secondary beryllium fluence", &result.secondary_beryllium_fluence_mm},
                 {"secondary lithium fluence", &result.secondary_lithium_fluence_mm},
                 {"secondary helium fluence", &result.secondary_helium_fluence_mm},
                 {"secondary proton fluence", &result.secondary_proton_fluence_mm},
                 {"secondary other fluence", &result.secondary_other_charged_fluence_mm},
                 {"primary LET numerator", &result.primary_letd_numerator},
                 {"primary LET denominator", &result.primary_letd_denominator},
                 {"hadron LET numerator", &result.all_hadron_letd_numerator},
                 {"hadron LET denominator", &result.all_hadron_letd_denominator},
                 {"charged-origin LET numerator", &result.charged_origin_letd_numerator},
                 {"charged-origin LET denominator", &result.charged_origin_letd_denominator},
                 {"light-isotope LET numerator", &result.light_isotope_letd_numerator},
                 {"light-isotope LET denominator", &result.light_isotope_letd_denominator},
                 {"birth kinetic-energy sum", &result.birth_ke_sum_MeV_by_generation},
                 {"primary voxel LET numerator", &result.primary_voxel_letd_numerator},
                 {"primary voxel LET denominator", &result.primary_voxel_letd_denominator},
                 {"hadron voxel LET numerator", &result.all_hadron_voxel_letd_numerator},
                 {"hadron voxel LET denominator", &result.all_hadron_voxel_letd_denominator},
                 {"neutron-origin dose", &result.neutron_origin_deposited_energy_MeV},
                 {"gamma-origin dose", &result.gamma_origin_deposited_energy_MeV},
             }) {
            if (!finite_vector(*values)) {
                finite_issue(label);
            }
        }
    }

    if (result.fred_inelastic_events > 0) {
        const auto scaled_frac =
            static_cast<double>(result.fred_energy_scaled_events) /
            static_cast<double>(result.fred_inelastic_events);
        if (scaled_frac > 0.05) {
            add_issue({"fred_energy_scale_fraction",
                       "inelastic energy-rescale fraction exceeds 5%",
                       scaled_frac, 0.05},
                      false);
        }
        const auto residual_frac =
            (result.initial_energy_MeV > 0.0)
                ? result.fred_model_residual_MeV / result.initial_energy_MeV
                : 0.0;
        if (residual_frac > 1.0e-3) {
            add_issue({"fred_model_residual",
                       "labeled inelastic model residual exceeds 0.1% of incident energy",
                       residual_frac, 1.0e-3},
                      false);
        }
        if (result.fred_resample_failed_events > 0) {
            const auto fail_frac =
                static_cast<double>(result.fred_resample_failed_events) /
                static_cast<double>(result.fred_inelastic_events);
            add_issue({"fred_resample_failed",
                       "inelastic resampling exhausted without an accepted set",
                       fail_frac, 0.01},
                      fail_frac > 0.01);
        }
        if (result.fred_projectile_az_open_events > 0) {
            add_issue({"fred_projectile_az_open",
                       "projectile A/Z leftover after accepted inelastic event",
                       static_cast<double>(result.fred_projectile_az_open_events), 0.0},
                      false);
        }
    }

    if (std::isfinite(report.absolute_energy_residual_MeV) &&
        std::isfinite(report.relative_energy_residual)) {
        const auto allowed_residual = std::max(
            config.quality_maximum_absolute_energy_residual_MeV,
            config.quality_maximum_relative_energy_residual *
                std::abs(result.initial_energy_MeV));
        if (report.absolute_energy_residual_MeV > allowed_residual) {
            add_issue({"energy_residual_exceeded",
                       "energy ledger residual exceeds the configured tolerance",
                       report.absolute_energy_residual_MeV, allowed_residual}, true);
        }
    }

    report.accepted = report.failures.empty();
    return report;
}

void write_run_quality_report_json(const std::filesystem::path& path,
                                   const RunQualityReport& report) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot create run quality report: " + path.string());
    }
    output << std::setprecision(12)
           << "{\n  \"schema_version\": 1,\n  \"status\": ";
    append_json_string(output, report.status());
    output << ",\n  \"run_mode\": ";
    append_json_string(output, run_mode_name(report.mode));
    output << ",\n  \"accepted\": " << (report.accepted ? "true" : "false")
           << ",\n  \"absolute_energy_residual_MeV\": ";
    write_json_number(output, report.absolute_energy_residual_MeV);
    output << ",\n  \"relative_energy_residual\": ";
    write_json_number(output, report.relative_energy_residual);
    output
           << ",\n  \"queue_overflow_count\": " << report.queue_overflow_count
           << ",\n  \"queue_overflow_energy_MeV\": ";
    write_json_number(output, report.queue_overflow_energy_MeV);
    output << ",\n  \"failures\": ";
    write_issues(output, report.failures);
    output << ",\n  \"approximations\": ";
    write_issues(output, report.approximations);
    output << "\n}\n";
    if (!output) {
        throw std::runtime_error("Failed while writing run quality report: " +
                                 path.string());
    }
}

}  // namespace carbon

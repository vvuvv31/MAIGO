#include "carbon/run_quality.hpp"
#include "carbon/min_json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
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

    // Keep the two closure equations explicit.  The physical residual omits
    // the intentional TOPAS compatibility sink; the accounting residual
    // includes it.  Both use the same fred_model_unassigned term as the
    // corresponding TransportResult helper.
    const auto physical_raw_residual =
        result.initial_energy_MeV - result.total_deposited_energy_MeV -
        result.escaped_energy_MeV - result.beamline_removed_energy_MeV -
        result.untracked_nuclear_energy_MeV - result.fred_model_unassigned_MeV;
    const auto accounting_raw_residual =
        physical_raw_residual - result.topas_compat_discarded_kinetic_total_MeV();
    report.absolute_physical_energy_residual_MeV = std::abs(physical_raw_residual);
    report.physical_relative_energy_residual =
        result.physical_relative_energy_balance_error();
    report.absolute_accounting_energy_residual_MeV = std::abs(accounting_raw_residual);
    report.accounting_relative_energy_residual =
        result.relative_energy_balance_error();
    // Preserve the v1 field meanings for existing scripts: they represented
    // accounting closure.
    report.absolute_energy_residual_MeV = report.absolute_accounting_energy_residual_MeV;
    report.relative_energy_residual = report.accounting_relative_energy_residual;
    report.topas_reference_energy_sink_active =
        config.cinel02_topas_compatibility_mode &&
        result.topas_compat_discarded_kinetic_total_MeV() > 0.0;

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

    {
        double voxel_sum_MeV = 0.0;
        for (const double value : result.voxel_deposited_energy_MeV) {
            voxel_sum_MeV += value;
        }
        report.voxel_scored_energy_MeV = voxel_sum_MeV;
        report.in_grid_deposited_energy_MeV =
            result.in_grid_deposited_energy_MeV;
        report.outside_grid_deposited_energy_MeV =
            result.outside_grid_deposited_energy_MeV;
        const double total = result.total_deposited_energy_MeV;
        report.voxel_to_total_deposited_ratio =
            total > 0.0 ? voxel_sum_MeV / total
                        : (voxel_sum_MeV > 0.0
                               ? std::numeric_limits<double>::infinity()
                               : 0.0);
        constexpr double kVoxelOverTotalTolerance = 0.02;
        if (config.quality_reject_voxel_over_total &&
            report.voxel_to_total_deposited_ratio > 1.0 + kVoxelOverTotalTolerance) {
            report.failures.push_back(
                {"voxel_exceeds_deposited_total",
                 "scored 3D voxel energy exceeds the global deposited total",
                 report.voxel_to_total_deposited_ratio,
                 1.0 + kVoxelOverTotalTolerance});
        }
        // Explicit grid closure from device-side sinks (no subtraction
        // inference): total ≈ in + outside, voxel_sum ≈ in_grid. Tolerance
        // 1e-3 covers float32/float64 atomic accumulation order only;
        // measured device agreement is ~1e-6 on slabs and shards. The gates
        // stay silent when the split sinks are all zero (legacy/CPU results
        // predate them): any current-binary run with deposits records a
        // nonzero split, since every deposited-ledger credit splits.
        constexpr double kGridClosureTolerance = 1.0e-3;
        const double split_sum =
            report.in_grid_deposited_energy_MeV +
            report.outside_grid_deposited_energy_MeV;
        report.grid_split_closure_ratio =
            total > 0.0 ? split_sum / total
                        : (split_sum == 0.0
                               ? 1.0
                               : std::numeric_limits<double>::infinity());
        report.voxel_to_ingrid_ratio =
            report.in_grid_deposited_energy_MeV > 0.0
                ? voxel_sum_MeV / report.in_grid_deposited_energy_MeV
                : (voxel_sum_MeV > 0.0
                       ? std::numeric_limits<double>::infinity()
                       : 0.0);
        if (config.quality_reject_grid_closure && split_sum > 0.0) {
            if (std::abs(report.grid_split_closure_ratio - 1.0) >
                kGridClosureTolerance) {
                report.failures.push_back(
                    {"grid_split_closure",
                     "in-grid + outside-grid sinks disagree with the deposited total",
                     report.grid_split_closure_ratio, 1.0});
            }
            if (report.in_grid_deposited_energy_MeV > 0.0 &&
                std::abs(report.voxel_to_ingrid_ratio - 1.0) >
                    kGridClosureTolerance) {
                report.failures.push_back(
                    {"voxel_ingrid_closure",
                     "scored 3D voxel energy disagrees with the in-grid sink",
                     report.voxel_to_ingrid_ratio, 1.0});
            }
        }
    }

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
            result.topas_compat_discarded_kinetic_total_MeV(),
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
            report.absolute_physical_energy_residual_MeV,
            report.physical_relative_energy_residual,
            report.absolute_accounting_energy_residual_MeV,
            report.accounting_relative_energy_residual,
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

    if (report.topas_reference_energy_sink_active) {
        add_issue({"topas_reference_energy_sink_active",
                   "TOPAS compatibility mode discarded unsupported prompt-ion kinetic energy",
                   result.topas_compat_discarded_kinetic_total_MeV(), 0.0},
                  false);
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

    if (std::isfinite(report.absolute_accounting_energy_residual_MeV) &&
        std::isfinite(report.accounting_relative_energy_residual)) {
        const auto allowed_residual = std::max(
            config.quality_maximum_absolute_energy_residual_MeV,
            config.quality_maximum_relative_energy_residual *
                std::abs(result.initial_energy_MeV));
        if (report.absolute_accounting_energy_residual_MeV > allowed_residual) {
            add_issue({"energy_residual_exceeded",
                       "accounting energy ledger residual exceeds the configured tolerance",
                       report.absolute_accounting_energy_residual_MeV, allowed_residual}, true);
        }
    }

    if (std::isfinite(report.physical_relative_energy_residual) &&
        !report.topas_reference_energy_sink_active) {
        if (report.physical_relative_energy_residual > config.quality_maximum_relative_energy_residual) {
            report.failures.push_back(
                {"physical_energy_residual_exceeded",
                 "physical relative energy residual exceeds the configured tolerance",
                 report.physical_relative_energy_residual,
                 config.quality_maximum_relative_energy_residual});
        }
    }

    // Schneider-CT production gates: any non-zero physical failure in the
    // named diagnostics fails validation. Research mode still completes and
    // writes diagnostics, but accepted is false. Production likewise reports
    // accepted=false so the dose is refused as a formal result.
    if (config.is_schneider_ct_mode() && !config.is_primary_attenuation_only_mode()) {
        const auto& sch = result.schneider_diagnostics;
        const auto schneider_fail = [&](const char* code, const std::string& message,
                                        const double value) {
            report.failures.push_back({code, message, value, 0.0});
        };
        if (sch.primary_missing_projectile != 0) {
            schneider_fail("schneider_primary_missing_projectile",
                           "Schneider primary lookup hit an unsupported projectile",
                           static_cast<double>(sch.primary_missing_projectile));
        }
        if (sch.primary_missing_target != 0) {
            schneider_fail("schneider_primary_missing_target",
                           "Schneider primary lookup missed an exact target channel (no alias allowed)",
                           static_cast<double>(sch.primary_missing_target));
        }
        if (sch.primary_below_domain != 0 || sch.primary_above_domain != 0) {
            schneider_fail("schneider_primary_out_of_domain",
                           "Schneider primary query outside the exact channel energy domain",
                           static_cast<double>(sch.primary_below_domain + sch.primary_above_domain));
        }
        if (sch.primary_energy_gap_misses != 0) {
            schneider_fail("schneider_primary_energy_gap",
                           "Schneider primary bracketing gap exceeds the fixed threshold",
                           static_cast<double>(sch.primary_energy_gap_misses));
        }
        if (sch.primary_empty_nodes != 0) {
            schneider_fail("schneider_primary_empty_node",
                           "Schneider primary lookup selected an empty event node",
                           static_cast<double>(sch.primary_empty_nodes));
        }
        // Coverage gate uses the per-TRACK count (counted once per track at
        // track start), never the inflated per-step evaluation count. Be6
        // tracks are exempt: the frozen TopasCompatKill policy covers them
        // (continuous slowing only, never a nuclear hazard), so they need
        // no rate/CINEL data.
        //
        // v3 (bundle) amendments — v1 gates below are byte-identical:
        // (a) missing_projectile fails only on bundle-SCOPE misses: actual
        //     lookup misses, or logged unsupported tracks whose (Z,A) IS in
        //     the bundle registry. Out-of-scope isotopes (no event data on
        //     disk by design) deposit locally and are reported, not failed.
        //     Fail-closed: log truncation (dropped>0) or missing bundle
        //     refuses acceptance since scope cannot be proven.
        // (b) missing_target fails only on actual package-query misses.
        //     Sampler-empty draws (slowing left every channel domain between
        //     hazard and collision; no query issued) resolve via
        //     stopped-before-replay and never fail.
        bool is_v3_bundle = false;
        std::set<std::pair<int, int>> bundle_registry;
        {
            std::filesystem::path sec_rate = config.ct_schneider_secondary_rate_file;
            if (sec_rate.empty()) {
                sec_rate = "data/schneider/secondary_inelastic_rates_v1.bin";
            }
            if (std::filesystem::exists(sec_rate)) {
                std::ifstream rate_in(sec_rate, std::ios::binary);
                char magic[8]{};
                std::uint32_t version{0};
                rate_in.read(magic, 8);
                rate_in.read(reinterpret_cast<char*>(&version), sizeof(version));
                if (rate_in && std::memcmp(magic, "SCHN2RAT", 8) == 0 && version == 3) {
                    is_v3_bundle = true;
                    if (config.ct_schneider_physics_bundle_file.empty()) {
                        schneider_fail("schneider_bundle_missing",
                                       "v3 rate file requires ct_schneider_physics_bundle_file",
                                       1.0);
                    } else {
                        std::ifstream bundle_in(config.ct_schneider_physics_bundle_file);
                        std::string bundle_content((std::istreambuf_iterator<char>(bundle_in)),
                                                   std::istreambuf_iterator<char>());
                        minjson::Parser bundle_parser(bundle_content);
                        const minjson::Value bundle_doc = bundle_parser.parse();
                        for (const auto& entry :
                             bundle_doc.at("projectile_registry").arr) {
                            const int z = static_cast<int>(
                                minjson::require_uint(entry.at("z"), "registry.z"));
                            const int a = static_cast<int>(
                                minjson::require_uint(entry.at("a"), "registry.a"));
                            bundle_registry.emplace(z, a);
                        }
                    }
                }
            }
        }
        const auto unsupported_nonbe6_tracks =
            sch.unsupported_projectile_tracks - std::min(sch.unsupported_projectile_tracks,
                                                         sch.unsupported_be6_tracks);
        if (!is_v3_bundle) {
            if (sch.secondary_missing_projectile != 0 || unsupported_nonbe6_tracks != 0) {
                schneider_fail("schneider_secondary_missing_projectile",
                               "Schneider secondary lookup hit an unsupported non-Be6 projectile",
                               static_cast<double>(sch.secondary_missing_projectile +
                                                   unsupported_nonbe6_tracks));
            }
            if (sch.secondary_missing_target != 0 || sch.unsupported_targets != 0) {
                schneider_fail("schneider_secondary_missing_target",
                               "Schneider secondary lookup missed an exact target channel (no alias allowed)",
                               static_cast<double>(sch.secondary_missing_target +
                                                   sch.unsupported_targets));
            }
        } else {
            if (sch.secondary_missing_projectile != 0) {
                schneider_fail("schneider_secondary_missing_projectile",
                               "Schneider secondary lookup hit an unsupported non-Be6 projectile",
                               static_cast<double>(sch.secondary_missing_projectile));
            }
            if (sch.unsupported_log_dropped != 0) {
                schneider_fail("schneider_unsupported_log_truncated",
                               "unsupported-track log truncated; bundle scope unprovable",
                               static_cast<double>(sch.unsupported_log_dropped));
            } else {
                std::uint64_t bundle_scope_unsupported = 0;
                for (const auto& track : result.schneider_unsupported_tracks) {
                    if (track.projectile_z == 4 && track.projectile_a == 6) {
                        continue;  // Be6 TopasCompatKill exempt
                    }
                    if (bundle_registry.count({track.projectile_z, track.projectile_a}) != 0) {
                        ++bundle_scope_unsupported;
                    }
                }
                if (bundle_scope_unsupported != 0) {
                    schneider_fail("schneider_secondary_missing_projectile",
                                   "Schneider secondary bundle-scope projectile unsupported",
                                   static_cast<double>(bundle_scope_unsupported));
                }
            }
            // UnsupportedTargets is a hard failure in every mode (v1/v3):
            // the v3 post-EM null collision has its own counter and never
            // touches this slot.
            if (sch.secondary_missing_target != 0 || sch.unsupported_targets != 0) {
                schneider_fail("schneider_secondary_missing_target",
                               "Schneider secondary lookup missed an exact target channel (no alias allowed)",
                               static_cast<double>(sch.secondary_missing_target +
                                                   sch.unsupported_targets));
            }
            // Declared research approximations (reported, never failed):
            // post-EM null collisions resolve via track continuation.
            if (sch.secondary_post_em_null_collisions != 0) {
                report.approximations.push_back(
                    {"schneider_post_em_null_collisions",
                     "sampled candidates resolved as post-EM null collisions (track continues, no lookup)",
                     static_cast<double>(sch.secondary_post_em_null_collisions), 0.0});
            }
        }
        if (sch.secondary_below_domain != 0 || sch.secondary_above_domain != 0) {
            schneider_fail("schneider_secondary_out_of_domain",
                           "Schneider secondary query outside the exact channel energy domain",
                           static_cast<double>(sch.secondary_below_domain + sch.secondary_above_domain));
        }
        if (sch.secondary_energy_gap_misses != 0) {
            schneider_fail("schneider_secondary_energy_gap",
                           "Schneider secondary bracketing gap exceeds the fixed threshold",
                           static_cast<double>(sch.secondary_energy_gap_misses));
        }
        if (sch.secondary_empty_nodes != 0) {
            schneider_fail("schneider_secondary_empty_node",
                           "Schneider secondary lookup selected an empty event node",
                           static_cast<double>(sch.secondary_empty_nodes));
        }
        // Conservation invariants: every hazard must resolve to exactly one
        // replayed event or one explicit failure category.
        const auto primary_failures =
            sch.primary_missing_projectile + sch.primary_missing_target +
            sch.primary_below_domain + sch.primary_above_domain +
            sch.primary_energy_gap_misses + sch.primary_empty_nodes +
            sch.primary_post_em_null_collisions;
        if (sch.primary_post_em_null_collisions != 0) {
            report.approximations.push_back(
                {"schneider_primary_post_em_null_collisions",
                 "primary candidates resolved as post-EM null collisions (track continues, no lookup)",
                 static_cast<double>(sch.primary_post_em_null_collisions), 0.0});
        }
        if (sch.primary_hazards != sch.primary_events_replayed + primary_failures) {
            schneider_fail("schneider_primary_conservation",
                           "primary hazards != replayed + failure categories",
                           static_cast<double>(sch.primary_hazards));
        }
        // Conservation: every sampled candidate resolves to exactly one
        // replayed event, post-EM null collision, cutoff stop, or explicit
        // lookup-failure category.
        const auto secondary_failures =
            sch.secondary_missing_projectile + sch.secondary_missing_target +
            sch.secondary_below_domain + sch.secondary_above_domain +
            sch.secondary_energy_gap_misses + sch.secondary_empty_nodes +
            sch.secondary_stopped_before_replay + sch.secondary_post_em_null_collisions;
        if (sch.secondary_hazards != sch.secondary_events_replayed + secondary_failures) {
            schneider_fail("schneider_secondary_conservation",
                           "secondary hazards != replayed + failure categories",
                           static_cast<double>(sch.secondary_hazards));
        }
        // Strict equality: born counts role-0 charged products excluding Be6
        // (which has its own TopasCompatKill sink and is NOT counted in
        // born), so every born product must resolve to exactly one of
        // queued / cutoff / overflow. Both directions fail: lost products
        // (born > terminals) and phantom terminals (born < terminals).
        const auto primary_born_terminals =
            sch.primary_charged_products_queued + sch.primary_charged_cutoff_kills +
            sch.primary_queue_overflows;
        if (sch.primary_charged_products_born != primary_born_terminals) {
            schneider_fail("schneider_primary_born_conservation",
                           "primary charged born != queued + cutoff + overflow terminals (Be6 excluded on both sides)",
                           static_cast<double>(sch.primary_charged_products_born) -
                               static_cast<double>(primary_born_terminals));
        }
        const auto secondary_born_terminals =
            sch.secondary_charged_products_queued + sch.secondary_charged_cutoff_kills +
            sch.secondary_queue_overflows;
        if (sch.secondary_charged_products_born != secondary_born_terminals) {
            schneider_fail("schneider_secondary_born_conservation",
                           "secondary charged born != queued + cutoff + overflow terminals (Be6 excluded on both sides)",
                           static_cast<double>(sch.secondary_charged_products_born) -
                               static_cast<double>(secondary_born_terminals));
        }
        if (sch.queue_overflows != 0 || sch.primary_queue_overflows != 0 ||
            sch.secondary_queue_overflows != 0) {
            schneider_fail("schneider_queue_overflow",
                           "Schneider secondary queue overflow (shard must be split and rerun)",
                           static_cast<double>(sch.queue_overflows + sch.primary_queue_overflows +
                                               sch.secondary_queue_overflows));
        }
        if (sch.lookup_failure_energy_MeV > 0.0) {
            schneider_fail("schneider_lookup_failure_energy",
                           "non-zero energy absorbed by lookup failures",
                           sch.lookup_failure_energy_MeV);
        }
        if (result.energy_ledger.E_out_of_domain > 0.0) {
            schneider_fail("schneider_out_of_domain_energy",
                           "non-zero out-of-domain query energy",
                           result.energy_ledger.E_out_of_domain);
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
           << "{\n  \"schema_version\": 3,\n  \"status\": ";
    append_json_string(output, report.status());
    output << ",\n  \"run_mode\": ";
    append_json_string(output, run_mode_name(report.mode));
    output << ",\n  \"accepted\": " << (report.accepted ? "true" : "false")
           << ",\n  \"absolute_physical_energy_residual_MeV\": ";
    write_json_number(output, report.absolute_physical_energy_residual_MeV);
    output << ",\n  \"physical_relative_energy_residual\": ";
    write_json_number(output, report.physical_relative_energy_residual);
    output << ",\n  \"absolute_accounting_energy_residual_MeV\": ";
    write_json_number(output, report.absolute_accounting_energy_residual_MeV);
    output << ",\n  \"accounting_relative_energy_residual\": ";
    write_json_number(output, report.accounting_relative_energy_residual);
    output << ",\n  \"absolute_energy_residual_MeV\": ";
    write_json_number(output, report.absolute_energy_residual_MeV);
    output << ",\n  \"relative_energy_residual\": ";
    write_json_number(output, report.relative_energy_residual);
    output << ",\n  \"topas_reference_energy_sink_active\": "
           << (report.topas_reference_energy_sink_active ? "true" : "false");
    output
           << ",\n  \"queue_overflow_count\": " << report.queue_overflow_count
           << ",\n  \"queue_overflow_energy_MeV\": ";
    write_json_number(output, report.queue_overflow_energy_MeV);
    output << ",\n  \"voxel_scored_energy_MeV\": ";
    write_json_number(output, report.voxel_scored_energy_MeV);
    output << ",\n  \"in_grid_deposited_energy_MeV\": ";
    write_json_number(output, report.in_grid_deposited_energy_MeV);
    output << ",\n  \"outside_grid_deposited_energy_MeV\": ";
    write_json_number(output, report.outside_grid_deposited_energy_MeV);
    output << ",\n  \"grid_split_closure_ratio\": ";
    write_json_number(output, report.grid_split_closure_ratio);
    output << ",\n  \"voxel_to_ingrid_ratio\": ";
    write_json_number(output, report.voxel_to_ingrid_ratio);
    output << ",\n  \"voxel_to_total_deposited_ratio\": ";
    write_json_number(output, report.voxel_to_total_deposited_ratio);
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

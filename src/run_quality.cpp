#include "secondary_schedule.hpp"
#include "run_quality.hpp"
#include "secondary_rate_table.hpp"

#include <algorithm>
#include <cmath>
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
    report.primary_medium_scattering = config.primary_medium_scattering_model();
    report.secondary_medium_scattering = config.secondary_medium_scattering_model();
    if (!config.enable_secondary_energy_straggling)
        report.approximations.push_back({"secondary_mean_energy_loss",
            "Charged secondaries retain transport, scattering and nuclear reactions but use the continuous26 unrestricted electronic mean loss without fluctuation draws. Qualification of this mean-loss revision remains configuration-specific.",1.,0.});
    if (config.enable_minibeam && !config.minibeam_copper_fragment_enable_energy_straggling)
        report.approximations.push_back({"copper_fragment_full_fluctuations_disabled",
            "Cu/air fragment full electronic loss uses its mean without fluctuation or separate delta draws; Cu primary full fluctuations follow the primary switch. Carbon minibeam dose acceptance remains pending.",1.,0.});
    if(result.ct_broadbeam_step_cap_enabled)
        report.approximations.push_back({"ct_broadbeam_ion_step_cap",
            "Configured maximum ion step is enforced before CT face and nuclear competition for primary and charged fragments; FP32 Fermi-Eyges retained; ion fluctuations follow the configured primary/secondary switches",
            config.maximum_step_mm,0.});
    report.approximations.push_back({"unrestricted_electronic_csda",
            "Water/CT registered ions use native continuous26 unrestricted electronic stopping; CSDA and inverse derive from the same full response with continuous density-effect correction. Small losses use midpoint integration; larger losses use residual-range inversion. One Universal/Ion draw uses that full mean, native unrestricted dispersion and Tmax. Residual-range stepping has no medium ionization proposal clock. Cu/air use their native unrestricted electronic response; additional elastic recoils retain full electronic plus nuclear stopping",1.,0.});

    if (config.secondary_species_grouping && config.enable_inelastic &&
        config.enable_secondary_transport)
        report.approximations.push_back({"secondary_species_grouping_accuracy_pending",
            "GPU species index grouping; RNG identities preserved; production integration authorized; RT07575 scheduling dose difference 0.0203% of peak remains under investigation", 1., 0.});
    if (config.secondary_step_chunking)
        report.approximations.push_back({
            "secondary_step_chunking_accepted",
            std::to_string(kSecondarySegmentSteps) +
            " complete iterations per launch, stable survivor index compaction, tail below 8192; " +
            "user-accepted 2026-09-15 production interval; RT07575 compact+16 dose difference 0.00033% of peak vs 64-step (not a new TOPAS Gamma validation)",
            1., 0.});
    report.approximations.push_back({"unified_material_em_accuracy_pending",
            "Unified water/Schneider unrestricted electronic EM for all 18 charged ions; all ionization loss deposited locally; full-plan dose qualification remains configuration-specific",1.,0.});

    if(!config.all_ion_elastic_file.empty()) {
        report.approximations.push_back({"all_ion_elastic_candidate",
            "Finite TOPAS elastic event bank; additional heavy recoil isotopes use total stopping and MCS with EM-only transport; patient gamma and bank convergence validation pending",1.,0.});
    }
    if (config.enable_minibeam) {
        report.approximations.push_back({"copper_native_fluctuation",
            "Cu/air use native unrestricted electronic stopping, consistent range/inverse and full Universal/Ion fluctuations; no explicit delta transfers. Beamline losses remain explicit sinks",1.,0.});
    }
    if (config.enable_minibeam && config.minibeam_copper_fragment_native_transport) {
        report.approximations.push_back({"copper_ion_transport",
            "Cu charged fragments use full electronic stopping and optional full fluctuations, species Urban scattering and finite species elastic banks. Beamline losses and nuclear neutrals remain explicit sinks. Elastic bank convergence and carbon minibeam dose acceptance remain pending",1.,0.});
    } else if (config.enable_minibeam) {
        report.approximations.push_back({"minibeam_copper_candidate",
            "Copper primary transport uses the configured source ZA, elastic bank and nuclear event package; charged nuclear products use full native species electronic response in Copper and Highland secondary MCS. Nuclear neutrals remain an explicit beamline energy sink; dose acceptance pending",1.,0.});
    }

    if(config.fermi_eyges_gaussian_only)
        report.approximations.push_back({"gaussian_fermi_eyges_candidate",
            "Gaussian FE matches local angular/displacement second moments of the existing FE core+tail model; non-Gaussian tails and heterogeneous patient dose require independent validation",1.,0.});
    if(config.broadbeam_maximum_mean_loss_fraction != 0.01)
        report.approximations.push_back({"broadbeam_extended_mean_loss",
            "Research FE mean-loss cap relaxed; geometry faces, nuclear competition and sampled loss remain active; patient gamma requires validation",
            config.broadbeam_maximum_mean_loss_fraction,0.01});
    if(config.transport_profile=="broadbeam_fast") {
            report.approximations.push_back({"broadbeam_fast_local_electron_neutral_sink",
                "Broad beam uses local ionization deposition and no neutron transport; nuclear neutral energy remains explicitly untransported. Field edges and tissue interfaces require dose validation",1.,0.});
    }

    if (config.unified_water_nuclear_transport)
        report.approximations.push_back({"water_topas_match_pending",
            "Shared CINEL03 framework enabled for production; water TOPAS accuracy validation remains incomplete",1.,0.});
    // One physical ledger includes all explicit terminal sinks.
    const auto physical_raw_residual =
        result.initial_energy_MeV - result.total_deposited_energy_MeV -
        result.escaped_energy_MeV - result.beamline_removed_energy_MeV -
        result.untracked_nuclear_energy_MeV -
        result.primary_other_terminal_kinetic_MeV;
    report.absolute_physical_energy_residual_MeV = std::abs(physical_raw_residual);
    report.physical_relative_energy_residual =
        result.physical_relative_energy_balance_error();
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
        if ((config.quality_reject_any_queue_overflow) && (config.run_mode == RunMode::production)) {
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
    add_overflow("minibeam_copper_cascade_queue_overflow",
                 "minibeam Copper cascade",
                 result.minibeam.copper_fragment_cascade_queue_overflows,
                 0.0);

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
        // require split accounting for every nonzero deposited total.
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
        if (config.quality_reject_grid_closure && (total > 0.0 || split_sum > 0.0)) {
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
            result.generated_direct_secondary_energy_MeV,
            result.queued_secondary_energy_MeV,
            result.secondary_queue_overflow_energy_MeV,
            result.nuclear_energy_not_in_direct_secondaries_MeV,
            result.elastic_local_deposited_energy_MeV,
            result.elastic_queued_charged_energy_MeV,
            result.elastic_queue_overflow_energy_MeV,
            result.secondary_deposited_energy_MeV,
            result.secondary_escaped_energy_MeV,
            result.queued_cascade_energy_MeV,
            result.cascade_nuclear_energy_MeV,
            result.elapsed_seconds,
            result.primary_kernel_seconds,
            result.secondary_kernel_seconds,
            report.absolute_physical_energy_residual_MeV,
            report.physical_relative_energy_residual,
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
             }) {
            if (!finite_vector(*values)) {
                finite_issue(label);
            }
        }
    }

    if (std::isfinite(report.absolute_physical_energy_residual_MeV) &&
        std::isfinite(report.physical_relative_energy_residual)) {
        const auto allowed_residual = std::max(
            config.quality_maximum_absolute_energy_residual_MeV,
            config.quality_maximum_relative_energy_residual *
                std::abs(result.initial_energy_MeV));
        if (report.absolute_physical_energy_residual_MeV > allowed_residual) {
            add_issue({"energy_residual_exceeded",
                       "physical energy ledger residual exceeds the configured tolerance",
                       report.absolute_physical_energy_residual_MeV, allowed_residual}, true);
        }
    }

    if (std::isfinite(report.physical_relative_energy_residual)) {
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
    if ((config.is_schneider_ct_mode() || config.unified_water_nuclear_transport)) {
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
        // Current rate-registry scope coverage policy:
        // (a) missing_projectile fails only on rate-registry scope misses: actual
        //     lookup misses, or logged unsupported tracks whose (Z,A) IS in
        //     the binary rate registry. Out-of-scope isotopes (no event data on
        //     disk by design) deposit locally and are reported, not failed.
        //     Fail-closed: log truncation (dropped>0) refuses acceptance since scope cannot be proven.
        // (b) missing_target fails only on actual package-query misses.
        //     Sampler-empty draws (slowing left every channel domain between
        //     hazard and collision; no query issued) resolve via
        //     stopped-before-replay and never fail.
        std::set<std::pair<int, int>> projectile_registry;
        if (!result.schneider_unsupported_tracks.empty()) {
            for (const auto& key : read_secondary_rate_projectiles(config.ct_schneider_secondary_rate_file))
                projectile_registry.emplace(key.z, key.a);
        }
        if (sch.secondary_missing_projectile != 0) {
            schneider_fail("schneider_secondary_missing_projectile",
                           "Schneider secondary lookup hit an unsupported non-Be6 projectile",
                           static_cast<double>(sch.secondary_missing_projectile));
        }
        if (sch.unsupported_log_dropped != 0) {
            schneider_fail("schneider_unsupported_log_truncated",
                           "unsupported-track log truncated; rate-registry scope unprovable",
                           static_cast<double>(sch.unsupported_log_dropped));
        } else {
            std::uint64_t in_scope_unsupported = 0;
            for (const auto& track : result.schneider_unsupported_tracks) {
                if (track.projectile_z == 4 && track.projectile_a == 6) {
                    continue;  // Be6 TopasCompatKill exempt
                }
                if (projectile_registry.count({track.projectile_z, track.projectile_a}) != 0) {
                    ++in_scope_unsupported;
                }
            }
            if (in_scope_unsupported != 0) {
                schneider_fail("schneider_secondary_missing_projectile",
                               "Schneider secondary rate-registry scope projectile unsupported",
                               static_cast<double>(in_scope_unsupported));
            }
        }
        // UnsupportedTargets is always a hard failure:
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
           << "{\n  \"schema_version\": 4,\n  \"status\": ";
    append_json_string(output, report.status());
    output << ",\n  \"run_mode\": ";
    append_json_string(output, run_mode_name(report.mode));
    output << ",\n  \"medium_scattering\": {\"primary\": ";
    append_json_string(output, report.primary_medium_scattering);
    output << ", \"secondary\": ";
    append_json_string(output, report.secondary_medium_scattering);
    output << "}";
    output << ",\n  \"accepted\": " << (report.accepted ? "true" : "false")
           << ",\n  \"absolute_physical_energy_residual_MeV\": ";
    write_json_number(output, report.absolute_physical_energy_residual_MeV);
    output << ",\n  \"physical_relative_energy_residual\": ";
    write_json_number(output, report.physical_relative_energy_residual);
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

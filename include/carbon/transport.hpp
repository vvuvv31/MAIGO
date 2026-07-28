#pragma once

#include "carbon/cascade_package.hpp"
#include "carbon/cross_section.hpp"
#include "carbon/neutral_package.hpp"
#include "carbon/reaction_package.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/transport_config.hpp"
#include "carbon/transport_profile.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace carbon {

class SyclTransportContext;

struct MinibeamDiagnostics {
    static constexpr std::size_t slit_count = 15;
    static constexpr std::size_t touched_energy_bin_count = 12;
    bool enabled{false};
    std::uint64_t incident_histories{0};
    std::uint64_t direct_air_slit_histories{0};
    std::uint64_t copper_touched_histories{0};
    std::uint64_t copper_nuclear_interactions{0};
    std::uint64_t copper_generated_direct_secondaries{0};
    std::uint64_t copper_charged_survivors{0};
    std::uint64_t copper_neutral_survivors{0};
    std::uint64_t water_entrance_primary_c12{0};
    double beamline_removed_energy_MeV{0.0};
    double copper_charged_survivor_energy_MeV{0.0};
    double copper_neutral_survivor_energy_MeV{0.0};
    double direct_air_primary_energy_MeV{0.0};
    double copper_touched_primary_energy_MeV{0.0};
    // C, B, Be, Li, He, p, d, t, other.
    std::array<std::uint64_t, 9> copper_charged_survivors_by_species{};
    std::array<double, 9> copper_charged_survivor_energy_by_species_MeV{};
    std::array<std::uint64_t, slit_count>
        water_entrance_primary_c12_by_slit{};
    std::array<std::uint64_t, slit_count>
        collimator_entrance_primary_c12_by_slit{};
    std::array<std::uint64_t, slit_count>
        direct_air_primary_c12_by_slit{};
    std::array<std::uint64_t, touched_energy_bin_count>
        copper_touched_primary_energy_histogram{};
    double energy_sum_MeV{0.0};
    double energy_squared_sum_MeV2{0.0};
    double x_sum_mm{0.0};
    double x_squared_sum_mm2{0.0};
    double y_sum_mm{0.0};
    double y_squared_sum_mm2{0.0};
    double direction_x_sum{0.0};
    double direction_x_squared_sum{0.0};
    double direction_y_sum{0.0};
    double direction_y_squared_sum{0.0};
};

struct TransportResult {
    std::vector<double> deposited_energy_MeV;
    std::vector<double> voxel_deposited_energy_MeV;
    // Category-major layout: category * number_of_voxels + voxel index.
    std::vector<double> charged_origin_voxel_deposited_energy_MeV;
    std::vector<double> neutral_origin_voxel_deposited_energy_MeV;
    std::vector<double> primary_c12_deposited_energy_MeV;
    std::vector<double> secondary_carbon_deposited_energy_MeV;
    std::vector<double> boron_deposited_energy_MeV;
    std::vector<double> beryllium_deposited_energy_MeV;
    std::vector<double> lithium_deposited_energy_MeV;
    std::vector<double> helium_deposited_energy_MeV;
    std::vector<double> proton_deposited_energy_MeV;
    std::vector<double> other_charged_deposited_energy_MeV;
    // HadronLET raw dose-weighted moments. Numerator unit:
    // MeV * MeV/mm/(g/cm3); denominator unit: MeV.
    std::vector<double> primary_c12_letd_numerator;
    std::vector<double> primary_c12_letd_denominator;
    std::vector<double> all_hadron_letd_numerator;
    std::vector<double> all_hadron_letd_denominator;
    // Category-major primary-C12, secondary C, B, Be, Li, He, p, other.
    std::vector<double> charged_origin_letd_numerator;
    std::vector<double> charged_origin_letd_denominator;
    // Optional category-major p, d, t, He-3, He-4 LET moments.
    std::vector<double> light_isotope_letd_numerator;
    std::vector<double> light_isotope_letd_denominator;
    // Optional light-isotope birth spectra (p/d/t/He-3/He-4). Layout documented
    // in particle.hpp birth_* constants. Counts are event tallies (not /primary).
    // Histograms are species × generation × bin (see birth_hist_index).
    std::vector<std::uint64_t> birth_counts_by_generation;  // cat * gen_bins
    std::vector<double> birth_ke_sum_MeV_by_generation;     // cat * gen_bins
    std::vector<std::uint64_t> birth_mevu_hist;             // cat*gen*mevu_bins
    std::vector<std::uint64_t> birth_depth_hist;            // cat*gen*depth_bins
    std::vector<std::uint64_t> birth_cos_hist;              // cat*gen*cos_bins
    std::vector<std::uint64_t> birth_parent_mevu_hist;      // cat*gen*parent_mevu
    std::vector<std::uint64_t> birth_parent_z_hist;         // cat*gen*parent_z
    // cat*gen*(parent_mevu_bins * product_mevu_bins)
    std::vector<std::uint64_t> birth_parent_product_mevu_hist;
    // Same four moments on the optional voxel grid (z-major, x fastest).
    std::vector<double> primary_c12_voxel_letd_numerator;
    std::vector<double> primary_c12_voxel_letd_denominator;
    std::vector<double> all_hadron_voxel_letd_numerator;
    std::vector<double> all_hadron_voxel_letd_denominator;
    std::vector<double> neutron_origin_deposited_energy_MeV;
    std::vector<double> gamma_origin_deposited_energy_MeV;
    double initial_energy_MeV{0.0};
    double total_deposited_energy_MeV{0.0};
    double escaped_energy_MeV{0.0};
    // Energy lost/deposited/terminated before the scored phantom by the
    // optional minibeam beamline.
    double beamline_removed_energy_MeV{0.0};
    double untracked_nuclear_energy_MeV{0.0};
    std::uint64_t nuclear_interactions{0};
    std::uint64_t sampled_reaction_packages{0};
    std::uint64_t generated_direct_secondaries{0};
    std::uint64_t queued_secondaries{0};
    std::uint64_t secondary_queue_overflow{0};
    double generated_direct_secondary_energy_MeV{0.0};
    double queued_secondary_energy_MeV{0.0};
    double secondary_queue_overflow_energy_MeV{0.0};
    double untransported_neutral_energy_MeV{0.0};
    double untransported_unsupported_charged_energy_MeV{0.0};
    double nuclear_energy_not_in_direct_secondaries_MeV{0.0};
    std::uint64_t transported_secondaries{0};
    std::uint64_t secondary_transport_steps{0};
    double secondary_deposited_energy_MeV{0.0};
    double secondary_escaped_energy_MeV{0.0};
    std::uint64_t cascade_interactions{0};
    std::uint64_t generated_cascade_products{0};
    std::uint64_t queued_cascade_secondaries{0};
    std::uint64_t cascade_queue_overflow{0};
    double queued_cascade_energy_MeV{0.0};
    double cascade_nuclear_energy_MeV{0.0};
    std::uint64_t queued_neutrals{0};
    std::uint64_t neutral_queue_overflow{0};
    std::uint64_t transported_neutrals{0};
    std::uint64_t neutral_interactions{0};
    std::uint64_t neutral_transport_steps{0};
    double queued_neutral_energy_MeV{0.0};
    double neutral_queue_overflow_energy_MeV{0.0};
    double neutral_deposited_energy_MeV{0.0};
    double neutral_escaped_energy_MeV{0.0};
    double residual_neutral_energy_MeV{0.0};
    double charged_from_neutral_energy_MeV{0.0};
    std::uint64_t total_steps{0};
    double elapsed_seconds{0.0};
    double primary_kernel_seconds{0.0};
    double secondary_kernel_seconds{0.0};
    double neutral_kernel_seconds{0.0};
    double charged_after_neutral_kernel_seconds{0.0};
    std::string backend;
    MinibeamDiagnostics minibeam;
    // Populated only when built with CARBON_TRANSPORT_PROFILE=1.
    TransportProfile profile;

    [[nodiscard]] double relative_energy_balance_error() const noexcept;
};

[[nodiscard]] double choose_step_mm(double energy_MeV,
                                    double stopping_power_MeV_per_mm,
                                    double maximum_step_mm,
                                    double maximum_relative_energy_loss);

TransportResult transport_serial(const TransportConfig& config,
                                 const StoppingPowerTable& stopping_power,
                                 const CrossSectionTable& cross_section);

#ifdef CARBON_HAS_SYCL
class SyclTransportContext {
public:
    explicit SyclTransportContext(const std::string& device_name);
    ~SyclTransportContext();

    SyclTransportContext(SyclTransportContext&&) noexcept;
    SyclTransportContext& operator=(SyclTransportContext&&) noexcept;
    SyclTransportContext(const SyclTransportContext&) = delete;
    SyclTransportContext& operator=(const SyclTransportContext&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend TransportResult transport_sycl(
        const TransportConfig&, const StoppingPowerTable&, const CrossSectionTable&,
        const std::string&, const ReactionPackageTable*, const CascadePackageTable*,
        const NeutralPackageTable*, SyclTransportContext*);
#if defined(CARBON_ENABLE_MINIBEAM)
    friend TransportResult transport_sycl_legacy(
        const TransportConfig&, const StoppingPowerTable&, const CrossSectionTable&,
        const std::string&, const ReactionPackageTable*, const CascadePackageTable*,
        const NeutralPackageTable*, SyclTransportContext*);
    friend TransportResult transport_sycl_minibeam(
        const TransportConfig&, const StoppingPowerTable&, const CrossSectionTable&,
        const std::string&, const ReactionPackageTable*, const CascadePackageTable*,
        const NeutralPackageTable*, SyclTransportContext*);
#endif
};

TransportResult transport_sycl(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const CrossSectionTable& cross_section,
                               const std::string& device_name,
                               const ReactionPackageTable* reaction_packages = nullptr,
                               const CascadePackageTable* cascade_packages = nullptr,
                               const NeutralPackageTable* neutral_packages = nullptr,
                               SyclTransportContext* context = nullptr);
#if defined(CARBON_ENABLE_MINIBEAM)
// Dual kernels: legacy is the master-compatible water/CT path; minibeam is the
// Copper beamline path. transport_sycl() dispatches at runtime.
TransportResult transport_sycl_legacy(
    const TransportConfig& config, const StoppingPowerTable& stopping_power,
    const CrossSectionTable& cross_section, const std::string& device_name,
    const ReactionPackageTable* reaction_packages = nullptr,
    const CascadePackageTable* cascade_packages = nullptr,
    const NeutralPackageTable* neutral_packages = nullptr,
    SyclTransportContext* context = nullptr);
TransportResult transport_sycl_minibeam(
    const TransportConfig& config, const StoppingPowerTable& stopping_power,
    const CrossSectionTable& cross_section, const std::string& device_name,
    const ReactionPackageTable* reaction_packages = nullptr,
    const CascadePackageTable* cascade_packages = nullptr,
    const NeutralPackageTable* neutral_packages = nullptr,
    SyclTransportContext* context = nullptr);
#endif
std::string describe_sycl_device(const std::string& device_name);
#endif

}  // namespace carbon

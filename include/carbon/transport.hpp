#pragma once

#include "carbon/cascade_package.hpp"
#include "carbon/cross_section.hpp"
#include "carbon/neutral_package.hpp"
#include "carbon/reaction_package.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/transport_config.hpp"
#include "carbon/transport_profile.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace carbon {

class SyclTransportContext;

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
    std::vector<double> neutron_origin_deposited_energy_MeV;
    std::vector<double> gamma_origin_deposited_energy_MeV;
    double initial_energy_MeV{0.0};
    double total_deposited_energy_MeV{0.0};
    double escaped_energy_MeV{0.0};
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
};

TransportResult transport_sycl(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const CrossSectionTable& cross_section,
                               const std::string& device_name,
                               const ReactionPackageTable* reaction_packages = nullptr,
                               const CascadePackageTable* cascade_packages = nullptr,
                               const NeutralPackageTable* neutral_packages = nullptr,
                               SyclTransportContext* context = nullptr);
std::string describe_sycl_device(const std::string& device_name);
#endif

}  // namespace carbon

#include "carbon/transport.hpp"

#include <stdexcept>

#ifdef CARBON_HAS_SYCL

namespace carbon {

TransportResult transport_sycl(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const CrossSectionTable& cross_section,
                               const std::string& device_name,
                               const ReactionPackageTable* reaction_packages,
                               const CascadePackageTable* cascade_packages,
                               const NeutralPackageTable* neutral_packages,
                               SyclTransportContext* context,
                               const CrossSectionTable* elastic_cross_section,
                               const ElasticPackageTable* elastic_packages) {
#if defined(CARBON_ENABLE_MINIBEAM)
    if (config.enable_minibeam) {
        if (config.enable_primary_elastic_interactions) {
            throw std::invalid_argument(
                "primary elastic interactions are supported only by the legacy SYCL path");
        }
        return transport_sycl_minibeam(config, stopping_power, cross_section,
                                       device_name, reaction_packages,
                                       cascade_packages, neutral_packages,
                                       context, elastic_cross_section, elastic_packages);
    }
    // ON build + minibeam:false must use the master-compatible legacy kernel so
    // ordinary CT/water cases do not pay Copper register/code-size cost.
    return transport_sycl_legacy(config, stopping_power, cross_section,
                                 device_name, reaction_packages,
                                 cascade_packages, neutral_packages,
                                 context, elastic_cross_section, elastic_packages);
#else
    // OFF builds only compile the legacy TU, which exports transport_sycl.
    // This TU is not linked when CARBON_ENABLE_MINIBEAM is off.
    (void)config;
    (void)stopping_power;
    (void)cross_section;
    (void)device_name;
    (void)reaction_packages;
    (void)cascade_packages;
    (void)neutral_packages;
    (void)elastic_cross_section;
    (void)elastic_packages;
    (void)context;
    throw std::logic_error(
        "transport_sycl_dispatch.cpp should not be linked without "
        "CARBON_ENABLE_MINIBEAM");
#endif
}

}  // namespace carbon

#endif

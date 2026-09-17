#pragma once

#include "carbon/cross_section.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/topas_spots.hpp"
#include "carbon/transport.hpp"
#include "carbon/transport_config.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace carbon {

struct UpstreamAirLossAudit {
    double distance_to_entrance_mm{0.0};
    double energy_loss_MeV{0.0};
};

void accumulate_transport_result(TransportResult& total, const TransportResult& part);

void apply_spot_to_config(TransportConfig& config,
                          const TopasSpotPlan& plan,
                          const TopasSpot& spot,
                          std::size_t spot_index,
                          std::uint64_t base_seed,
                          const StoppingPowerTable* upstream_air_stopping_power = nullptr,
                          UpstreamAirLossAudit* upstream_air_audit = nullptr);

PrimarySpotBatchEntry make_spot_batch_entry(const TransportConfig& spot_config,
                                          std::uint64_t history_begin);

TransportResult run_transport(
    const TransportConfig& config,
    const StoppingPowerTable& stopping_power,
    const CrossSectionTable& cross_section,
    SyclTransportContext* sycl_context = nullptr);

}  // namespace carbon

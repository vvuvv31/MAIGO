#pragma once
#include "carbon/stopping_power.hpp"
#include "carbon/transport.hpp"
namespace carbon {
TransportResult run_transport(const TransportConfig&, const StoppingPowerTable&,
                              SyclTransportContext* = nullptr);
}

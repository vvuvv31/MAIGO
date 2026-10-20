#pragma once
#include "stopping_power.hpp"
#include "transport.hpp"
namespace carbon {
TransportResult run_transport(const TransportConfig&, const StoppingPowerTable&,
                              SyclTransportContext* = nullptr);
}

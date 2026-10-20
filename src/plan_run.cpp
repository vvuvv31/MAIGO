#include "plan_run.hpp"
namespace carbon {
TransportResult run_transport(const TransportConfig& config,
                              const StoppingPowerTable& stopping_power,
                              SyclTransportContext* context) {
    return transport_sycl(config,stopping_power,config.device,context);
}
}

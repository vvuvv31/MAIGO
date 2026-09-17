SyclTransportContext::SyclTransportContext(const std::string& device_name)
    : impl_(std::make_unique<Impl>(device_name)) {}

SyclTransportContext::~SyclTransportContext() = default;
SyclTransportContext::SyclTransportContext(SyclTransportContext&&) noexcept = default;
SyclTransportContext& SyclTransportContext::operator=(SyclTransportContext&&) noexcept = default;

#include "carbon/elastic_device_storage.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <sstream>
#include <utility>

namespace carbon {

std::size_t checked_elastic_device_bytes(const std::size_t element_count,
                                         const std::size_t element_size) {
    if (element_size == 0) {
        if (element_count != 0) {
            throw std::invalid_argument("ELPKG device element size must be non-zero");
        }
        return 0;
    }
    if (element_count > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error("ELPKG device allocation byte count overflow");
    }
    return element_count * element_size;
}

ElasticPackageFlatData flatten_elastic_package(const ElasticPackageTable& package) {
    ElasticPackageFlatData result;
    result.energy_bins = package.energy_bins();
    result.events = package.events();
    result.products = package.products();
    result.minimum_energy_MeV_per_u = package.minimum_energy_MeV_per_u();
    result.energy_bin_width_MeV_per_u = package.energy_bin_width_MeV_per_u();
    return result;
}

#ifdef CARBON_HAS_SYCL

#include <new>

namespace {

template <typename T>
T* allocate_device(const std::size_t count, sycl::queue& queue) {
    (void)checked_elastic_device_bytes(count, sizeof(T));
    if (count == 0) return nullptr;
    auto* pointer = sycl::malloc_device<T>(count, queue);
    if (pointer == nullptr) {
        throw std::bad_alloc();
    }
    return pointer;
}

template <typename T>
void copy_to_device(const std::vector<T>& source, T* destination, sycl::queue& queue) {
    if (source.empty()) return;
    const auto bytes = checked_elastic_device_bytes(source.size(), sizeof(T));
    queue.memcpy(destination, source.data(), bytes);
}

template <typename T>
void free_device(T*& pointer, const sycl::queue& queue) noexcept {
    if (pointer == nullptr) return;
    try {
        sycl::free(pointer, queue);
    } catch (...) {
        // Destruction must not throw.  The queue/context was checked before
        // normal release; this is only a last-resort destructor path.
    }
    pointer = nullptr;
}

}  // namespace

ElasticPackageDeviceStorage::ElasticPackageDeviceStorage(const ElasticPackageTable& package,
                                                         sycl::queue queue)
    : ElasticPackageDeviceStorage(flatten_elastic_package(package), std::move(queue)) {}

ElasticPackageDeviceStorage::ElasticPackageDeviceStorage(ElasticPackageFlatData data,
                                                         sycl::queue queue)
    : queue_(std::move(queue)), host_data_(std::move(data)) {
    validate_host_data();
    try {
        upload();
    } catch (...) {
        release_noexcept();
        throw;
    }
}

ElasticPackageDeviceStorage::~ElasticPackageDeviceStorage() noexcept { release_noexcept(); }

ElasticPackageDeviceStorage::ElasticPackageDeviceStorage(ElasticPackageDeviceStorage&& other)
    : queue_(other.queue_),
      host_data_(std::move(other.host_data_)),
      energy_bins_device_(other.energy_bins_device_),
      events_device_(other.events_device_),
      products_device_(other.products_device_) {
    other.energy_bins_device_ = nullptr;
    other.events_device_ = nullptr;
    other.products_device_ = nullptr;
    other.host_data_ = {};
}

ElasticPackageDeviceStorage& ElasticPackageDeviceStorage::operator=(
    ElasticPackageDeviceStorage&& other) {
    if (this == &other) return *this;
    reset();
    queue_ = other.queue_;
    host_data_ = std::move(other.host_data_);
    energy_bins_device_ = other.energy_bins_device_;
    events_device_ = other.events_device_;
    products_device_ = other.products_device_;
    other.energy_bins_device_ = nullptr;
    other.events_device_ = nullptr;
    other.products_device_ = nullptr;
    other.host_data_ = {};
    return *this;
}

void ElasticPackageDeviceStorage::validate_host_data() const {
    if (host_data_.energy_bins.size() > std::numeric_limits<std::uint32_t>::max() ||
        host_data_.events.size() > std::numeric_limits<std::uint32_t>::max() ||
        host_data_.products.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("ELPKG device view count exceeds uint32 offset capacity");
    }
    if (host_data_.energy_bins.empty()) {
        if (!host_data_.events.empty() || !host_data_.products.empty()) {
            throw std::invalid_argument("ELPKG empty energy bins cannot own events/products");
        }
        return;
    }
    std::uint64_t expected_event_offset = 0;
    for (const auto& bin : host_data_.energy_bins) {
        if (bin.event_offset != expected_event_offset ||
            static_cast<std::uint64_t>(bin.event_offset) + bin.event_count >
                host_data_.events.size()) {
            throw std::invalid_argument("ELPKG energy-bin event range is not flat/contiguous");
        }
        expected_event_offset += bin.event_count;
    }
    if (expected_event_offset != host_data_.events.size()) {
        throw std::invalid_argument("ELPKG energy-bin event coverage mismatch");
    }
    std::uint64_t expected_product_offset = 0;
    for (const auto& event : host_data_.events) {
        if (event.product_offset != expected_product_offset ||
            static_cast<std::uint64_t>(event.product_offset) + event.product_count >
                host_data_.products.size()) {
            throw std::invalid_argument("ELPKG event product range is not flat/contiguous");
        }
        expected_product_offset += event.product_count;
    }
    if (expected_product_offset != host_data_.products.size()) {
        throw std::invalid_argument("ELPKG event product coverage mismatch");
    }
}

void ElasticPackageDeviceStorage::upload() {
    energy_bins_device_ =
        allocate_device<ElasticEnergyBin>(host_data_.energy_bins.size(), queue_);
    events_device_ = allocate_device<ElasticEvent>(host_data_.events.size(), queue_);
    products_device_ = allocate_device<ElasticProduct>(host_data_.products.size(), queue_);
    copy_to_device(host_data_.energy_bins, energy_bins_device_, queue_);
    copy_to_device(host_data_.events, events_device_, queue_);
    copy_to_device(host_data_.products, products_device_, queue_);
    queue_.wait_and_throw();
}

void ElasticPackageDeviceStorage::require_same_context(const sycl::queue& release_queue) const {
    if (queue_.get_context() != release_queue.get_context()) {
        throw std::invalid_argument("ELPKG device storage queue context mismatch");
    }
}

void ElasticPackageDeviceStorage::release(sycl::queue& release_queue) {
    require_same_context(release_queue);
    std::exception_ptr asynchronous_failure;
    try {
        release_queue.wait_and_throw();
    } catch (...) {
        asynchronous_failure = std::current_exception();
    }
    free_device(energy_bins_device_, release_queue);
    free_device(events_device_, release_queue);
    free_device(products_device_, release_queue);
    host_data_ = {};
    if (asynchronous_failure) std::rethrow_exception(asynchronous_failure);
}

void ElasticPackageDeviceStorage::release_noexcept() noexcept {
    try {
        release(queue_);
    } catch (...) {
        free_device(energy_bins_device_, queue_);
        free_device(events_device_, queue_);
        free_device(products_device_, queue_);
        host_data_ = {};
    }
}

void ElasticPackageDeviceStorage::reset() { release(queue_); }

void ElasticPackageDeviceStorage::reset(const sycl::queue& release_queue) {
    require_same_context(release_queue);
    auto release_queue_copy = release_queue;
    release(release_queue_copy);
}

bool ElasticPackageDeviceStorage::empty() const noexcept {
    return host_data_.energy_bins.empty() && host_data_.events.empty() &&
           host_data_.products.empty();
}

const ElasticPackageFlatData& ElasticPackageDeviceStorage::host_data() const noexcept {
    return host_data_;
}

ElasticPackageDeviceView ElasticPackageDeviceStorage::device_view() const noexcept {
    return ElasticPackageDeviceView{energy_bins_device_, host_data_.energy_bins.size(),
                                    events_device_, host_data_.events.size(), products_device_,
                                    host_data_.products.size(), host_data_.minimum_energy_MeV_per_u,
                                    host_data_.energy_bin_width_MeV_per_u};
}

const sycl::queue& ElasticPackageDeviceStorage::queue() const noexcept { return queue_; }

#endif  // CARBON_HAS_SYCL

}  // namespace carbon

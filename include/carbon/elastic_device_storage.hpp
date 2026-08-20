#pragma once

#include "carbon/elastic_sampling.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef CARBON_HAS_SYCL
#include <sycl/sycl.hpp>
#endif

namespace carbon {

// Owning, host-side copy of the flat ELPKG arrays.  The records are kept in
// their device-compatible POD form so upload does not require a conversion
// kernel or host/device-specific representation.
struct ElasticPackageFlatData {
    std::vector<ElasticEnergyBin> energy_bins;
    std::vector<ElasticEvent> events;
    std::vector<ElasticProduct> products;
    float minimum_energy_MeV_per_u{0.0F};
    float energy_bin_width_MeV_per_u{1.0F};
};

static_assert(std::is_trivially_copyable_v<ElasticEnergyBin>);
static_assert(std::is_trivially_copyable_v<ElasticEvent>);
static_assert(std::is_trivially_copyable_v<ElasticProduct>);
static_assert(std::is_standard_layout_v<ElasticEnergyBin>);
static_assert(std::is_standard_layout_v<ElasticEvent>);
static_assert(std::is_standard_layout_v<ElasticProduct>);
static_assert(sizeof(ElasticEnergyBin) == 16);
static_assert(sizeof(ElasticEvent) == 44);
static_assert(sizeof(ElasticProduct) == 36);
static_assert(offsetof(ElasticEnergyBin, event_offset) == 8);
static_assert(offsetof(ElasticEnergyBin, event_count) == 12);
static_assert(offsetof(ElasticEvent, product_offset) == 28);
static_assert(offsetof(ElasticEvent, product_count) == 32);
static_assert(offsetof(ElasticProduct, kinetic_energy_MeV) == 8);
static_assert(offsetof(ElasticProduct, transport_disposition) == 32);
static_assert(std::is_trivially_copyable_v<ElasticPackageDeviceView>);
static_assert(std::is_standard_layout_v<ElasticPackageDeviceView>);

// Throws std::overflow_error before a byte count can wrap.  This is public so
// callers and tests can validate counts without attempting an allocation.
[[nodiscard]] std::size_t checked_elastic_device_bytes(std::size_t element_count,
                                                       std::size_t element_size);

[[nodiscard]] ElasticPackageFlatData flatten_elastic_package(
    const ElasticPackageTable& package);

// Device ownership is deliberately separate from transport.  It owns only
// ELPKG flat arrays and publishes a non-owning device view after all copies
// have completed.  The class is non-copyable and movable.
#ifdef CARBON_HAS_SYCL

class ElasticPackageDeviceStorage final {
public:
    ElasticPackageDeviceStorage(const ElasticPackageTable& package, sycl::queue queue);
    ElasticPackageDeviceStorage(ElasticPackageFlatData data, sycl::queue queue);
    ~ElasticPackageDeviceStorage() noexcept;

    ElasticPackageDeviceStorage(const ElasticPackageDeviceStorage&) = delete;
    ElasticPackageDeviceStorage& operator=(const ElasticPackageDeviceStorage&) = delete;
    ElasticPackageDeviceStorage(ElasticPackageDeviceStorage&& other);
    ElasticPackageDeviceStorage& operator=(ElasticPackageDeviceStorage&& other);

    // Waits for queued transfers and releases all allocations.  The optional
    // queue must belong to the same context as the allocation queue; this
    // catches accidental freeing through an unrelated SYCL context.
    void reset();
    void reset(const sycl::queue& release_queue);

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] const ElasticPackageFlatData& host_data() const noexcept;
    [[nodiscard]] ElasticPackageDeviceView device_view() const noexcept;
    [[nodiscard]] const sycl::queue& queue() const noexcept;

private:
    void upload();
    void validate_host_data() const;
    void release_noexcept() noexcept;
    void release(sycl::queue& release_queue);
    void require_same_context(const sycl::queue& release_queue) const;

    sycl::queue queue_;
    ElasticPackageFlatData host_data_;
    ElasticEnergyBin* energy_bins_device_{nullptr};
    ElasticEvent* events_device_{nullptr};
    ElasticProduct* products_device_{nullptr};
};

#endif  // CARBON_HAS_SYCL

}  // namespace carbon

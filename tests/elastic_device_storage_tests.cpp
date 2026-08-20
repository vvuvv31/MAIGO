#include "carbon/elastic_device_storage.hpp"

#ifdef CARBON_HAS_SYCL
#include "carbon/device.hpp"
#include <sycl/sycl.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

carbon::ElasticPackageFlatData fixture() {
    carbon::ElasticPackageFlatData data;
    data.energy_bins = {{0.0F, 1.0F, 0U, 2U}, {1.0F, 2.0F, 2U, 1U}};
    data.events = {{1, 1, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0U, 0U, 1, 0},
                   {1, 1, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0U, 0U, 1, 0},
                   {1, 1, 1.0F, 0.5F, 0.0F, 1.0F, 0.0F, 0.25F, 0U, 1U, 1, 0}};
    data.products = {{2212, 1, 1, 0.25F, 0.0F, 0.0F, 1.0F, 1.0F, 0, 1}};
    data.minimum_energy_MeV_per_u = 0.0F;
    data.energy_bin_width_MeV_per_u = 1.0F;
    return data;
}

#pragma pack(push, 1)
struct TestElasticHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint32_t energy_bin_record_size;
    std::uint32_t event_record_size;
    std::uint32_t product_record_size;
    std::uint32_t energy_bin_count;
    std::uint32_t flags;
    std::uint64_t event_count;
    std::uint64_t product_count;
    std::uint64_t expected_file_size;
};
struct TestElasticBin {
    float minimum_energy_MeV_per_u;
    float maximum_energy_MeV_per_u;
    std::uint32_t event_offset;
    std::uint32_t event_count;
};
struct TestElasticEvent {
    std::int16_t projectile_atomic_number;
    std::int16_t projectile_mass_number;
    float incident_energy_MeV_per_u;
    float outgoing_projectile_energy_MeV_per_u;
    float outgoing_direction_x;
    float outgoing_direction_y;
    float outgoing_direction_z;
    float local_deposit_MeV;
    std::uint32_t product_offset;
    std::uint32_t product_count;
    std::int32_t continuation;
    std::int32_t generation;
};
struct TestElasticProduct {
    std::int32_t pdg_id;
    std::int16_t atomic_number;
    std::int16_t mass_number;
    float kinetic_energy_MeV;
    float direction_x;
    float direction_y;
    float direction_z;
    float charge_e;
    std::int32_t generation;
    std::int32_t transport_disposition;
};
#pragma pack(pop)

void test_flatten_round_trip() {
    const auto path = std::filesystem::temp_directory_path() /
                      "carbon_elastic_device_storage_flatten.bin";
    const TestElasticHeader header{{'E', 'L', 'P', 'K', 'G', '0', '1', '\0'},
                                   1U,
                                   60U,
                                   16U,
                                   44U,
                                   36U,
                                   1U,
                                   0U,
                                   1U,
                                   1U,
                                   156U};
    const TestElasticBin bin{0.0F, 1.0F, 0U, 1U};
    const TestElasticEvent event{1, 1, 0.5F, 0.25F, 0.0F, 0.0F, 1.0F, 0.0F, 0U, 1U, 1, 0};
    const TestElasticProduct product{2212, 1, 1, 0.25F, 0.0F, 0.0F, 1.0F, 1.0F, 0, 1};
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        require(static_cast<bool>(output), "cannot create flatten fixture");
        output.write(reinterpret_cast<const char*>(&header), sizeof(header));
        output.write(reinterpret_cast<const char*>(&bin), sizeof(bin));
        output.write(reinterpret_cast<const char*>(&event), sizeof(event));
        output.write(reinterpret_cast<const char*>(&product), sizeof(product));
    }
    try {
        const auto package = carbon::ElasticPackageTable::from_binary(path);
        const auto flat = carbon::flatten_elastic_package(package);
        require(flat.energy_bins.size() == 1U && flat.events.size() == 1U &&
                    flat.products.size() == 1U,
                "flattened ELPKG shape mismatch");
        require(std::memcmp(flat.energy_bins.data(), package.energy_bins().data(),
                             sizeof(carbon::ElasticEnergyBin)) == 0,
                "flattened bin differs from reader");
        require(std::memcmp(flat.events.data(), package.events().data(),
                             sizeof(carbon::ElasticEvent)) == 0,
                "flattened event differs from reader");
        require(std::memcmp(flat.products.data(), package.products().data(),
                             sizeof(carbon::ElasticProduct)) == 0,
                "flattened product differs from reader");
    } catch (...) {
        std::filesystem::remove(path);
        throw;
    }
    std::filesystem::remove(path);
}

void test_flatten_and_overflow() {
    const auto input = fixture();
    require(input.energy_bins.size() == 2U && input.events.size() == 3U &&
                input.products.size() == 1U,
            "fixture shape changed");
    require(carbon::checked_elastic_device_bytes(3U, sizeof(carbon::ElasticEvent)) ==
                3U * sizeof(carbon::ElasticEvent),
            "device byte count is incorrect");
    bool overflow = false;
    try {
        (void)carbon::checked_elastic_device_bytes(
            std::numeric_limits<std::size_t>::max(), sizeof(carbon::ElasticEvent));
    } catch (const std::overflow_error&) {
        overflow = true;
    }
    require(overflow, "device byte overflow was not rejected");
    bool bad_element_size = false;
    try {
        (void)carbon::checked_elastic_device_bytes(1U, 0U);
    } catch (const std::invalid_argument&) {
        bad_element_size = true;
    }
    require(bad_element_size, "zero element size was not rejected");
}

#ifdef CARBON_HAS_SYCL
void test_storage_roundtrip() {
    auto storage = carbon::ElasticPackageDeviceStorage(fixture(), carbon::make_sycl_queue("cpu"));
    const auto view = storage.device_view();
    require(view.energy_bin_count == 2U && view.event_count == 3U && view.product_count == 1U,
            "device view counts are incorrect");
    auto queue = storage.queue();
    auto* bins = sycl::malloc_shared<carbon::ElasticEnergyBin>(2U, queue);
    auto* events = sycl::malloc_shared<carbon::ElasticEvent>(3U, queue);
    auto* products = sycl::malloc_shared<carbon::ElasticProduct>(1U, queue);
    require(bins != nullptr && events != nullptr && products != nullptr,
            "round-trip output allocation failed");
    queue.memcpy(bins, view.energy_bins, 2U * sizeof(*bins));
    queue.memcpy(events, view.events, 3U * sizeof(*events));
    queue.memcpy(products, view.products, sizeof(*products));
    queue.wait_and_throw();
    const auto input = fixture();
    require(std::memcmp(bins, input.energy_bins.data(), 2U * sizeof(*bins)) == 0,
            "bin round-trip mismatch");
    require(std::memcmp(events, input.events.data(), 3U * sizeof(*events)) == 0,
            "event round-trip mismatch");
    require(std::memcmp(products, input.products.data(), sizeof(*products)) == 0,
            "product round-trip mismatch");
    sycl::free(bins, queue);
    sycl::free(events, queue);
    sycl::free(products, queue);

    auto moved = std::move(storage);
    require(storage.empty() && moved.device_view().event_count == 3U,
            "move did not transfer ELPKG ownership");
    moved.reset(queue);
    require(moved.empty() && moved.device_view().events == nullptr,
            "reset did not release ELPKG storage");
}

void test_empty_products() {
    auto data = fixture();
    data.products.clear();
    for (auto& event : data.events) {
        event.product_offset = 0;
        event.product_count = 0;
    }
    auto storage = carbon::ElasticPackageDeviceStorage(
        std::move(data), carbon::make_sycl_queue("cpu"));
    const auto view = storage.device_view();
    require(view.product_count == 0U && view.products == nullptr,
            "empty product array was not represented as null/zero");
}

void test_empty_data() {
    carbon::ElasticPackageFlatData data;
    auto storage = carbon::ElasticPackageDeviceStorage(
        std::move(data), carbon::make_sycl_queue("cpu"));
    require(storage.empty() && storage.device_view().energy_bins == nullptr,
            "empty ELPKG arrays were not accepted");
}
#endif

}  // namespace

int main() {
    try {
        test_flatten_and_overflow();
        test_flatten_round_trip();
#ifdef CARBON_HAS_SYCL
        test_storage_roundtrip();
        test_empty_products();
        test_empty_data();
#endif
        std::cout << "elastic_device_storage_tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

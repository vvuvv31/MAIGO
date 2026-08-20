#include "carbon/elastic_queue_adapter.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef CARBON_HAS_SYCL
#include "carbon/device.hpp"
#include <sycl/sycl.hpp>
#endif

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

carbon::ElasticQueueParentMetadata parent() {
    carbon::ElasticQueueParentMetadata value{};
    value.position_x_mm = 1.25F;
    value.position_y_mm = -2.5F;
    value.position_z_mm = 3.75F;
    value.time_ns = 12.5F;
    value.parent_pdg_id = 2212;
    value.parent_atomic_number = 1;
    value.parent_mass_number = 1;
    value.parent_generation = 2;
    value.parent_lineage = carbon::charged_lineage;
    value.parent_rng_stream = 0x1234567890abcdefULL;
    value.interaction_index = 19;
    return value;
}

carbon::ElasticApplicationResult application(const std::uint32_t charged,
                                             const std::uint32_t neutral,
                                             const float charged_energy,
                                             const float neutral_energy) {
    carbon::ElasticApplicationResult value{};
    value.status = carbon::ElasticApplicationStatus::success;
    value.incident_energy_MeV = 10.0F;
    value.outgoing_primary_energy_MeV = 7.0F;
    value.local_deposit_MeV = 0.5F;
    value.queued_charged_count = charged;
    value.queued_neutral_count = neutral;
    value.queued_charged_energy_MeV = charged_energy;
    value.queued_neutral_energy_MeV = neutral_energy;
    value.discarded_energy_MeV = 2.5F - charged_energy - neutral_energy;
    value.energy_closure_residual_MeV = 0.0F;
    return value;
}

carbon::ElasticQueuedProduct charged_product(const std::int32_t pdg = 2212,
                                             const float energy = 1.0F,
                                             const std::uint8_t generation = 4U) {
    return carbon::ElasticQueuedProduct{pdg, 1, 1, 1.0F, energy,
                                        {0.0F, 0.0F, 1.0F}, generation,
                                        carbon::ElasticProductRoute::charged};
}

carbon::ElasticQueuedProduct neutral_product(const std::int32_t pdg = 22,
                                             const float energy = 1.0F,
                                             const std::uint8_t generation = 5U) {
    return carbon::ElasticQueuedProduct{pdg, 0, 0, 0.0F, energy,
                                        {0.0F, 1.0F, 0.0F}, generation,
                                        carbon::ElasticProductRoute::neutral};
}

void test_charged_neutral_and_mapping() {
    std::vector<carbon::ElasticQueuedProduct> descriptors{
        charged_product(2212, 1.0F, 4U), neutral_product(22, 0.5F, 5U)};
    std::vector<carbon::SecondaryParticle3D> charged(1);
    std::vector<carbon::NeutralParticle3D> neutral(1);
    std::atomic<std::uint64_t> charged_counter{0};
    std::atomic<std::uint64_t> charged_filled{0};
    std::atomic<std::uint64_t> neutral_counter{0};
    std::atomic<std::uint64_t> neutral_filled{0};
    carbon::ElasticHostAtomicQueueStorage storage{
        charged.data(), 1, &charged_counter, &charged_filled,
        neutral.data(), 1, &neutral_counter, &neutral_filled};
    const auto app = application(1, 1, 1.0F, 0.5F);
    const auto result = carbon::commit_elastic_queues(
        app, {descriptors.data(), descriptors.size()}, parent(), storage);
    require(result.success() && result.written_charged_count == 1U &&
                result.written_neutral_count == 1U,
            "mixed charged/neutral commit failed");
    require(charged_counter.load() == 1U && charged_filled.load() == 1U &&
                neutral_counter.load() == 1U && neutral_filled.load() == 1U,
            "route counters were not committed exactly once");
    require(charged[0].pdg_id == 2212 && charged[0].atomic_number == 1 &&
                charged[0].mass_number == 1 && charged[0].generation == 4U &&
                charged[0].reserved == carbon::charged_lineage,
            "charged identity/generation/lineage mapping failed");
    require(neutral[0].pdg_id == 22 && neutral[0].generation == 5U &&
                neutral[0].reserved == carbon::gamma_lineage,
            "neutral identity/generation/lineage mapping failed");
    require(charged[0].position_x_mm == 1.25F && charged[0].position_y_mm == -2.5F &&
                charged[0].position_z_mm == 3.75F && charged[0].direction_z == 1.0F,
            "charged position/direction mapping failed");
    require(result.scorer.origin == carbon::ElasticQueueOrigin::elastic &&
                result.scorer.time_ns == 12.5F &&
                result.scorer.local_deposit_MeV == 0.5F,
            "elastic scorer contribution did not preserve parent metadata");
    require(std::abs(result.committed_ledger_closure_residual_MeV) < 1.0e-5F,
            "mixed event ledger did not close");
    require(charged[0].rng_stream != 0 && neutral[0].rng_stream != 0 &&
                charged[0].rng_stream != neutral[0].rng_stream,
            "elastic child RNG lineage was not assigned independently");
}

void test_route_overflow_is_whole_route_and_mixed_is_explicit() {
    std::vector<carbon::ElasticQueuedProduct> descriptors{
        charged_product(2212, 1.0F), charged_product(1000010020, 0.75F),
        neutral_product(22, 0.5F)};
    carbon::SecondaryParticle3D charged_sentinel{9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
    carbon::NeutralParticle3D neutral_sentinel{8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
    carbon::SecondaryParticle3D charged = charged_sentinel;
    carbon::NeutralParticle3D neutral = neutral_sentinel;
    std::atomic<std::uint64_t> charged_counter{0};
    std::atomic<std::uint64_t> charged_filled{0};
    std::atomic<std::uint64_t> neutral_counter{0};
    std::atomic<std::uint64_t> neutral_filled{0};
    carbon::ElasticHostAtomicQueueStorage storage{
        &charged, 1, &charged_counter, &charged_filled,
        &neutral, 1, &neutral_counter, &neutral_filled};
    const auto app = application(2, 1, 1.75F, 0.5F);
    const auto result = carbon::commit_elastic_queues(
        app, {descriptors.data(), descriptors.size()}, parent(), storage);
    require(result.status == carbon::ElasticQueueCommitStatus::partial_route_commit,
            "mixed route overflow was hidden as all-or-none");
    require(result.written_charged_count == 0U && result.dropped_charged_count == 2U &&
                result.written_neutral_count == 1U,
            "charged overflow did not drop the whole route");
    require(charged.pdg_id == charged_sentinel.pdg_id &&
                charged.position_x_mm == charged_sentinel.position_x_mm &&
                neutral.pdg_id == 22,
            "overflow route wrote a queue record or damaged the other route");
    require(result.dropped_charged_energy_MeV == 1.75F &&
                result.scorer.overflow_charged_energy_MeV == 1.75F &&
                std::abs(result.committed_ledger_closure_residual_MeV) < 1.0e-5F,
            "overflow energy was not retained in the explicit ledger");
}

void test_exact_capacity_and_empty_routes() {
    std::vector<carbon::ElasticQueuedProduct> charged_descriptors{
        charged_product(2212, 0.25F), charged_product(1000010030, 0.5F)};
    std::vector<carbon::SecondaryParticle3D> records(2);
    std::atomic<std::uint64_t> counter{0};
    std::atomic<std::uint64_t> filled{0};
    carbon::ElasticHostAtomicQueueStorage storage{
        records.data(), 2, &counter, &filled, nullptr, 0, nullptr, nullptr};
    const auto app = application(2, 0, 0.75F, 0.0F);
    const auto result = carbon::commit_elastic_queues(
        app, {charged_descriptors.data(), charged_descriptors.size()}, parent(), storage);
    require(result.success() && result.written_charged_count == 2U &&
                counter.load() == 2U && filled.load() == 2U,
            "exact charged capacity did not commit");

    const auto empty_app = application(0, 0, 0.0F, 0.0F);
    const auto empty = carbon::commit_elastic_queues(
        empty_app, {nullptr, 0}, parent(), storage);
    require(empty.success() && empty.written_charged_count == 0U &&
                empty.written_neutral_count == 0U,
            "empty elastic event was not accepted");
}

void test_concurrent_reservations_do_not_overlap() {
    constexpr std::size_t thread_count = 32;
    std::vector<carbon::SecondaryParticle3D> records(thread_count);
    std::atomic<std::uint64_t> counter{0};
    std::atomic<std::uint64_t> filled{0};
    carbon::ElasticHostAtomicQueueStorage storage{
        records.data(), thread_count, &counter, &filled, nullptr, 0, nullptr, nullptr};
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (std::size_t index = 0; index < thread_count; ++index) {
        threads.emplace_back([&, index] {
            auto descriptor = charged_product(
                static_cast<std::int32_t>(1000 + index), 0.25F,
                static_cast<std::uint8_t>(index));
            auto app = application(1, 0, 0.25F, 0.0F);
            const auto result = carbon::commit_elastic_queues(
                app, {&descriptor, 1}, parent(), storage);
            require(result.success() && result.written_charged_count == 1U,
                    "concurrent queue reservation failed");
        });
    }
    for (auto& thread : threads) thread.join();
    require(counter.load() == thread_count && filled.load() == thread_count,
            "concurrent reservation counters overlap or lost entries");
    std::vector<bool> seen(thread_count, false);
    for (const auto& record : records) {
        require(record.pdg_id >= 1000 && record.pdg_id < 1000 + thread_count,
                "concurrent reservation wrote an invalid record");
        const auto index = static_cast<std::size_t>(record.pdg_id - 1000);
        require(!seen[index], "concurrent reservations overlapped a queue slot");
        seen[index] = true;
    }
    for (const auto value : seen) require(value, "a concurrent queue slot was not written");
}

#ifdef CARBON_HAS_SYCL
void test_sycl_device_smoke() {
    auto queue = carbon::make_sycl_queue("cpu");
    auto* descriptors = sycl::malloc_shared<carbon::ElasticQueuedProduct>(1, queue);
    auto* app_device = sycl::malloc_shared<carbon::ElasticApplicationResult>(1, queue);
    auto* parent_value = sycl::malloc_shared<carbon::ElasticQueueParentMetadata>(1, queue);
    auto* charged = sycl::malloc_shared<carbon::SecondaryParticle3D>(1, queue);
    auto* charged_counter = sycl::malloc_shared<std::uint64_t>(1, queue);
    auto* charged_filled = sycl::malloc_shared<std::uint64_t>(1, queue);
    auto* result = sycl::malloc_shared<carbon::ElasticQueueCommitResult>(1, queue);
    require(descriptors && app_device && parent_value && charged && charged_counter &&
                charged_filled && result,
            "elastic queue adapter SYCL allocation failed");
    *descriptors = charged_product(2212, 1.0F, 3U);
    *app_device = application(1, 0, 1.0F, 0.0F);
    *parent_value = parent();
    *charged_counter = 0;
    *charged_filled = 0;
    const carbon::ElasticDeviceQueueStorage storage{
        charged, 1, charged_counter, charged_filled, nullptr, 0, nullptr, nullptr};
    queue.single_task([=] {
        *result = carbon::commit_elastic_queues(
            *app_device, {descriptors, 1}, *parent_value, storage);
    }).wait_and_throw();
    require(result->success() && result->written_charged_count == 1U &&
                *charged_counter == 1U && *charged_filled == 1U && charged->pdg_id == 2212,
            "elastic queue adapter SYCL smoke failed");
    sycl::free(descriptors, queue);
    sycl::free(app_device, queue);
    sycl::free(parent_value, queue);
    sycl::free(charged, queue);
    sycl::free(charged_counter, queue);
    sycl::free(charged_filled, queue);
    sycl::free(result, queue);
}
#endif

}  // namespace

int main() {
    try {
        test_charged_neutral_and_mapping();
        test_route_overflow_is_whole_route_and_mixed_is_explicit();
        test_exact_capacity_and_empty_routes();
        test_concurrent_reservations_do_not_overlap();
#ifdef CARBON_HAS_SYCL
        test_sycl_device_smoke();
#endif
        std::cout << "elastic_queue_adapter_tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

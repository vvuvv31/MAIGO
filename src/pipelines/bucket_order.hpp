#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace carbon::pipelines {

// Device buffers for counting-sort of live continuation slots. Sorting changes
// lane order only; particle state, history identity and RNG stay at each slot.
struct BucketOrderBuffers {
    const std::uint32_t* source;
    std::uint32_t* destination;
    std::uint32_t* histogram;
    std::uint32_t* offsets;
    unsigned bucket_count;
};

// The caller supplies an in-order queue and live buffers. key(slot) must be a
// deterministic value in [0, bucket_count). Events remain with the caller for
// timing and lifetime tracking; this helper adds no waits or allocations.
template <class Key>
void enqueue_bucket_order(sycl::queue& queue, BucketOrderBuffers buffers, std::uint32_t active,
                          Key key, std::vector<sycl::event>& events) {
    const auto [source, destination, histogram, offsets, bucket_count] = buffers;
    events.push_back(queue.fill(histogram, 0U, bucket_count));
    events.push_back(queue.parallel_for(sycl::range<1>(active), [=](sycl::id<1> id) {
        const auto bucket = key(source[id[0]]);
        sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(histogram[bucket])
            .fetch_add(1U);
    }));
    events.push_back(queue.single_task([=] {
        unsigned offset = 0;
        for (unsigned bucket = 0; bucket < bucket_count; ++bucket) {
            offsets[bucket] = offset;
            offset += histogram[bucket];
            histogram[bucket] = 0;
        }
    }));
    events.push_back(queue.parallel_for(sycl::range<1>(active), [=](sycl::id<1> id) {
        const auto slot = source[id[0]];
        const auto bucket = key(slot);
        const auto rank =
            sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(histogram[bucket])
                .fetch_add(1U);
        destination[offsets[bucket] + rank] = slot;
    }));
}

} // namespace carbon::pipelines

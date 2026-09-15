#pragma once

#ifdef CARBON_HAS_SYCL

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <sycl/sycl.hpp>

namespace carbon::detail {

struct DeviceMemoryTracker {
    sycl::queue& queue;
    std::vector<void*> allocations{};
    std::unordered_map<void*, std::size_t> sizes{};
    std::size_t live_bytes{0};
    std::size_t peak_bytes{0};
    // Explicit USM allocation budget. Default 0 = unlimited, preserving the
    // historical behavior. The budget covers allocations made through this
    // tracker only, not the driver/runtime's own device memory.
    std::size_t limit_bytes{0};
    std::function<void(void*)> on_free_hook{};

    explicit DeviceMemoryTracker(sycl::queue& q, std::size_t budget_bytes = 0)
        : queue(q), limit_bytes(budget_bytes) {}

    DeviceMemoryTracker(const DeviceMemoryTracker&) = delete;
    DeviceMemoryTracker& operator=(const DeviceMemoryTracker&) = delete;
    DeviceMemoryTracker(DeviceMemoryTracker&&) = delete;
    DeviceMemoryTracker& operator=(DeviceMemoryTracker&&) = delete;

    bool unlimited() const noexcept { return limit_bytes == 0; }

    template <typename T>
    T* allocate(std::size_t count) {
        if (count == 0) {
            return nullptr;
        }
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        const auto bytes = count * sizeof(T);
        if (!unlimited() && bytes > limit_bytes - std::min(live_bytes, limit_bytes)) {
            throw std::runtime_error(
                "Device memory allocation budget exceeded "
                "(device_memory_budget_gib; reduce shard size or merge count); "
                "no valid dose produced");
        }
        auto* ptr = sycl::malloc_device<T>(count, queue);
        if (ptr == nullptr) {
            return nullptr;
        }
        // Register in two steps so a bookkeeping failure never leaves a freed
        // pointer in `allocations` (which would be double-freed) and counters
        // only advance after both registrations succeeded.
        try {
            allocations.push_back(ptr);
        } catch (...) {
            try {
                sycl::free(ptr, queue);
            } catch (...) {}
            throw;
        }
        try {
            sizes[ptr] = bytes;
        } catch (...) {
            const auto it = std::find(allocations.begin(), allocations.end(),
                                      static_cast<void*>(ptr));
            if (it != allocations.end()) allocations.erase(it);
            try {
                sycl::free(ptr, queue);
            } catch (...) {}
            throw;
        }
        live_bytes += bytes;
        peak_bytes = std::max(peak_bytes, live_bytes);
        return ptr;
    }

    template <typename T>
    T* track(T* ptr) {
        if (ptr != nullptr) {
            auto it = std::find(allocations.begin(), allocations.end(), static_cast<void*>(ptr));
            if (it == allocations.end()) {
                allocations.push_back(static_cast<void*>(ptr));
            }
        }
        return ptr;
    }

    void free(void* ptr) {
        if (ptr != nullptr) {
            auto it = std::find(allocations.begin(), allocations.end(), ptr);
            if (it != allocations.end()) {
                sycl::free(ptr, queue);
                allocations.erase(it);
                auto size_it = sizes.find(ptr);
                if (size_it != sizes.end()) {
                    live_bytes -= std::min(live_bytes, size_it->second);
                    sizes.erase(size_it);
                }
                if (on_free_hook) {
                    on_free_hook(ptr);
                }
            }
        }
    }

    std::size_t active_allocation_count() const noexcept {
        return allocations.size();
    }

    const std::vector<void*>& active_allocations() const noexcept {
        return allocations;
    }

    ~DeviceMemoryTracker() {
        if (!unlimited() || std::getenv("CARBON_RUNTIME_BREAKDOWN") != nullptr) {
            std::cerr << "[device-memory] tracked_peak_GiB="
                      << static_cast<double>(peak_bytes) / (1024.0 * 1024.0 * 1024.0)
                      << " limit_GiB="
                      << (unlimited()
                              ? std::string("unlimited")
                              : std::to_string(static_cast<double>(limit_bytes) /
                                               (1024.0 * 1024.0 * 1024.0)))
                      << " (excludes driver/runtime allocations)\n";
        }
        for (auto* ptr : allocations) {
            if (ptr != nullptr) {
                try {
                    sycl::free(ptr, queue);
                    if (on_free_hook) {
                        on_free_hook(ptr);
                    }
                } catch (...) {}
            }
        }
        allocations.clear();
    }
};

}  // namespace carbon::detail

#endif  // CARBON_HAS_SYCL

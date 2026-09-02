#pragma once

#ifdef CARBON_HAS_SYCL

#include <algorithm>
#include <cstddef>
#include <functional>
#include <vector>
#include <sycl/sycl.hpp>

namespace carbon::detail {

struct DeviceMemoryTracker {
    sycl::queue& queue;
    std::vector<void*> allocations{};
    std::function<void(void*)> on_free_hook{};

    explicit DeviceMemoryTracker(sycl::queue& q) : queue(q) {}

    DeviceMemoryTracker(const DeviceMemoryTracker&) = delete;
    DeviceMemoryTracker& operator=(const DeviceMemoryTracker&) = delete;
    DeviceMemoryTracker(DeviceMemoryTracker&&) = delete;
    DeviceMemoryTracker& operator=(DeviceMemoryTracker&&) = delete;

    template <typename T>
    T* allocate(std::size_t count) {
        if (count == 0) {
            return nullptr;
        }
        auto* ptr = sycl::malloc_device<T>(count, queue);
        if (ptr == nullptr) {
            return nullptr;
        }
        try {
            allocations.push_back(ptr);
        } catch (...) {
            try {
                sycl::free(ptr, queue);
            } catch (...) {}
            throw;
        }
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

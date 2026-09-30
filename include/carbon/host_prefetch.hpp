#pragma once

#include "carbon/delta_moments_data.hpp"
#include "carbon/inelastic_package_v3.hpp"
#include "carbon/unified_em_package.hpp"

#include <filesystem>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace carbon {

// Background parsing of large read-only input packages during transport
// setup. A prefetch runs exactly the loader the consumer would call; take()
// returns that result (rethrowing its exception) or, when nothing was
// prefetched for the key, runs the loader synchronously.
template <class T>
class HostPrefetch {
public:
    HostPrefetch() = default;
    HostPrefetch(const HostPrefetch&) = delete;
    HostPrefetch& operator=(const HostPrefetch&) = delete;

    void start(const std::string& key, std::function<T()> load) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.count(key) == 0)
            entries_.emplace(key, std::async(std::launch::async, std::move(load)));
    }

    T take(const std::string& key, const std::function<T()>& load) {
        std::future<T> pending;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto found = entries_.find(key);
            if (found != entries_.end()) {
                pending = std::move(found->second);
                entries_.erase(found);
            }
        }
        return pending.valid() ? pending.get() : load();
    }

private:
    std::mutex mutex_;
    std::map<std::string, std::future<T>> entries_;
};

struct LoadedCinel03 {
    std::shared_ptr<InelasticPackageV3Table> package;
    std::shared_ptr<Cinel03DeviceTables> tables;
};

inline LoadedCinel03 load_cinel03(const std::filesystem::path& path) {
    auto package = std::make_shared<InelasticPackageV3Table>(
        InelasticPackageV3Table::from_binary(path));
    auto tables = std::make_shared<Cinel03DeviceTables>(package->make_device_tables());
    return {std::move(package), std::move(tables)};
}

inline HostPrefetch<LoadedCinel03>& cinel03_prefetch() {
    static HostPrefetch<LoadedCinel03> registry;
    return registry;
}

inline void prefetch_cinel03(const std::filesystem::path& path) {
    std::error_code error;
    if (path.empty() || !std::filesystem::is_regular_file(path, error)) return;
    cinel03_prefetch().start(path.string(), [path] { return load_cinel03(path); });
}

inline LoadedCinel03 take_cinel03(const std::filesystem::path& path) {
    return cinel03_prefetch().take(path.string(), [&] { return load_cinel03(path); });
}

struct LoadedUnifiedEm {
    std::shared_ptr<const UnifiedEmPackage> package;
    std::shared_ptr<const std::vector<float>> delta_means;
};

struct UnifiedEmRequest {
    std::filesystem::path package_file;
    std::string package_sha256;
    std::filesystem::path delta_file;
    std::string delta_sha256;
    std::string delta_source_sha256;

    [[nodiscard]] std::string key() const {
        return package_file.string() + "|" + package_sha256 + "|" + delta_file.string() + "|" +
               delta_sha256 + "|" + delta_source_sha256;
    }
    [[nodiscard]] LoadedUnifiedEm load() const {
        auto package = std::make_shared<const UnifiedEmPackage>(
            UnifiedEmPackage::load(package_file, package_sha256));
        auto delta = std::make_shared<const std::vector<float>>(load_delta_moments(
            delta_file, package_sha256, package->nodes.size(), delta_sha256, delta_source_sha256));
        return {std::move(package), std::move(delta)};
    }
};

inline HostPrefetch<LoadedUnifiedEm>& unified_em_prefetch() {
    static HostPrefetch<LoadedUnifiedEm> registry;
    return registry;
}

inline void prefetch_unified_em(const UnifiedEmRequest& request) {
    unified_em_prefetch().start(request.key(), [request] { return request.load(); });
}

inline LoadedUnifiedEm take_unified_em(const UnifiedEmRequest& request) {
    return unified_em_prefetch().take(request.key(), [&] { return request.load(); });
}

}  // namespace carbon

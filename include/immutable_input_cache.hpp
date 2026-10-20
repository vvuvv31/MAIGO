#pragma once
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <utility>

namespace carbon {
// Read-only parsed inputs, owned by one run session. Particle state never enters
// this cache. A changed path, file size or timestamp requires a fresh load.
inline std::string input_cache_key(const std::filesystem::path& path) {
    auto key = std::filesystem::absolute(path).lexically_normal().string();
    const auto append = [&](const std::filesystem::path& p) {
        key += "|" + std::to_string(std::filesystem::file_size(p)) + ":" +
            std::to_string(std::filesystem::last_write_time(p).time_since_epoch().count());
    };
    append(path);
    return key;
}
class ImmutableInputCache {
    using Key = std::pair<std::type_index, std::string>;
    std::map<Key, std::shared_ptr<const void>> entries_;
public:
    std::size_t hits{}, misses{};
    template<class T> bool contains(const std::string& key) const {
        return entries_.contains(Key{typeid(T), key});
    }
    template<class T, class Loader> const T& get(const std::string& key, Loader&& load) {
        const Key typed{typeid(T), key};
        if (auto it = entries_.find(typed); it != entries_.end()) {
            ++hits;
            return *std::static_pointer_cast<const T>(it->second);
        }
        auto value = std::make_shared<const T>(std::forward<Loader>(load)());
        entries_.emplace(typed, value);
        ++misses;
        return *value;
    }
};
}

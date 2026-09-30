#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <future>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#if defined(__unix__)
#include <sys/stat.h>
#endif
#ifdef CARBON_OPENSSL_SHA256
#include <openssl/crypto.h>
#include <openssl/evp.h>
#endif

namespace carbon {

namespace detail {

inline constexpr std::array<std::uint32_t, 64> k_sha256_k = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
};

inline constexpr std::uint32_t sha256_rotr(std::uint32_t x, std::uint32_t n) noexcept {
    return (x >> n) | (x << (32U - n));
}

inline constexpr std::uint32_t sha256_choose(std::uint32_t e, std::uint32_t f, std::uint32_t g) noexcept {
    return (e & f) ^ (~e & g);
}

inline constexpr std::uint32_t sha256_majority(std::uint32_t a, std::uint32_t b, std::uint32_t c) noexcept {
    return (a & b) ^ (a & c) ^ (b & c);
}

inline constexpr std::uint32_t sha256_sigma0(std::uint32_t x) noexcept {
    return sha256_rotr(x, 2U) ^ sha256_rotr(x, 13U) ^ sha256_rotr(x, 22U);
}

inline constexpr std::uint32_t sha256_sigma1(std::uint32_t x) noexcept {
    return sha256_rotr(x, 6U) ^ sha256_rotr(x, 11U) ^ sha256_rotr(x, 25U);
}

inline constexpr std::uint32_t sha256_gamma0(std::uint32_t x) noexcept {
    return sha256_rotr(x, 7U) ^ sha256_rotr(x, 18U) ^ (x >> 3U);
}

inline constexpr std::uint32_t sha256_gamma1(std::uint32_t x) noexcept {
    return sha256_rotr(x, 17U) ^ sha256_rotr(x, 19U) ^ (x >> 10U);
}

inline void sha256_transform(std::uint32_t state[8], const std::uint8_t block[64]) noexcept {
    std::uint32_t w[64];
    for (std::size_t i = 0; i < 16; ++i) {
        w[i] = (static_cast<std::uint32_t>(block[i * 4 + 0]) << 24U) |
               (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16U) |
               (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8U) |
               (static_cast<std::uint32_t>(block[i * 4 + 3]));
    }
    for (std::size_t i = 16; i < 64; ++i) {
        w[i] = sha256_gamma1(w[i - 2]) + w[i - 7] + sha256_gamma0(w[i - 15]) + w[i - 16];
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];

    for (std::size_t i = 0; i < 64; ++i) {
        const std::uint32_t t1 = h + sha256_sigma1(e) + sha256_choose(e, f, g) + k_sha256_k[i] + w[i];
        const std::uint32_t t2 = sha256_sigma0(a) + sha256_majority(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

}  // namespace detail

class Sha256 {
public:
    Sha256() noexcept { reset(); }

    void reset() noexcept {
        state_[0] = 0x6a09e667U;
        state_[1] = 0xbb67ae85U;
        state_[2] = 0x3c6ef372U;
        state_[3] = 0xa54ff53aU;
        state_[4] = 0x510e527fU;
        state_[5] = 0x9b05688cU;
        state_[6] = 0x1f83d9abU;
        state_[7] = 0x5be0cd19U;
        count_ = 0;
        buffer_len_ = 0;
    }

    void update(const void* data, std::size_t len) noexcept {
        const auto* ptr = static_cast<const std::uint8_t*>(data);
        count_ += static_cast<std::uint64_t>(len) * 8U;

        if (buffer_len_ > 0) {
            const std::size_t to_copy = std::min(len, 64U - buffer_len_);
            for (std::size_t i = 0; i < to_copy; ++i) {
                buffer_[buffer_len_ + i] = ptr[i];
            }
            buffer_len_ += to_copy;
            ptr += to_copy;
            len -= to_copy;

            if (buffer_len_ == 64U) {
                detail::sha256_transform(state_.data(), buffer_.data());
                buffer_len_ = 0;
            }
        }

        while (len >= 64U) {
            detail::sha256_transform(state_.data(), ptr);
            ptr += 64U;
            len -= 64U;
        }

        if (len > 0) {
            for (std::size_t i = 0; i < len; ++i) {
                buffer_[i] = ptr[i];
            }
            buffer_len_ = len;
        }
    }

    [[nodiscard]] std::array<std::uint8_t, 32> finalize() noexcept {
        buffer_[buffer_len_++] = 0x80U;
        if (buffer_len_ > 56U) {
            while (buffer_len_ < 64U) {
                buffer_[buffer_len_++] = 0x00U;
            }
            detail::sha256_transform(state_.data(), buffer_.data());
            buffer_len_ = 0;
        }
        while (buffer_len_ < 56U) {
            buffer_[buffer_len_++] = 0x00U;
        }
        for (int i = 7; i >= 0; --i) {
            buffer_[56 + (7 - i)] = static_cast<std::uint8_t>((count_ >> (i * 8)) & 0xffU);
        }
        detail::sha256_transform(state_.data(), buffer_.data());

        std::array<std::uint8_t, 32> hash{};
        for (std::size_t i = 0; i < 8; ++i) {
            hash[i * 4 + 0] = static_cast<std::uint8_t>((state_[i] >> 24U) & 0xffU);
            hash[i * 4 + 1] = static_cast<std::uint8_t>((state_[i] >> 16U) & 0xffU);
            hash[i * 4 + 2] = static_cast<std::uint8_t>((state_[i] >> 8U) & 0xffU);
            hash[i * 4 + 3] = static_cast<std::uint8_t>(state_[i] & 0xffU);
        }
        return hash;
    }

    [[nodiscard]] std::string finalize_hex() noexcept {
        const auto hash = finalize();
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (const auto byte : hash) {
            oss << std::setw(2) << static_cast<unsigned>(byte);
        }
        return oss.str();
    }

private:
    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buffer_{};
    std::uint64_t count_{0};
    std::size_t buffer_len_{0};
};

inline std::string compute_sha256_hex(const void* data, std::size_t size) {
    Sha256 ctx;
    ctx.update(data, size);
    return ctx.finalize_hex();
}

inline std::string compute_sha256_hex(std::string_view text) {
    return compute_sha256_hex(text.data(), text.size());
}

// Complete OpenSSL's one-time initialisation, including the atexit cleanup it
// registers, on the calling thread. OPENSSL_init_crypto(0, ...) is a no-op, so
// without this the first digest on a helper thread would register the cleanup
// after the objects that join those threads, and an early process exit would
// tear OpenSSL down under a running hash.
inline void initialize_sha256_backend() {
#ifdef CARBON_OPENSSL_SHA256
    static const bool initialized = [] {
        OPENSSL_init_crypto(OPENSSL_INIT_LOAD_CONFIG, nullptr);
        std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
        return ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1;
    }();
    (void)initialized;
#endif
}

namespace detail {
inline std::string compute_file_sha256_hex_uncached(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("compute_file_sha256_hex: cannot open file " + path.string());
    }
#ifdef CARBON_OPENSSL_SHA256
    std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),EVP_MD_CTX_free);
    if(!ctx || EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)!=1)
        throw std::runtime_error("Cannot initialize SHA256");
    std::vector<char> buffer(1024*1024);
    while(stream.read(buffer.data(),buffer.size()) || stream.gcount()>0) {
        if(EVP_DigestUpdate(ctx.get(),buffer.data(),static_cast<std::size_t>(stream.gcount()))!=1)
            throw std::runtime_error("SHA256 update failed");
    }
    if(stream.bad())throw std::runtime_error("SHA256 file read failed: "+path.string());
    std::array<unsigned char,EVP_MAX_MD_SIZE> digest{};unsigned size=0;
    if(EVP_DigestFinal_ex(ctx.get(),digest.data(),&size)!=1 || size!=32)
        throw std::runtime_error("SHA256 finalization failed");
    std::ostringstream out;out<<std::hex<<std::setfill('0');
    for(unsigned i=0;i<size;++i)out<<std::setw(2)<<unsigned(digest[i]);
    return out.str();
#else
    Sha256 ctx;
    std::array<char, 4096> buffer{};
    while (stream.read(buffer.data(), buffer.size()) || stream.gcount() > 0) {
        ctx.update(buffer.data(), static_cast<std::size_t>(stream.gcount()));
    }
    if(stream.bad())throw std::runtime_error("SHA256 file read failed: "+path.string());
    return ctx.finalize_hex();
#endif
}

// Process-local digest memo. Large packages are pinned and re-verified by
// several independent loaders/reports; each re-read costs ~0.5 s per GB.
// A digest is reused only while the file keeps the device/inode/size/mtime
// observed before it was hashed in this process, and waiters re-stat after
// the digest is ready. CARBON_SHA256_MEMO=0 restores one full read per call.
class FileSha256Memo {
public:
    FileSha256Memo() {
        const char* value = std::getenv("CARBON_SHA256_MEMO");
        enabled_ = !(value && std::string_view(value) == "0");
        // OpenSSL's cleanup must be registered before this object's
        // destructor, which joins the prefetch workers.
        initialize_sha256_backend();
    }
    ~FileSha256Memo() { join(); }
    FileSha256Memo(const FileSha256Memo&) = delete;
    FileSha256Memo& operator=(const FileSha256Memo&) = delete;

    std::string digest(const std::filesystem::path& path) {
        Identity id{};
        if (!enabled_ || !identity(path, id)) return compute_file_sha256_hex_uncached(path);
        const auto key = std::make_pair(id.device, id.inode);
        std::promise<std::string> promise;
        std::shared_future<std::string> ready;
        bool owner = false;
        std::uint64_t token = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto found = entries_.find(key);
            if (found != entries_.end() && found->second.id == id) {
                ready = found->second.digest;
            } else {
                ready = promise.get_future().share();
                token = ++next_token_;
                entries_[key] = Entry{id, ready, token};
                owner = true;
            }
        }
        if (!owner) {
            try {
                auto value = ready.get();
                Identity now{};
                if (identity(path, now) && now == id) return value;
            } catch (...) {
            }
            return compute_file_sha256_hex_uncached(path);
        }
        try {
            auto value = compute_file_sha256_hex_uncached(path);
            Identity after{};
            if (!identity(path, after) || !(after == id)) forget(key, token);
            promise.set_value(value);
            return value;
        } catch (...) {
            forget(key, token);
            promise.set_exception(std::current_exception());
            throw;
        }
    }

    // Hash files on background threads; failures are left to the loader that
    // later requests the digest, which reproduces its original error.
    void prefetch(std::vector<std::filesystem::path> paths, unsigned threads) {
        if (!enabled_ || threads == 0 || paths.empty()) return;
        auto work = std::make_shared<std::vector<std::filesystem::path>>(std::move(paths));
        auto next = std::make_shared<std::atomic<std::size_t>>(0);
        threads = static_cast<unsigned>(std::min<std::size_t>(threads, work->size()));
        std::lock_guard<std::mutex> lock(mutex_);
        for (unsigned i = 0; i < threads; ++i) {
            workers_.emplace_back([this, work, next] {
                for (std::size_t j; !stopping_.load() && (j = next->fetch_add(1)) < work->size();) {
                    try {
                        (void)digest((*work)[j]);
                    } catch (...) {
                    }
                }
            });
        }
    }

    // Workers finish the file in hand and take no further files.
    void join() {
        stopping_.store(true);
        std::vector<std::thread> workers;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            workers.swap(workers_);
        }
        for (auto& worker : workers) worker.join();
    }

private:
    struct Identity {
        std::uint64_t device{}, inode{}, size{};
        std::int64_t mtime_s{}, mtime_ns{};
        bool operator==(const Identity& other) const noexcept {
            return device == other.device && inode == other.inode && size == other.size &&
                   mtime_s == other.mtime_s && mtime_ns == other.mtime_ns;
        }
    };
    struct Entry {
        Identity id;
        std::shared_future<std::string> digest;
        std::uint64_t token{};
    };
    static bool identity(const std::filesystem::path& path, Identity& out) noexcept {
#if defined(__unix__)
        struct stat status {};
        if (::stat(path.c_str(), &status) != 0 || !S_ISREG(status.st_mode)) return false;
        out = Identity{static_cast<std::uint64_t>(status.st_dev),
                       static_cast<std::uint64_t>(status.st_ino),
                       static_cast<std::uint64_t>(status.st_size),
                       static_cast<std::int64_t>(status.st_mtim.tv_sec),
                       static_cast<std::int64_t>(status.st_mtim.tv_nsec)};
        return true;
#else
        (void)path;
        (void)out;
        return false;
#endif
    }
    void forget(const std::pair<std::uint64_t, std::uint64_t>& key, std::uint64_t token) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = entries_.find(key);
        if (found != entries_.end() && found->second.token == token) entries_.erase(found);
    }

    bool enabled_{true};
    std::atomic<bool> stopping_{false};
    std::uint64_t next_token_{0};
    std::mutex mutex_;
    std::map<std::pair<std::uint64_t, std::uint64_t>, Entry> entries_;
    std::vector<std::thread> workers_;
};

inline FileSha256Memo& file_sha256_memo() {
    static FileSha256Memo memo;
    return memo;
}
}  // namespace detail

inline std::string compute_file_sha256_hex(const std::filesystem::path& path) {
    return detail::file_sha256_memo().digest(path);
}

// Start hashing input files in the background; later checks reuse the digest.
inline void prefetch_file_sha256(std::vector<std::filesystem::path> paths, unsigned threads) {
    detail::file_sha256_memo().prefetch(std::move(paths), threads);
}

// Wait for background hashing (e.g. before returning from main on an error).
inline void join_file_sha256_prefetch() {
    detail::file_sha256_memo().join();
}

inline constexpr bool file_integrity_checks_enabled =
#if CARBON_DISABLE_INTEGRITY_CHECKS
    false;
#else
    true;
#endif

inline bool file_sha256_matches(const std::filesystem::path& path,
                                const std::string_view expected) {
    if constexpr (!file_integrity_checks_enabled) {
        (void)path;
        (void)expected;
        return true;
    }
    return compute_file_sha256_hex(path) == expected;
}

inline std::string file_sha256_for_report(const std::filesystem::path& path) {
    if constexpr (!file_integrity_checks_enabled) {
        (void)path;
        return "disabled";
    }
    return compute_file_sha256_hex(path);
}

}  // namespace carbon

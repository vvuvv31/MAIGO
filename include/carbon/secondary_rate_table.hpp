#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace carbon {

constexpr std::size_t kSecondaryNumProjectiles = 13;
constexpr std::size_t kSecondaryNumSections = 25;
constexpr std::size_t kSecondaryNumTargets = 13;
// v1 frozen grid: 860 nodes over [0.5, 430.0], step 0.5.
constexpr std::size_t kSecondaryNumEnergies = 860;
constexpr double kSecondaryV1EnergyMin = 0.5;
constexpr double kSecondaryV1EnergyMax = 430.0;
// v2 grid: 921 nodes over [0.1, 460.1], step 0.5. The v2.1 data product
// covers the full chain scope so device lookup needs no clamp: outside
// [emin, emax] the rate is exactly 0 (no hazard -> no query -> no miss).
constexpr std::size_t kSecondaryV2NumEnergies = 921;
constexpr double kSecondaryV2EnergyMin = 0.1;
constexpr double kSecondaryV2EnergyMax = 460.1;
constexpr double kSecondaryEnergyStep = 0.5;

#pragma pack(push, 1)
struct SecondaryRateHeader {
    char magic[8];             // "SCHN2RAT"
    uint32_t version;          // 1
    uint32_t num_projectiles;  // 13
    uint32_t num_sections;     // 25
    uint32_t num_targets;      // 13
    uint32_t num_energies;     // 860
    double energy_min_mevu;    // 0.5
    double energy_max_mevu;    // 430.0
    double energy_step_mevu;   // 0.5
};

struct SecondaryProjectileKey {
    int32_t z;
    int32_t a;
};

// v3 per-(projectile, target) valid-domain entry appended after the totals
// payload. has_support==1 iff the event package carries that channel;
// [emin, emax] are the package channel bounds. Queries outside the channel
// domain MUST yield exactly 0 (mask before interpolation, never after).
struct SecondaryRateDomainEntry {
    double energy_min_mevu;
    double energy_max_mevu;
    std::uint8_t has_support;
    std::uint8_t reserved[7];
};
#pragma pack(pop)
static_assert(sizeof(SecondaryRateDomainEntry) == 24,
              "SecondaryRateDomainEntry must be 24 bytes");

// Reads only magic + version from a SCHN2RAT binary (no full load).
// Throws on missing file, bad magic, or short header.
std::uint32_t secondary_rate_binary_version(const std::filesystem::path& binary_path);

class SecondaryRateTable {
public:
    SecondaryRateTable() = default;

    static SecondaryRateTable from_binary(
        const std::filesystem::path& binary_path,
        const std::filesystem::path& metadata_path = {});

    [[nodiscard]] double energy_min_mevu() const noexcept { return energy_min_mevu_; }
    [[nodiscard]] double energy_max_mevu() const noexcept { return energy_max_mevu_; }
    [[nodiscard]] double energy_step_mevu() const noexcept { return energy_step_mevu_; }
    [[nodiscard]] std::size_t num_projectiles() const noexcept { return num_projectiles_; }
    [[nodiscard]] std::size_t num_energies() const noexcept { return num_energies_; }
    [[nodiscard]] std::uint32_t binary_version() const noexcept { return binary_version_; }

    [[nodiscard]] int projectile_index(int proj_z, int proj_a) const noexcept;
    [[nodiscard]] static std::size_t target_index_from_z(int target_z);

    [[nodiscard]] double mass_partial_rate(std::size_t proj_idx, std::size_t section_id, std::size_t target_idx, std::size_t energy_idx) const;
    [[nodiscard]] double mass_total_rate(std::size_t proj_idx, std::size_t section_id, std::size_t energy_idx) const;

    [[nodiscard]] double interpolate_mass_total(std::size_t proj_idx, std::size_t section_id, double energy_mevu) const;

    [[nodiscard]] const std::vector<SecondaryProjectileKey>& projectiles() const noexcept { return projectiles_; }
    [[nodiscard]] const std::vector<double>& mass_partial_rates() const noexcept { return mass_partial_rates_; }
    [[nodiscard]] const std::vector<double>& mass_total_rates() const noexcept { return mass_total_rates_; }
    // v3 only: per-(projectile, target) valid domain, np*nt entries in
    // [proj][target] order. Empty for v1 binaries.
    [[nodiscard]] bool has_channel_domains() const noexcept { return !channel_domains_.empty(); }
    [[nodiscard]] const std::vector<SecondaryRateDomainEntry>& channel_domains() const noexcept {
        return channel_domains_;
    }
    [[nodiscard]] const SecondaryRateDomainEntry& channel_domain(std::size_t proj_idx,
                                                                std::size_t target_idx) const {
        if (proj_idx >= num_projectiles_ || target_idx >= kSecondaryNumTargets ||
            channel_domains_.size() != num_projectiles_ * kSecondaryNumTargets) {
            throw std::out_of_range("SecondaryRateTable::channel_domain: index out of range");
        }
        return channel_domains_[proj_idx * kSecondaryNumTargets + target_idx];
    }
    // v3 masked partial at an energy node (host mirror of the device mask):
    // exactly 0 outside the channel domain, raw node value inside.
    [[nodiscard]] double masked_partial_at_node(std::size_t proj_idx, std::size_t section_id,
                                               std::size_t target_idx, std::size_t energy_idx) const {
        const SecondaryRateDomainEntry& dom = channel_domain(proj_idx, target_idx);
        if (!dom.has_support) {
            return 0.0;
        }
        const double e = energy_min_mevu_ + static_cast<double>(energy_idx) * energy_step_mevu_;
        if (e < dom.energy_min_mevu || e > dom.energy_max_mevu) {
            return 0.0;
        }
        return mass_partial_rate(proj_idx, section_id, target_idx, energy_idx);
    }

private:
    uint32_t num_projectiles_{13};
    uint32_t binary_version_{1};
    std::size_t num_energies_{kSecondaryNumEnergies};
    double energy_min_mevu_{0.5};
    double energy_max_mevu_{430.0};
    double energy_step_mevu_{0.5};
    std::vector<SecondaryProjectileKey> projectiles_;
    std::array<int32_t, 13> canonical_targets_{};
    std::vector<double> mass_partial_rates_;
    std::vector<double> mass_total_rates_;
    std::vector<SecondaryRateDomainEntry> channel_domains_;
};

inline int secondary_projectile_index_device(int pz, int pa) noexcept {
    if (pz == 5 && pa == 11) return 0;
    if (pz == 5 && pa == 10) return 1;
    if (pz == 4 && pa == 9) return 2;
    if (pz == 4 && pa == 7) return 3;
    if (pz == 4 && pa == 10) return 4;
    if (pz == 3 && pa == 7) return 5;
    if (pz == 3 && pa == 6) return 6;
    if (pz == 2 && pa == 4) return 7;
    if (pz == 2 && pa == 3) return 8;
    if (pz == 1 && pa == 1) return 9;
    if (pz == 1 && pa == 2) return 10;
    if (pz == 1 && pa == 3) return 11;
    if (pz == 6 && pa == 11) return 12;
    return -1;
}

// Grid-parameterized device lookups shared by the v1 ([0.5,430]/860) and
// v2 ([0.1,460.1]/921) tables. Out-of-grid policy is data-driven:
//   - v1 grid (emin ~= 0.5, 860 nodes): legacy endpoint clamp. The v1 grid
//     does not cover chain scope, and frozen v1 results depend on the clamp,
//     so it is preserved EXACTLY for v1 files (bitwise reproducibility).
//   - any other grid: strict zero outside [emin, emax] (no clamp, no
//     extrapolation). Safe only because v2 grids provably cover chain scope;
//     a hazard is never sampled where the table declares no support.
inline bool secondary_grid_is_legacy_v1(float energy_min_MeV_per_u,
                                        std::uint32_t num_energies) noexcept {
    const float d = energy_min_MeV_per_u - 0.5F;
    return num_energies == 860 && (d < 1.0e-6F && d > -1.0e-6F);
}

inline float secondary_total_mass_rate_device(
    const float* sec_total_rates,
    int proj_idx,
    std::size_t section_id,
    float energy_mevu,
    float energy_min_MeV_per_u,
    float inv_energy_step,
    std::uint32_t num_energies) noexcept {
    if (sec_total_rates == nullptr || proj_idx < 0 || proj_idx >= 13 || section_id >= 25 ||
        num_energies < 2) {
        return 0.0F;
    }
    const float e_max =
        energy_min_MeV_per_u + static_cast<float>(num_energies - 1) / inv_energy_step;
    float e = energy_mevu;
    if (secondary_grid_is_legacy_v1(energy_min_MeV_per_u, num_energies)) {
        e = e < energy_min_MeV_per_u ? energy_min_MeV_per_u : (e > e_max ? e_max : e);
    } else if (!(e >= energy_min_MeV_per_u) || !(e <= e_max)) {
        return 0.0F;
    }
    const float node_flt = (e - energy_min_MeV_per_u) * inv_energy_step;
    auto node_idx = static_cast<int>(node_flt);
    if (node_idx < 0) node_idx = 0;
    if (node_idx >= static_cast<int>(num_energies) - 1) {
        node_idx = static_cast<int>(num_energies) - 2;
    }
    const float frac = node_flt - static_cast<float>(node_idx);

    const std::size_t base =
        (static_cast<std::size_t>(proj_idx) * 25 + section_id) *
            static_cast<std::size_t>(num_energies) +
        static_cast<std::size_t>(node_idx);
    return sec_total_rates[base] + frac * (sec_total_rates[base + 1] - sec_total_rates[base]);
}

inline int sample_secondary_target_device(
    const float* sec_partial_rates,
    int proj_idx,
    std::size_t section_id,
    float energy_mevu,
    float u01,
    float energy_min_MeV_per_u,
    float inv_energy_step,
    std::uint32_t num_energies) noexcept {
    constexpr int kSecCanonicalTargets[13] = {1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22};
    // No water/oxygen fallback: null tables, out-of-range indices, or
    // out-of-grid energies yield an invalid target (0) so the caller records
    // an explicit unsupported event instead of silently replaying oxygen
    // physics. Zero-weight partials (channels without event support at this
    // energy) are never selected: the categorical draw skips them, and an
    // all-zero row returns 0. v1-grid callers keep the legacy endpoint clamp
    // (see secondary_grid_is_legacy_v1); other grids are strict.
    if (sec_partial_rates == nullptr || proj_idx < 0 || proj_idx >= 13 || section_id >= 25 ||
        num_energies < 2) {
        return 0;
    }
    const float e_max =
        energy_min_MeV_per_u + static_cast<float>(num_energies - 1) / inv_energy_step;
    float e = energy_mevu;
    if (secondary_grid_is_legacy_v1(energy_min_MeV_per_u, num_energies)) {
        e = e < energy_min_MeV_per_u ? energy_min_MeV_per_u : (e > e_max ? e_max : e);
    } else if (!(e >= energy_min_MeV_per_u) || !(e <= e_max)) {
        return 0;
    }
    const float node_flt = (e - energy_min_MeV_per_u) * inv_energy_step;
    auto node_idx = static_cast<int>(node_flt);
    if (node_idx < 0) node_idx = 0;
    if (node_idx >= static_cast<int>(num_energies) - 1) {
        node_idx = static_cast<int>(num_energies) - 2;
    }
    const float frac = node_flt - static_cast<float>(node_idx);

    float partials[13];
    float sum = 0.0F;
    const std::size_t proj_sec_base =
        (static_cast<std::size_t>(proj_idx) * 25 + section_id) *
        (13 * static_cast<std::size_t>(num_energies));
    for (std::size_t t = 0; t < 13; ++t) {
        const std::size_t base = proj_sec_base +
                                 t * static_cast<std::size_t>(num_energies) +
                                 static_cast<std::size_t>(node_idx);
        partials[t] = sec_partial_rates[base] + frac * (sec_partial_rates[base + 1] - sec_partial_rates[base]);
        if (partials[t] < 0.0F) partials[t] = 0.0F;
        sum += partials[t];
    }

    if (sum <= 1.0e-12F) {
        return 0;
    }

    const float u = (u01 < 0.0F ? 0.0F : (u01 >= 1.0F ? 0.9999999F : u01)) * sum;
    float cum = 0.0F;
    for (std::size_t t = 0; t < 13; ++t) {
        cum += partials[t];
        if (u < cum || t == 12) {
            return kSecCanonicalTargets[t];
        }
    }
    return kSecCanonicalTargets[12];
}

// ---- v3 data-driven path (binary version 3 only; v1 never calls these) ----

// Projectile registry lookup over the uploaded bundle-ordered key list
// (layout [z0,a0,z1,a1,...], np entries). No hardcoded isotopes: any
// (Z,A) absent from the list returns -1 (unsupported, deposited locally).
inline int secondary_projectile_lut_index_device(const std::int32_t* keys,
                                                 std::uint32_t num_projectiles,
                                                 int pz, int pa) noexcept {
    if (keys == nullptr) {
        return -1;
    }
    for (std::uint32_t i = 0; i < num_projectiles; ++i) {
        if (keys[2 * i] == pz && keys[2 * i + 1] == pa) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

struct SecondaryMaskedRates {
    float partials[13];
    float total{0.0F};
};

// v3 masked partial computation. The per-channel domain is applied BEFORE
// interpolation: any channel without support, or queried outside its
// [emin, emax], contributes EXACTLY 0 (no boundary-interpolation leak).
// total is the sum of the masked partials, so the hazard can never fire
// where no channel has support. Strict global-grid zero outside
// [grid_emin, grid_emax]; no clamp, no extrapolation. Domain arrays are
// [proj][target] row-major with num_projectiles rows.
inline SecondaryMaskedRates secondary_masked_rates_device(
    const float* sec_partial_rates,
    const float* dom_emin,
    const float* dom_emax,
    const unsigned char* dom_has,
    std::uint32_t num_projectiles,
    int proj_idx,
    std::size_t section_id,
    float energy_mevu,
    float grid_emin,
    float inv_energy_step,
    std::uint32_t num_energies) noexcept {
    SecondaryMaskedRates out;
    for (std::size_t t = 0; t < 13; ++t) {
        out.partials[t] = 0.0F;
    }
    if (sec_partial_rates == nullptr || dom_emin == nullptr || dom_emax == nullptr ||
        dom_has == nullptr || proj_idx < 0 ||
        static_cast<std::uint32_t>(proj_idx) >= num_projectiles || section_id >= 25 ||
        num_energies < 2) {
        return out;
    }
    const float e_max =
        grid_emin + static_cast<float>(num_energies - 1) / inv_energy_step;
    const float e = energy_mevu;
    if (!(e >= grid_emin) || !(e <= e_max)) {
        return out;
    }
    float node_flt = (e - grid_emin) * inv_energy_step;
    auto node_idx = static_cast<int>(node_flt);
    if (node_idx < 0) node_idx = 0;
    if (node_idx >= static_cast<int>(num_energies) - 1) {
        node_idx = static_cast<int>(num_energies) - 2;
    }
    const float frac = node_flt - static_cast<float>(node_idx);

    const std::size_t proj_sec_base =
        (static_cast<std::size_t>(proj_idx) * 25 + section_id) *
        (13 * static_cast<std::size_t>(num_energies));
    const std::size_t dom_base = static_cast<std::size_t>(proj_idx) * 13;
    float sum = 0.0F;
    for (std::size_t t = 0; t < 13; ++t) {
        if (dom_has[dom_base + t] == 0 || e < dom_emin[dom_base + t] ||
            e > dom_emax[dom_base + t]) {
            out.partials[t] = 0.0F;
            continue;
        }
        const std::size_t base = proj_sec_base +
                                 t * static_cast<std::size_t>(num_energies) +
                                 static_cast<std::size_t>(node_idx);
        float v = sec_partial_rates[base] +
                  frac * (sec_partial_rates[base + 1] - sec_partial_rates[base]);
        if (v < 0.0F) v = 0.0F;
        out.partials[t] = v;
        sum += v;
    }
    // Hazard/sampler threshold unity: a total at or below the sampler's
    // all-zero cutoff is exactly zero, so a hazard can never fire where the
    // sampler draws empty at the same energy (float-sliver guard).
    if (!(sum > 1.0e-12F)) {
        for (std::size_t t = 0; t < 13; ++t) {
            out.partials[t] = 0.0F;
        }
        sum = 0.0F;
    }
    out.total = sum;
    return out;
}

// Categorical draw from v3 masked partials. Zero-weight channels are never
// selected; an all-zero row returns invalid target 0.
inline int sample_masked_secondary_target_device(const float (&partials)[13],
                                                 float u01) noexcept {
    constexpr int kTargets[13] = {1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22};
    float sum = 0.0F;
    for (std::size_t t = 0; t < 13; ++t) {
        sum += partials[t];
    }
    if (sum <= 1.0e-12F) {
        return 0;
    }
    const float u = (u01 < 0.0F ? 0.0F : (u01 >= 1.0F ? 0.9999999F : u01)) * sum;
    float cum = 0.0F;
    for (std::size_t t = 0; t < 13; ++t) {
        cum += partials[t];
        if (u < cum || t == 12) {
            return kTargets[t];
        }
    }
    return kTargets[12];
}

}  // namespace carbon

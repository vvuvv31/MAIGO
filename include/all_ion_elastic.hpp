#pragma once

#include "beamline_em.hpp"
#include "nuclear_elastic_scattering.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace carbon {

// ELBANK01 dimensions and projectile order are part of the binary format.
inline constexpr int kElasticProjectileCount = 18;
inline constexpr int kElasticMaterialCount = 26;
inline constexpr int kElasticTargetCount = 13;
inline constexpr std::array<std::array<int, 2>, kElasticProjectileCount> elastic_projectiles{
    {{1, 1},
     {1, 2},
     {1, 3},
     {2, 3},
     {2, 4},
     {2, 6},
     {3, 6},
     {3, 7},
     {4, 7},
     {4, 9},
     {4, 10},
     {5, 8},
     {5, 10},
     {5, 11},
     {6, 10},
     {6, 11},
     {6, 12},
     {4, 6}}};

struct ElasticSample {
    float transfer_fraction{}, target_mass_MeV{};
    std::uint32_t target_a{};
};

struct ElasticDraw {
    NuclearElasticOutcome outcome{};
    int target_z{}, target_a{};
    bool valid{};
};

struct AllIonElasticView {
    const float* energies{};
    const float* rates{};
    const ElasticSample* samples{};
    const double* masses{};
    std::uint32_t ne{}, nq{};
    const float* primary_totals{};
    int totals_projectile{-1};
    const float* species_totals{};

    // Energies are MeV/u; rates are material mass rates. Keep target rates
    // separate from material-independent final-state quantiles.
    auto rate_index(int projectile, int section, int target, int energy) const noexcept {
        return ((projectile * kElasticMaterialCount + section) * kElasticTargetCount + target) *
                   ne +
               energy;
    }
    auto sample_index(int projectile, int target, int energy, unsigned quantile) const noexcept {
        return ((projectile * kElasticTargetCount + target) * ne + energy) * nq + quantile;
    }

    int projectile(int z, int a) const noexcept {
        for (int i = 0; i < kElasticProjectileCount; ++i) {
            if (elastic_projectiles[i][0] == z && elastic_projectiles[i][1] == a)
                return i;
        }
        return -1;
    }

    int cell(float energy) const noexcept {
        if (!energies || !std::isfinite(energy) || energy < energies[0] ||
            energy > energies[ne - 1])
            return -1;
        unsigned lower = 0, upper = ne - 1;
        while (upper - lower > 1) {
            const unsigned middle = (lower + upper) / 2;
            if (energy < energies[middle])
                upper = middle;
            else
                lower = middle;
        }
        return static_cast<int>(lower);
    }

    float rate(int projectile, int section, float energy, int target = -1) const noexcept {
        const int cell_index = cell(energy);
        if (projectile < 0 || projectile >= kElasticProjectileCount || section < 0 ||
            section >= kElasticMaterialCount || cell_index < 0)
            return -1;
        const float weight =
            (energy - energies[cell_index]) / (energies[cell_index + 1] - energies[cell_index]);
        float sum = 0;
        for (int element = 0; element < kElasticTargetCount; ++element) {
            if (target >= 0 && element != target)
                continue;
            const auto base = rate_index(projectile, section, element, cell_index);
            sum += rates[base] * (1 - weight) + rates[base + 1] * weight;
        }
        return sum;
    }

    // Derived node totals leave target rates and sampled final states intact.
    template <bool UseDerivedTotals = true>
    float primary_total_rate(int projectile, int section, float energy) const noexcept {
        if constexpr (!UseDerivedTotals) {
            return rate(projectile, section, energy);
        } else {
            if (!primary_totals || projectile != totals_projectile)
                return rate(projectile, section, energy);
            const int cell_index = cell(energy);
            if (projectile < 0 || projectile >= kElasticProjectileCount || section < 0 ||
                section >= kElasticMaterialCount || cell_index < 0)
                return -1;
            const float weight =
                (energy - energies[cell_index]) / (energies[cell_index + 1] - energies[cell_index]);
            const auto base = section * ne + cell_index;
            return primary_totals[base] * (1 - weight) + primary_totals[base + 1] * weight;
        }
    }

    template <bool UseDerivedTotals = true>
    float species_total_rate(int projectile, int section, float energy) const noexcept {
        if constexpr (!UseDerivedTotals) {
            return rate(projectile, section, energy);
        } else {
            if (!species_totals)
                return rate(projectile, section, energy);
            const int cell_index = cell(energy);
            if (projectile < 0 || projectile >= kElasticProjectileCount || section < 0 ||
                section >= kElasticMaterialCount || cell_index < 0)
                return -1;
            const float weight =
                (energy - energies[cell_index]) / (energies[cell_index + 1] - energies[cell_index]);
            const auto base = (projectile * kElasticMaterialCount + section) * ne + cell_index;
            return species_totals[base] * (1 - weight) + species_totals[base + 1] * weight;
        }
    }

    template <bool UsePrimary, bool UseSpecies>
    float transport_total_rate(int projectile, int section, float energy) const noexcept {
        if constexpr (UseSpecies)
            return species_total_rate<true>(projectile, section, energy);
        else
            return primary_total_rate<UsePrimary>(projectile, section, energy);
    }

    ElasticDraw draw(int projectile, int section, float kinetic_energy_MeV, float dx, float dy,
                     float dz, float target_uniform, float node_uniform, float sample_uniform,
                     float azimuth_uniform) const noexcept {
        ElasticDraw out{};
        if (projectile < 0 || projectile >= kElasticProjectileCount)
            return out;
        const float energy = kinetic_energy_MeV / elastic_projectiles[projectile][1];
        const int cell_index = cell(energy);
        const float total = rate(projectile, section, energy);
        if (cell_index < 0 || !(total > 0))
            return out;

        // Select a target from interpolated rates. The last positive target
        // handles roundoff at the upper end of the cumulative distribution.
        float accumulated = 0;
        int target = -1, last_positive = -1;
        for (int element = 0; element < kElasticTargetCount; ++element) {
            const float value = rate(projectile, section, energy, element);
            if (value > 0)
                last_positive = element;
            accumulated += value;
            if (target_uniform * total < accumulated) {
                target = element;
                break;
            }
        }
        if (target < 0)
            target = last_positive;
        if (target < 0)
            return out;

        // Choose a neighboring energy node with rate-weighted probability,
        // then one quantile at that node; do not interpolate final states.
        const float weight =
            (energy - energies[cell_index]) / (energies[cell_index + 1] - energies[cell_index]);
        const auto base = rate_index(projectile, section, target, cell_index);
        const float lower = (1 - weight) * rates[base], upper = weight * rates[base + 1];
        const int node = cell_index + (node_uniform * (lower + upper) >= lower);
        auto quantile = static_cast<unsigned>(sample_uniform * nq);
        if (quantile >= nq)
            quantile = nq - 1;
        const auto record = samples[sample_index(projectile, target, node, quantile)];
        if (record.target_a == 0 || !(record.target_mass_MeV > 0))
            return out;
        out.outcome =
            elastic_two_body(kinetic_energy_MeV, dx, dy, dz, 1.0 - 2.0 * record.transfer_fraction,
                             azimuth_uniform, masses[projectile], record.target_mass_MeV);
        out.target_z = kSchneiderCanonicalZ[target];
        out.target_a = record.target_a;
        out.valid = true;
        return out;
    }
};

struct AllIonElasticTable {
    std::vector<float> energies, rates;
    std::vector<ElasticSample> samples;
    std::array<double, kElasticProjectileCount> masses{};
    std::uint32_t nq{};

    static AllIonElasticTable load(const std::filesystem::path&);
    AllIonElasticView view() const {
        return {energies.data(),
                rates.data(),
                samples.data(),
                masses.data(),
                static_cast<std::uint32_t>(energies.size()),
                nq};
    }
};

struct ElasticRecoilStoppingView {
    const std::int32_t* keys{};
    const float* energies{};
    const float* values{};
    std::uint32_t np{}, ne{};
    // Derived full total-stopping CSDA for additional heavy recoils. ELRSP001
    // already contains unrestricted electronic + separate nuclear stopping;
    // electronic LET continues to use the independent LERSP001 bank.
    const BeamlineEmNode* full_nodes{};
    const double* full_ranges{};

    BeamlineEmPoint loss_point(int projectile, int section, float density) const {
        if (!full_nodes || !full_ranges || projectile < 0 || projectile >= int(np) || section < 0 ||
            section >= kElasticMaterialCount)
            return {};
        const auto base = (projectile * kElasticMaterialCount + section) * ne;
        return {full_nodes + base, full_ranges + base, ne, {}, density};
    }

    int projectile(int z, int a) const noexcept {
        for (unsigned i = 0; i < np; ++i) {
            if (keys[2 * i] == z && keys[2 * i + 1] == a)
                return i;
        }
        return -1;
    }

    float stopping(int projectile, int section, float energy) const noexcept {
        if (projectile < 0 || projectile >= static_cast<int>(np) || section < 0 ||
            section >= kElasticMaterialCount || !std::isfinite(energy) || energy < energies[0] ||
            energy > energies[ne - 1])
            return -1;
        unsigned lower = 0, upper = ne - 1;
        while (upper - lower > 1) {
            const unsigned middle = (lower + upper) / 2;
            if (energy < energies[middle])
                upper = middle;
            else
                lower = middle;
        }
        const float weight = (energy - energies[lower]) / (energies[upper] - energies[lower]);
        const auto base = (projectile * kElasticMaterialCount + section) * ne;
        return values[base + lower] * (1 - weight) + values[base + upper] * weight;
    }
};

struct ElasticRecoilStoppingTable {
    std::vector<std::int32_t> keys;
    std::vector<float> energies, values;
    static ElasticRecoilStoppingTable load(const std::filesystem::path&);
    // Separate magic prevents a total-stopping transport bank being used for electronic LET.
    static ElasticRecoilStoppingTable load_electronic(const std::filesystem::path&);
};

} // namespace carbon

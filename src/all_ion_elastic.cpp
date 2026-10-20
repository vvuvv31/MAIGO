#include "all_ion_elastic.hpp"
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace carbon {

AllIonElasticTable AllIonElasticTable::load(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    char magic[8];
    input.read(magic, 8);
    std::uint32_t header[6];
    input.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!input || std::memcmp(magic, "ELBANK01", 8) || header[0] != 1 ||
        header[1] != kElasticProjectileCount || header[2] != kElasticMaterialCount ||
        header[3] != kElasticTargetCount || header[4] < 2 || header[4] > 10000 || header[5] < 32 ||
        header[5] > 65536)
        throw std::runtime_error("Invalid elastic bank schema");
    const std::uint64_t energy_count = header[4], sample_count = header[5],
                        rate_count = kElasticProjectileCount * kElasticMaterialCount *
                                     kElasticTargetCount * energy_count,
                        final_state_count = kElasticProjectileCount * kElasticTargetCount *
                                            energy_count * sample_count;
    if (std::filesystem::file_size(path) !=
        32 + 18 * 8 + energy_count * 4 + rate_count * 4 + final_state_count * 12)
        throw std::runtime_error("Elastic bank size mismatch");
    AllIonElasticTable table;
    table.nq = sample_count;
    table.energies.resize(energy_count);
    table.rates.resize(rate_count);
    table.samples.resize(final_state_count);
    input.read(reinterpret_cast<char*>(table.masses.data()), 18 * 8);
    input.read(reinterpret_cast<char*>(table.energies.data()), energy_count * 4);
    input.read(reinterpret_cast<char*>(table.rates.data()), rate_count * 4);
    input.read(reinterpret_cast<char*>(table.samples.data()), final_state_count * 12);
    if (!input)
        throw std::runtime_error("Truncated elastic bank");
    for (double mass : table.masses)
        if (!std::isfinite(mass) || mass <= 0)
            throw std::runtime_error("Bad elastic mass");
    for (unsigned i = 0; i < energy_count; ++i)
        if (!std::isfinite(table.energies[i]) || table.energies[i] <= 0 ||
            (i && table.energies[i] <= table.energies[i - 1]))
            throw std::runtime_error("Bad elastic grid");
    for (float rate : table.rates)
        if (!std::isfinite(rate) || rate < 0)
            throw std::runtime_error("Bad elastic rate");
    for (auto sample : table.samples)
        if (!std::isfinite(sample.transfer_fraction) || sample.transfer_fraction < 0 ||
            sample.transfer_fraction > 1 || !std::isfinite(sample.target_mass_MeV) ||
            sample.target_mass_MeV <= 0 || sample.target_a > 250)
            throw std::runtime_error("Bad elastic sample");
    // Rates are [projectile][material][target][energy]; final states are
    // [projectile][target][energy][quantile], shared across material sections.
    // Every node with an active material rate must contain valid target samples.
    for (int projectile = 0; projectile < kElasticProjectileCount; ++projectile) {
        for (int target = 0; target < kElasticTargetCount; ++target) {
            for (unsigned energy = 0; energy < energy_count; ++energy) {
                bool has_active_rate = false;
                for (int material = 0; material < kElasticMaterialCount; ++material) {
                    const auto index =
                        ((projectile * kElasticMaterialCount + material) * kElasticTargetCount +
                         target) *
                            energy_count +
                        energy;
                    has_active_rate |= table.rates[index] > 0;
                }
                if (has_active_rate) {
                    for (unsigned quantile = 0; quantile < sample_count; ++quantile) {
                        const auto index =
                            ((projectile * kElasticTargetCount + target) * energy_count + energy) *
                                sample_count +
                            quantile;
                        if (table.samples[index].target_a == 0)
                            throw std::runtime_error("Missing active elastic channel");
                    }
                }
            }
        }
    }
    return table;
}

namespace {
ElasticRecoilStoppingTable read_recoil_stopping(const std::filesystem::path& path,
                                                bool electronic) {
    std::ifstream input(path, std::ios::binary);
    char magic[8];
    input.read(magic, 8);
    std::uint32_t header[3];
    input.read(reinterpret_cast<char*>(header), 12);
    if (!input || std::memcmp(magic, electronic ? "LERSP001" : "ELRSP001", 8) || header[0] == 0 ||
        header[0] > 200 || header[1] != 26 || header[2] < 2 || header[2] > 10000)
        throw std::runtime_error("Invalid recoil stopping schema");
    const std::uint64_t projectile_count = header[0], energy_count = header[2],
                        value_count = projectile_count * 26 * energy_count;
    if (std::filesystem::file_size(path) !=
        20 + projectile_count * 8 + energy_count * 4 + value_count * 4)
        throw std::runtime_error("Recoil stopping size mismatch");
    ElasticRecoilStoppingTable table;
    table.keys.resize(projectile_count * 2);
    table.energies.resize(energy_count);
    table.values.resize(value_count);
    input.read(reinterpret_cast<char*>(table.keys.data()), projectile_count * 8);
    input.read(reinterpret_cast<char*>(table.energies.data()), energy_count * 4);
    input.read(reinterpret_cast<char*>(table.values.data()), value_count * 4);
    if (!input)
        throw std::runtime_error("Truncated recoil stopping");
    for (unsigned i = 0; i < projectile_count; ++i) {
        if (table.keys[2 * i] <= 0 || table.keys[2 * i + 1] < table.keys[2 * i] ||
            table.keys[2 * i + 1] > 250)
            throw std::runtime_error("Invalid recoil isotope");
        for (unsigned previous = 0; previous < i; ++previous)
            if (table.keys[2 * i] == table.keys[2 * previous] &&
                table.keys[2 * i + 1] == table.keys[2 * previous + 1])
                throw std::runtime_error("Duplicate recoil isotope");
    }
    for (unsigned i = 0; i < energy_count; ++i)
        if (!std::isfinite(table.energies[i]) || table.energies[i] <= 0 ||
            (i && table.energies[i] <= table.energies[i - 1]))
            throw std::runtime_error("Bad recoil grid");
    for (float v : table.values)
        if (!std::isfinite(v) || v <= 0)
            throw std::runtime_error("Bad recoil stopping value");
    return table;
}
} // namespace
ElasticRecoilStoppingTable ElasticRecoilStoppingTable::load(const std::filesystem::path& path) {
    return read_recoil_stopping(path, false);
}
ElasticRecoilStoppingTable
ElasticRecoilStoppingTable::load_electronic(const std::filesystem::path& path) {
    return read_recoil_stopping(path, true);
}
} // namespace carbon

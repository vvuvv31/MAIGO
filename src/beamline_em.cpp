#include "beamline_em.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>

namespace carbon {

BeamlineEmPackage BeamlineEmPackage::load(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    char magic[8]{};
    std::array<std::uint32_t, 4> header{};
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    const auto [version, material_count, species_count, node_count] = header;
    const std::uint64_t expected =
        24ULL + material_count * sizeof(BeamlineEmMaterial) +
        species_count * sizeof(BeamlineEmSpecies) +
        std::uint64_t(material_count) * species_count * sizeof(BeamlineEmRecord) +
        std::uint64_t(node_count) * sizeof(BeamlineEmNode);
    if (!in || std::memcmp(magic, "BLEM0001", 8) || version != 1 || material_count != 2 ||
        !species_count || species_count > 512 || node_count < 2 ||
        expected != std::filesystem::file_size(path) || expected > 1024ULL * 1024 * 1024)
        throw std::runtime_error("Invalid full beamline electronic response header");
    BeamlineEmPackage package;
    package.materials.resize(material_count);
    package.species.resize(species_count);
    package.records.resize(material_count * species_count);
    package.nodes.resize(node_count);
    package.ranges.resize(node_count);
    auto read = [&](auto& values) {
        in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(values[0]));
    };
    read(package.materials);
    read(package.species);
    read(package.records);
    read(package.nodes);
    if (!in)
        throw std::runtime_error("Truncated full beamline electronic response");
    for (unsigned i = 0; i < material_count; ++i) {
        const auto material = package.materials[i];
        const float expected_density = i == 0 ? 8.96f : .00120479f;
        if (std::abs(material.density - expected_density) > expected_density * 1e-6f ||
            !(material.excitation_MeV > 0 && material.e0_MeV > 0) ||
            !std::isfinite(material.density + material.excitation_MeV + material.e0_MeV))
            throw std::runtime_error("Full beamline response requires native G4_Cu/G4_AIR");
    }
    std::set<std::pair<unsigned, unsigned>> seen;
    for (const auto species : package.species)
        if (!species.z || species.a < species.z || !(species.mass_MeV > 0) ||
            !std::isfinite(species.mass_MeV) || !seen.emplace(species.z, species.a).second)
            throw std::runtime_error("Invalid full beamline projectile");
    std::uint64_t offset = 0;
    for (const auto record : package.records) {
        if (record.offset != offset || record.count < 2 || offset + record.count > node_count)
            throw std::runtime_error("Invalid full beamline response bounds");
        double range = 0;
        for (unsigned i = 0; i < record.count; ++i) {
            const auto node = package.nodes[record.offset + i];
            if (!(node.energy > 0 && node.stopping > 0 && node.ion_variance >= 0 &&
                  node.universal_variance >= 0) ||
                !std::isfinite(node.energy + node.stopping + node.ion_variance +
                               node.universal_variance))
                throw std::runtime_error("Invalid full beamline response node");
            if (!i)
                range = 2. * node.energy / node.stopping; // native low-energy sqrt(E) continuation
            else {
                const auto previous = package.nodes[record.offset + i - 1];
                if (!(node.energy > previous.energy))
                    throw std::runtime_error("Unordered full beamline energies");
                range += linear_stopping_path(double(node.energy) - previous.energy,
                                              double(previous.stopping), double(node.stopping));
            }
            package.ranges[record.offset + i] = range;
            if (!(package.ranges[record.offset + i] > 0) ||
                !std::isfinite(package.ranges[record.offset + i]) ||
                (i && package.ranges[record.offset + i] <= package.ranges[record.offset + i - 1]))
                throw std::runtime_error(
                    "Non-monotone full electronic CSDA range at record offset=" +
                    std::to_string(record.offset) + " node=" + std::to_string(i) +
                    " E=" + std::to_string(node.energy));
        }
        offset += record.count;
    }
    if (offset != node_count)
        throw std::runtime_error("Unclaimed full beamline nodes");
    return package;
}
} // namespace carbon

#include "urban_mcs_package.hpp"
#include "detail/native_spline_wire.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>

namespace carbon {
namespace {
// URBANV22 disk ABI stays private and fully validated. Native loss splines
// and factors are not the active unrestricted loss model in the runtime.
struct DiskUrbanMcsRecord {
    float mass_ratio, minimum_scaled_energy;
    std::uint32_t node_offset, node_count, mfp_offset, mfp_count;
    std::uint32_t offsets[3], counts[3];
};
static_assert(sizeof(DiskUrbanMcsRecord) == 48);
} // namespace
UrbanMcsPackage UrbanMcsPackage::load(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    char magic[8];
    std::array<std::uint32_t, 7> header{};
    stream.read(magic, 8);
    stream.read(reinterpret_cast<char*>(header.data()), 28);
    const auto [version, material_count, species_count, node_count, mfp_count, segment_count,
                validation_flag] = header;
    if (!stream || std::memcmp(magic, "URBANV22", 8) || version != 2 || validation_flag != 1 ||
        material_count == 0 || material_count > 2048 || species_count == 0 || species_count > 512 ||
        node_count < 2 || mfp_count < 2 || segment_count < 3)
        throw std::runtime_error("Invalid or unvalidated Urban MSC package header");
    const std::uint64_t record_count = std::uint64_t(material_count) * species_count;
    const std::uint64_t bytes = 36 + std::uint64_t(material_count) * 20 +
                                std::uint64_t(species_count) * 12 + record_count * 48 +
                                std::uint64_t(node_count) * 8 + std::uint64_t(mfp_count) * 8 +
                                std::uint64_t(segment_count) * 24;
    if (bytes != std::filesystem::file_size(path) || bytes > 4ULL * 1024 * 1024 * 1024)
        throw std::runtime_error("Urban MSC package size is invalid");
    UrbanMcsPackage result;
    result.materials.resize(material_count);
    result.species.resize(species_count);
    result.mfps.resize(mfp_count);
    std::vector<DiskUrbanMcsRecord> records(record_count);
    std::vector<UrbanMcsNode> nodes(node_count);
    std::vector<detail::NativeSplineSegment> segments(segment_count);
    auto read = [&](auto& values) {
        stream.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(values[0])));
    };
    read(result.materials);
    read(result.species);
    read(records);
    read(nodes);
    read(result.mfps);
    read(segments);
    if (!stream)
        throw std::runtime_error("Truncated Urban MSC package");
    std::set<std::pair<int, float>> material_keys;
    for (const auto& material : result.materials)
        if (material.section < -3 || material.section > 24 ||
            !(material.density > 0 && material.zeff > 0 && material.radiation_length_mm > 0 &&
              material.production_cut_mm == .05f) ||
            !std::isfinite(material.density + material.zeff + material.radiation_length_mm) ||
            !material_keys.emplace(material.section, material.density).second)
            throw std::runtime_error("Invalid or duplicate Urban material couple");
    std::set<std::pair<unsigned, unsigned>> ions;
    for (const auto& ion : result.species)
        if (ion.z == 0 || ion.a < ion.z || ion.a > 250 || !(ion.mass_mev > 0) ||
            !std::isfinite(ion.mass_mev) || !ions.emplace(ion.z, ion.a).second)
            throw std::runtime_error("Invalid or duplicate Urban projectile");
    auto check_nodes = [](const auto& values, unsigned start, unsigned count) {
        if (count < 2 || std::uint64_t(start) + count > values.size())
            throw std::runtime_error("Invalid Urban node bounds");
        for (unsigned index = 0; index < count; ++index) {
            const auto& node = values[start + index];
            if (!(node.energy > 0 && node.value > 0) || !std::isfinite(node.energy + node.value) ||
                (index && !(node.energy > values[start + index - 1].energy)))
                throw std::runtime_error("Invalid Urban factor/MFP nodes");
        }
    };
    for (const auto& record : records) {
        if (!(record.mass_ratio > 0 && record.minimum_scaled_energy > 0) ||
            !std::isfinite(record.mass_ratio + record.minimum_scaled_energy))
            throw std::runtime_error("Invalid Urban mass scaling");
        check_nodes(nodes, record.node_offset, record.node_count);
        check_nodes(result.mfps, record.mfp_offset, record.mfp_count);
        for (unsigned spline = 0; spline < 3; ++spline) {
            if (!record.counts[spline] ||
                std::uint64_t(record.offsets[spline]) + record.counts[spline] > segment_count)
                throw std::runtime_error("Invalid Urban spline bounds");
            for (unsigned index = 0; index < record.counts[spline]; ++index) {
                const auto& segment = segments[record.offsets[spline] + index];
                if (!(segment.lower > 0 && segment.upper > segment.lower && segment.y0 > 0 &&
                      segment.ythird > 0 && segment.ytwothirds > 0 && segment.y1 > 0) ||
                    !std::isfinite(segment.lower + segment.upper + segment.y0 + segment.ythird +
                                   segment.ytwothirds + segment.y1) ||
                    (index && segment.lower != segments[record.offsets[spline] + index - 1].upper))
                    throw std::runtime_error("Invalid Urban native spline interval");
            }
        }
    }
    // Every factor and spline was checked above, even though the active
    // unrestricted model no longer consumes either numerical payload.
    // Certify that native FP32 factor interpolation is positive/finite for
    // every finite energy inside its original support, not just at nodes.
    // The 64-epsilon margin bounds subtraction/division/lerp roundoff; normal
    // gaps avoid FTZ zero denominators. Unusual curves retain their exact
    // original numerical guard instead of broadening accepted support.
    result.records.reserve(records.size());
    bool needs_native_guards = false;
    for (const auto& record : records) {
        const auto* begin = nodes.data() + record.node_offset;
        float minimum = begin[0].value, maximum = minimum;
        bool normal_gaps = true;
        for (unsigned index = 1; index < record.node_count; ++index) {
            minimum = std::min(minimum, begin[index].value);
            maximum = std::max(maximum, begin[index].value);
            normal_gaps = normal_gaps && begin[index].energy - begin[index - 1].energy >=
                                             std::numeric_limits<float>::min();
        }
        const bool certified = normal_gaps && minimum >= 16.f * std::numeric_limits<float>::min() &&
                               maximum <= std::numeric_limits<float>::max() / 8.f &&
                               minimum > 64.f * std::numeric_limits<float>::epsilon() * maximum;
        UrbanMcsRecord compact{begin[0].energy,
                               begin[record.node_count - 1].energy,
                               record.mfp_offset,
                               record.mfp_count,
                               0,
                               0};
        if (!certified) {
            needs_native_guards = true;
            compact.guard_offset = record.node_offset;
            compact.guard_count = record.node_count;
        }
        result.records.push_back(compact);
    }
    if (needs_native_guards)
        result.factor_guards = std::move(nodes);
    return result;
}
} // namespace carbon

#include "carbon/fred_event_library.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace carbon {
namespace {

constexpr std::uint32_t k_magic = 0x424c4546u;  // 'FELB' little-endian

struct Header {
    char magic[4];
    std::uint32_t version;
    std::uint32_t n_events;
    std::uint32_t max_frag;
    float ref_MeVu;
    std::uint16_t tgt_z;
    std::uint16_t tgt_a;
};

}  // namespace

FredEventLibrary load_fred_event_library(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open FRED event library: " + path.string());
    }
    Header header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || std::memcmp(header.magic, "FELB", 4) != 0 || header.version != 1 ||
        header.max_frag == 0 || header.max_frag > 12 || header.n_events == 0) {
        throw std::runtime_error("Invalid FRED event library header: " + path.string());
    }
    FredEventLibrary lib;
    lib.reference_energy_MeVu = header.ref_MeVu;
    lib.target_z = header.tgt_z;
    lib.target_a = header.tgt_a;
    lib.event_count = header.n_events;
    lib.max_fragments = header.max_frag;
    lib.fragment_count.resize(header.n_events);
    lib.neutron_ke_MeV.resize(header.n_events);
    const std::size_t nslot = static_cast<std::size_t>(header.n_events) * header.max_frag;
    lib.z.assign(nslot, 0);
    lib.a.assign(nslot, 0);
    lib.ke_MeV.assign(nslot, 0.0F);
    lib.ux.assign(nslot, 0.0F);
    lib.uy.assign(nslot, 0.0F);
    lib.uz.assign(nslot, 1.0F);
    for (std::uint32_t i = 0; i < header.n_events; ++i) {
        std::uint8_t nfrag = 0;
        float nke = 0.0F;
        input.read(reinterpret_cast<char*>(&nfrag), 1);
        input.read(reinterpret_cast<char*>(&nke), 4);
        lib.fragment_count[i] = nfrag;
        lib.neutron_ke_MeV[i] = nke;
        for (std::uint32_t f = 0; f < header.max_frag; ++f) {
            std::int8_t z = 0;
            std::int8_t a = 0;
            float ke = 0;
            float ux = 0;
            float uy = 0;
            float uz = 1;
            input.read(reinterpret_cast<char*>(&z), 1);
            input.read(reinterpret_cast<char*>(&a), 1);
            input.read(reinterpret_cast<char*>(&ke), 4);
            input.read(reinterpret_cast<char*>(&ux), 4);
            input.read(reinterpret_cast<char*>(&uy), 4);
            input.read(reinterpret_cast<char*>(&uz), 4);
            const std::size_t slot = static_cast<std::size_t>(i) * header.max_frag + f;
            lib.z[slot] = z;
            lib.a[slot] = a;
            lib.ke_MeV[slot] = ke;
            lib.ux[slot] = ux;
            lib.uy[slot] = uy;
            lib.uz[slot] = uz;
        }
    }
    if (!input) {
        throw std::runtime_error("Truncated FRED event library: " + path.string());
    }
    return lib;
}

}  // namespace carbon

// Host unit test: SCHNIOSP v1 format math (synthetic tables only;
// production data comes from TOPAS/Geant4 extraction, never synthesized).
#include "carbon/schneider_ion_stopping_table.hpp"
#include "carbon/sha256.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { ++failures; std::printf("FAIL: %s\n", what); }
}

void write_synthetic(const std::filesystem::path& bin,
                     const std::filesystem::path& meta) {
    carbon::SchneiderIonStoppingHeader h{};
    std::memcpy(h.magic, "SCHNIOSP", 8);
    h.version = 1;
    h.num_sections = carbon::kSchneiderIonSections;
    h.num_species = carbon::kSchneiderIonSpecies;
    h.num_energies = carbon::kSchneiderIonEnergies;
    h.energy_min_mevu = carbon::kSchneiderIonEnergyMin;
    h.energy_max_mevu = carbon::kSchneiderIonEnergyMax;
    h.energy_step_mevu = carbon::kSchneiderIonEnergyStep;
    std::ofstream out(bin, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    for (std::size_t s = 0; s < carbon::kSchneiderIonSections; ++s) {
        double rho = 0.01 + 0.1 * s;
        out.write(reinterpret_cast<const char*>(&rho), sizeof(rho));
    }
    // value = 1000 + 100*section + 10*species + 0.001*energy_index
    for (std::size_t s = 0; s < carbon::kSchneiderIonSections; ++s)
        for (std::size_t z = 0; z < carbon::kSchneiderIonSpecies; ++z)
            for (std::size_t i = 0; i < carbon::kSchneiderIonEnergies; ++i) {
                double v = 1000.0 + 100.0 * s + 10.0 * z + 0.001 * i;
                out.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }
    out.close();
    // sha of binary for the sidecar
    std::string sha = carbon::compute_file_sha256_hex(bin);
    std::ofstream m(meta);
    m << "{\"schema_version\": 1, \"format\": \"binary\", "
      << "\"data_filename\": \"" << bin.filename().string() << "\", "
      << "\"data_sha256\": \"" << sha << "\", "
      << "\"sections_count\": 25, \"species_count\": 18, "
      << "\"energies_count\": 60002, "
      << "\"quantity\": \"unrestricted electronic dE/dx, linear at unit density\", "
      << "\"energy_grid_MeV_per_u\": {\"minimum\":0.01,\"maximum\":6000.11,\"step\":0.1}, "
      << "\"species_za\": [[1,1],[1,2],[1,3],[2,3],[2,4],[2,6],[3,6],[3,7],[4,7],[4,9],[4,10],[5,8],[5,10],[5,11],[6,10],[6,11],[6,12],[4,6]], "
      << "\"provenance\": \"synthetic unit test vectors only\"}";
}

}  // namespace

int main() {
    const auto dir = std::filesystem::temp_directory_path() / "schniosp_test";
    std::filesystem::create_directories(dir);
    const auto bin = dir / "test_ion.bin";
    const auto meta = dir / "test_ion.metadata.json";
    write_synthetic(bin, meta);
    auto t = carbon::SchneiderIonStoppingTable::from_binary(bin, meta);
    check(t.num_sections() == 25, "sections");
    check(t.num_species() == 18, "species");
    check(t.num_energies() == 60002, "energies");
    // section 3, species 5, energy 10.01 + 0.05 (mid-node interpolation)
    double got = t.linear_stopping_at_unit_density(5, 3, 10.06);
    double want = 1000.0 + 300.0 + 50.0 + 0.001 * (100.5);
    check(std::abs(got - want) < 1e-9, "interp");
    // Domain violations must reject; no density-only fallback or NaN clamp.
    check(t.linear_stopping_at_unit_density(0,0,0.0)<0,"low reject");
    check(t.linear_stopping_at_unit_density(0,0,NAN)<0,"NaN reject");
    check(t.linear_stopping_at_unit_density(0,0,INFINITY)<0,"infinity reject");
    check(t.linear_stopping_at_unit_density(0,0,6000.11)>0,"upper endpoint");
    check(t.linear_stopping_at_unit_density(0, 0, 7000.0) < 0, "high reject");
    check(t.linear_stopping_at_unit_density(99, 0, 100.0) < 0, "bad species");
    check(t.linear_stopping_at_unit_density(0, 99, 100.0) < 0, "bad section");
    auto flat = t.to_flat_float();
    check(flat.size() == 25u * 18u * 60002u, "flat size");
    const std::size_t base = (3u * 18u + 5u) * 60002u;
    check(std::abs(flat[base + 100] - float(1000.0 + 300.0 + 50.0 + 0.1)) < 1e-3f,
          "flat layout");
    // corrupted magic must throw
    {
        std::fstream f(bin, std::ios::binary | std::ios::in | std::ios::out);
        f.write("XXXXXXXX", 8);
        f.close();
        bool threw = false;
        try { carbon::SchneiderIonStoppingTable::from_binary(bin, meta); }
        catch (...) { threw = true; }
        check(threw, "bad magic throws");
    }
    std::filesystem::remove_all(dir);
    if (failures == 0) std::printf("ion stopping format tests passed\n");
    return failures ? 1 : 0;
}

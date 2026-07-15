#include "carbon/ct_grid.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace carbon {

float hu_to_density_g_per_cm3(float hu) noexcept {
    if (hu < -1000.0F) {
        hu = -1000.0F;
    }
    if (hu > 3000.0F) {
        hu = 3000.0F;
    }
    if (hu < -980.0F) {
        return 0.001205F;
    }
    if (hu < 0.0F) {
        return 0.001205F + (hu + 980.0F) * (1.0F - 0.001205F) / 980.0F;
    }
    if (hu < 1000.0F) {
        return 1.0F + 0.001F * hu;
    }
    return 2.0F + 0.0005F * (hu - 1000.0F);
}

std::uint8_t density_to_material_id(float density_g_per_cm3) noexcept {
    if (density_g_per_cm3 < 0.1F) {
        return 0;
    }
    if (density_g_per_cm3 < 0.7F) {
        return 1;
    }
    if (density_g_per_cm3 < 1.25F) {
        return 2;
    }
    return 3;
}

CtGrid CtGrid::from_binary(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open CT grid: " + path.string());
    }
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    input.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != magic_value ||
        (version != version_value && version != version_v2 && version != version_legacy)) {
        throw std::runtime_error("Invalid CT grid magic/version: " + path.string());
    }
    CtGrid grid;
    grid.file_version = version;
    input.read(reinterpret_cast<char*>(&grid.nx), sizeof(grid.nx));
    input.read(reinterpret_cast<char*>(&grid.ny), sizeof(grid.ny));
    input.read(reinterpret_cast<char*>(&grid.nz), sizeof(grid.nz));
    input.read(reinterpret_cast<char*>(&grid.origin_x_mm), sizeof(grid.origin_x_mm));
    input.read(reinterpret_cast<char*>(&grid.origin_y_mm), sizeof(grid.origin_y_mm));
    input.read(reinterpret_cast<char*>(&grid.origin_z_mm), sizeof(grid.origin_z_mm));
    input.read(reinterpret_cast<char*>(&grid.spacing_x_mm), sizeof(grid.spacing_x_mm));
    input.read(reinterpret_cast<char*>(&grid.spacing_y_mm), sizeof(grid.spacing_y_mm));
    input.read(reinterpret_cast<char*>(&grid.spacing_z_mm), sizeof(grid.spacing_z_mm));
    const auto count = grid.number_of_voxels();
    if (count == 0 || count > 512ULL * 1024ULL * 1024ULL) {
        throw std::runtime_error("CT grid size is invalid: " + path.string());
    }
    grid.density_g_per_cm3.resize(count);
    grid.material_id.resize(count);
    input.read(reinterpret_cast<char*>(grid.density_g_per_cm3.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
    input.read(reinterpret_cast<char*>(grid.material_id.data()),
               static_cast<std::streamsize>(count * sizeof(std::uint8_t)));
    if (version == version_legacy) {
        grid.mass_sp_za_rel = {1.0F, 1.0F, 1.0F, 1.0F};
        grid.mass_sp_I_eV = {75.0F, 75.0F, 75.0F, 75.0F};
        grid.mass_sp_factor = {1.0F, 1.0F, 1.0F, 1.0F};
    } else if (version == version_v2) {
        std::uint32_t n_factors = 0;
        input.read(reinterpret_cast<char*>(&n_factors), sizeof(n_factors));
        if (n_factors == 0 || n_factors > 256U) {
            throw std::runtime_error("Invalid CT mass-SP factor count: " + path.string());
        }
        grid.mass_sp_factor.resize(n_factors);
        input.read(reinterpret_cast<char*>(grid.mass_sp_factor.data()),
                   static_cast<std::streamsize>(n_factors * sizeof(float)));
        grid.mass_sp_za_rel = grid.mass_sp_factor;
        grid.mass_sp_I_eV.assign(n_factors, 75.0F);
    } else {
        std::uint32_t n_factors = 0;
        input.read(reinterpret_cast<char*>(&n_factors), sizeof(n_factors));
        if (n_factors == 0 || n_factors > 256U) {
            throw std::runtime_error("Invalid CT mass-SP section count: " + path.string());
        }
        grid.mass_sp_za_rel.resize(n_factors);
        grid.mass_sp_I_eV.resize(n_factors);
        input.read(reinterpret_cast<char*>(grid.mass_sp_za_rel.data()),
                   static_cast<std::streamsize>(n_factors * sizeof(float)));
        input.read(reinterpret_cast<char*>(grid.mass_sp_I_eV.data()),
                   static_cast<std::streamsize>(n_factors * sizeof(float)));
        grid.mass_sp_factor = grid.mass_sp_za_rel;
    }
    if (!input) {
        throw std::runtime_error("Truncated CT grid file: " + path.string());
    }
    return grid;
}

void CtGrid::write_binary(const std::filesystem::path& path) const {
    if (density_g_per_cm3.size() != number_of_voxels() ||
        material_id.size() != number_of_voxels()) {
        throw std::invalid_argument("CT grid array size mismatch");
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot write CT grid: " + path.string());
    }
    const auto magic = magic_value;
    const auto version = version_value;
    output.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    output.write(reinterpret_cast<const char*>(&version), sizeof(version));
    output.write(reinterpret_cast<const char*>(&nx), sizeof(nx));
    output.write(reinterpret_cast<const char*>(&ny), sizeof(ny));
    output.write(reinterpret_cast<const char*>(&nz), sizeof(nz));
    output.write(reinterpret_cast<const char*>(&origin_x_mm), sizeof(origin_x_mm));
    output.write(reinterpret_cast<const char*>(&origin_y_mm), sizeof(origin_y_mm));
    output.write(reinterpret_cast<const char*>(&origin_z_mm), sizeof(origin_z_mm));
    output.write(reinterpret_cast<const char*>(&spacing_x_mm), sizeof(spacing_x_mm));
    output.write(reinterpret_cast<const char*>(&spacing_y_mm), sizeof(spacing_y_mm));
    output.write(reinterpret_cast<const char*>(&spacing_z_mm), sizeof(spacing_z_mm));
    output.write(reinterpret_cast<const char*>(density_g_per_cm3.data()),
                 static_cast<std::streamsize>(density_g_per_cm3.size() * sizeof(float)));
    output.write(reinterpret_cast<const char*>(material_id.data()),
                 static_cast<std::streamsize>(material_id.size() * sizeof(std::uint8_t)));

    std::vector<float> za = mass_sp_za_rel;
    std::vector<float> I = mass_sp_I_eV;
    if (za.empty() && !mass_sp_factor.empty()) {
        za = mass_sp_factor;
        I.assign(za.size(), 75.0F);
    }
    if (za.empty()) {
        za = {1.0F};
        I = {75.0F};
    }
    if (I.size() != za.size()) {
        I.assign(za.size(), 75.0F);
    }
    const auto n_factors = static_cast<std::uint32_t>(za.size());
    output.write(reinterpret_cast<const char*>(&n_factors), sizeof(n_factors));
    output.write(reinterpret_cast<const char*>(za.data()),
                 static_cast<std::streamsize>(za.size() * sizeof(float)));
    output.write(reinterpret_cast<const char*>(I.data()),
                 static_cast<std::streamsize>(I.size() * sizeof(float)));
}

}  // namespace carbon

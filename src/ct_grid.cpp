#include "carbon/runtime_timing.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/transport_config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
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

namespace {

SchneiderElement element_from_name(const std::string& name) {
    if (name == "Hydrogen") return {1, 1.008, "Hydrogen"};
    if (name == "Carbon") return {6, 12.011, "Carbon"};
    if (name == "Nitrogen") return {7, 14.007, "Nitrogen"};
    if (name == "Oxygen") return {8, 15.999, "Oxygen"};
    if (name == "Magnesium") return {12, 24.305, "Magnesium"};
    if (name == "Phosphorus") return {15, 30.973762, "Phosphorus"};
    if (name == "Sulfur") return {16, 32.06, "Sulfur"};
    if (name == "Chlorine") return {17, 35.45, "Chlorine"};
    if (name == "Argon") return {18, 39.948, "Argon"};
    if (name == "Calcium") return {20, 40.078, "Calcium"};
    if (name == "Sodium") return {11, 22.989769, "Sodium"};
    if (name == "Potassium") return {19, 39.0983, "Potassium"};
    if (name == "Titanium") return {22, 47.867, "Titanium"};
    throw std::invalid_argument("Unsupported or unknown Schneider element name: " + name);
}

std::vector<std::string> parse_string_list(const std::string& text) {
    std::vector<std::string> tokens;
    std::size_t pos = 0;
    while (pos < text.size()) {
        const auto quote_start = text.find('"', pos);
        if (quote_start == std::string::npos) break;
        const auto quote_end = text.find('"', quote_start + 1);
        if (quote_end == std::string::npos) {
            throw std::invalid_argument("Unterminated quoted string in element list");
        }
        tokens.push_back(text.substr(quote_start + 1, quote_end - quote_start - 1));
        pos = quote_end + 1;
    }
    return tokens;
}

std::vector<double> parse_number_list(const std::string& text) {
    std::vector<double> values;
    std::stringstream input(text);
    std::string token;
    while (input >> token) {
        if (token == "g/cm3" || token.front() == '"' ||
            std::isalpha(static_cast<unsigned char>(token.front()))) {
            continue;
        }
        try {
            values.push_back(std::stod(token));
        } catch (const std::exception&) {
            continue;
        }
    }
    return values;
}

int exact_integer(double x, const std::string& field_name) {
    if (!std::isfinite(x) || std::floor(x) != x ||
        x < static_cast<double>(std::numeric_limits<int>::min()) ||
        x > static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Expected exact integer value for " + field_name +
                                    ", got: " + std::to_string(x));
    }
    return static_cast<int>(x);
}

int find_section(const std::vector<int>& edges, const float hu) noexcept {
    if (edges.size() < 2) {
        return 0;
    }
    for (std::size_t index = 0; index + 1 < edges.size(); ++index) {
        if (hu < static_cast<float>(edges[index + 1])) {
            return static_cast<int>(index);
        }
    }
    return static_cast<int>(edges.size() - 2);
}

}  // namespace

SchneiderMaterialTable SchneiderMaterialTable::from_topas_file(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open Schneider file: " + path.string());
    }

    SchneiderMaterialTable table;
    bool has_elements = false;
    bool has_sections = false;
    std::array<bool, section_count> has_weight_row{};
    has_weight_row.fill(false);

    std::string line;
    while (std::getline(input, line)) {
        const auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line.erase(comment_pos);
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, eq);
        const std::string payload = line.substr(eq + 1);
        const auto first_key_char = key.find_first_not_of(" \t\r\n");
        if (first_key_char == std::string::npos) {
            continue;
        }
        key.erase(0, first_key_char);
        if (const auto last = key.find_last_not_of(" \t\r\n"); last != std::string::npos) {
            key.erase(last + 1);
        }

        if (key.find("SchneiderElements") != std::string::npos) {
            if (has_elements) {
                throw std::invalid_argument("Duplicate SchneiderElements entry in " + path.string());
            }
            std::stringstream ss(payload);
            std::size_t count = 0;
            if (!(ss >> count)) {
                throw std::invalid_argument("Missing count in SchneiderElements: " + path.string());
            }
            if (count != element_count) {
                throw std::invalid_argument("SchneiderElements count mismatch (expected " +
                                            std::to_string(element_count) + ", got " +
                                            std::to_string(count) + "): " + path.string());
            }
            const auto elem_names = parse_string_list(payload);
            if (elem_names.size() != element_count) {
                throw std::invalid_argument("SchneiderElements parsed count mismatch (expected " +
                                            std::to_string(element_count) + ", got " +
                                            std::to_string(elem_names.size()) + "): " + path.string());
            }
            for (std::size_t i = 0; i < element_count; ++i) {
                if (elem_names[i] != schneider_canonical_element_names[i]) {
                    throw std::invalid_argument(
                        "Schneider element order mismatch at column " + std::to_string(i) +
                        " (expected " + std::string(schneider_canonical_element_names[i]) +
                        ", got " + elem_names[i] + "): " + path.string());
                }
                table.elements[i] = element_from_name(elem_names[i]);
            }
            has_elements = true;
        } else if (key.find("SchneiderHUToMaterialSections") != std::string::npos) {
            if (has_sections) {
                throw std::invalid_argument("Duplicate SchneiderHUToMaterialSections entry in " + path.string());
            }
            const auto numbers = parse_number_list(payload);
            if (numbers.size() < section_count + 2) {
                throw std::invalid_argument("SchneiderHUToMaterialSections incomplete in " + path.string());
            }
            const auto count = static_cast<std::size_t>(
                exact_integer(numbers[0], "SchneiderHUToMaterialSections count"));
            if (count != section_count + 1 || numbers.size() != count + 1) {
                throw std::invalid_argument("SchneiderHUToMaterialSections count mismatch in " + path.string());
            }
            std::vector<int> edges(count);
            for (std::size_t i = 0; i < count; ++i) {
                edges[i] = exact_integer(numbers[i + 1],
                                         "SchneiderHUToMaterialSections edge[" + std::to_string(i) + "]");
                if (i > 0 && edges[i] <= edges[i - 1]) {
                    throw std::invalid_argument("SchneiderHUToMaterialSections not strictly increasing at index " +
                                                std::to_string(i) + " in " + path.string());
                }
            }
            for (std::size_t i = 0; i < section_count; ++i) {
                table.sections[i].hu_min_inclusive = edges[i];
                table.sections[i].hu_max_exclusive = edges[i + 1];
            }
            has_sections = true;
        } else if (const auto w_pos = key.find("SchneiderMaterialsWeight"); w_pos != std::string::npos) {
            const auto num_str = key.substr(w_pos + std::string("SchneiderMaterialsWeight").length());
            int row_idx = 0;
            try {
                row_idx = std::stoi(num_str);
            } catch (const std::exception&) {
                throw std::invalid_argument("Malformed SchneiderMaterialsWeight row key: " + key);
            }
            if (row_idx < 1 || row_idx > static_cast<int>(section_count)) {
                throw std::invalid_argument("SchneiderMaterialsWeight row index out of range [1..25]: " +
                                            std::to_string(row_idx));
            }
            if (has_weight_row[static_cast<std::size_t>(row_idx - 1)]) {
                throw std::invalid_argument("Duplicate SchneiderMaterialsWeight" + std::to_string(row_idx) +
                                            " in " + path.string());
            }
            const auto numbers = parse_number_list(payload);
            if (numbers.size() != element_count + 1) {
                throw std::invalid_argument("SchneiderMaterialsWeight" + std::to_string(row_idx) +
                                            " count mismatch (expected " + std::to_string(element_count + 1) +
                                            ", got " + std::to_string(numbers.size()) + ") in " + path.string());
            }
            const auto declared_count = static_cast<std::size_t>(
                exact_integer(numbers[0], "SchneiderMaterialsWeight" + std::to_string(row_idx) + " declared count"));
            if (declared_count != element_count) {
                throw std::invalid_argument("SchneiderMaterialsWeight" + std::to_string(row_idx) +
                                            " declared count mismatch in " + path.string());
            }
            double row_sum = 0.0;
            for (std::size_t el = 0; el < element_count; ++el) {
                const double w = numbers[el + 1];
                if (!std::isfinite(w) || w < 0.0) {
                    throw std::invalid_argument("SchneiderMaterialsWeight" + std::to_string(row_idx) +
                                                " contains negative or non-finite weight: " + std::to_string(w));
                }
                table.sections[static_cast<std::size_t>(row_idx - 1)].mass_fraction[el] = w;
                row_sum += w;
            }
            if (std::abs(row_sum - 1.0) > 1.0e-6) {
                throw std::invalid_argument("SchneiderMaterialsWeight" + std::to_string(row_idx) +
                                            " sum diverges from 1.0 (sum=" + std::to_string(row_sum) +
                                            ") in " + path.string());
            }
            has_weight_row[static_cast<std::size_t>(row_idx - 1)] = true;
        }
    }

    if (!has_elements) {
        throw std::runtime_error("SchneiderElements missing in " + path.string());
    }
    if (!has_sections) {
        throw std::runtime_error("SchneiderHUToMaterialSections missing in " + path.string());
    }
    for (std::size_t i = 0; i < section_count; ++i) {
        if (!has_weight_row[i]) {
            throw std::runtime_error("SchneiderMaterialsWeight" + std::to_string(i + 1) +
                                     " missing in " + path.string());
        }
    }

    return table;
}

SchneiderMaterialTable SchneiderMaterialTable::builtin() {
    const std::array<std::filesystem::path, 3> candidates = {
        std::filesystem::path("data/HUtoMaterialSchneider.txt"),
        std::filesystem::path("../data/HUtoMaterialSchneider.txt"),
        std::filesystem::path("../../data/HUtoMaterialSchneider.txt")
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return from_topas_file(candidate);
        }
    }
    throw std::runtime_error("Cannot locate built-in data/HUtoMaterialSchneider.txt");
}

std::uint8_t SchneiderMaterialTable::section_id(const int hu) const noexcept {
    if (hu < sections[0].hu_min_inclusive) {
        return 0;
    }
    for (std::size_t i = 0; i < section_count; ++i) {
        if (hu >= sections[i].hu_min_inclusive && hu < sections[i].hu_max_exclusive) {
            return static_cast<std::uint8_t>(i);
        }
    }
    return static_cast<std::uint8_t>(section_count - 1);
}

SchneiderHuTable SchneiderHuTable::builtin() {
    SchneiderHuTable table;
    table.density_hu_edges = {-1000, -98, 15, 23, 101, 2001, 2995, 2996};
    table.density_offset = {0.00121, 1.018, 1.03, 1.003, 1.017, 2.201, 4.54};
    table.density_factor = {0.001029700665188, 0.000893, 0.0, 0.001169, 0.000592,
                            0.0005, 0.0};
    table.density_factor_offset = {1000.0, 0.0, 1000.0, 0.0, 0.0, -2000.0, 0.0};
    table.density_correction.assign(3996, 1.0);
    table.density_correction_hu0 = -1000;
    table.material_hu_edges = {-1000, -950, -120, -83, -53, -23, 7, 18, 80, 120,
                               200,   300,  400,  500, 600, 700, 800, 900, 1000,
                               1100,  1200, 1300, 1400, 1500, 2995, 2996};
    table.za_rel = {
        0.8992799520492554F, 0.9917084574699402F, 1.0029765367507935F,
        1.0004115104675293F, 0.9977754354476929F, 0.9961254596710205F,
        0.9943869709968567F, 0.9917228817939758F, 0.9835737347602844F,
        0.9838401675224304F, 0.9782578945159912F, 0.9717891216278076F,
        0.9662479758262634F, 0.9616461992263794F, 0.9570088982582092F,
        0.9523995518684387F, 0.9477784633636475F, 0.9440864920616150F,
        0.9412810206413269F, 0.9375873804092407F, 0.9348113536834717F,
        0.9320344328880310F, 0.9292566776275635F, 0.9273962974548340F,
        0.8279811739921570F};
    table.I_eV = {85.6829605102539F,  69.6933822631836F,  62.00094985961914F,
                  63.43705368041992F, 64.97969818115234F, 66.1916732788086F,
                  67.23116302490234F, 69.33441925048828F, 70.08238220214844F,
                  70.11997985839844F, 72.77925872802734F, 75.77842712402344F,
                  78.65174102783203F, 81.14991760253906F, 83.7420654296875F,
                  86.30131530761719F, 89.0001220703125F,  91.22447967529297F,
                  93.24040985107422F, 95.57535552978516F, 97.48521423339844F,
                  99.42965698242188F, 101.4092025756836F, 102.7972640991211F,
                  233.0F};
    return table;
}

SchneiderHuTable SchneiderHuTable::from_topas_file(const std::filesystem::path& path) {
    auto table = builtin();
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open Schneider HU table: " + path.string());
    }
    std::string line;
    while (std::getline(input, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const auto key = line.substr(0, eq);
        const auto payload = line.substr(eq + 1);
        if (key.find("SchneiderElements") != std::string::npos ||
            key.find("SchneiderMaterialsWeight") != std::string::npos ||
            key.find("ImagingToMaterialConverter") != std::string::npos ||
            key.find("Color/PatientTissue") != std::string::npos) {
            continue;
        }
        auto numbers = parse_number_list(payload);
        if (numbers.empty()) {
            continue;
        }
        if (key.find("DensityCorrection") != std::string::npos) {
            const auto count = static_cast<std::size_t>(numbers.front());
            numbers.erase(numbers.begin());
            if (numbers.size() != count) {
                throw std::runtime_error("Schneider DensityCorrection count mismatch");
            }
            table.density_correction = std::move(numbers);
        } else if (key.find("SchneiderHounsfieldUnitSections") != std::string::npos) {
            numbers.erase(numbers.begin());
            table.density_hu_edges.assign(numbers.begin(), numbers.end());
        } else if (key.find("SchneiderDensityOffset") != std::string::npos) {
            numbers.erase(numbers.begin());
            table.density_offset = std::move(numbers);
        } else if (key.find("SchneiderDensityFactorOffset") != std::string::npos) {
            numbers.erase(numbers.begin());
            table.density_factor_offset = std::move(numbers);
        } else if (key.find("SchneiderDensityFactor") != std::string::npos) {
            numbers.erase(numbers.begin());
            table.density_factor = std::move(numbers);
        } else if (key.find("SchneiderHUToMaterialSections") != std::string::npos) {
            numbers.erase(numbers.begin());
            table.material_hu_edges.assign(numbers.begin(), numbers.end());
        }
    }
    if (table.density_hu_edges.size() < 2 || table.material_hu_edges.size() < 2) {
        throw std::runtime_error("Incomplete Schneider HU table: " + path.string());
    }
    const auto n_materials = table.material_hu_edges.size() - 1;
    if (table.za_rel.size() != n_materials) {
        table.za_rel.assign(n_materials, 1.0F);
        table.I_eV.assign(n_materials, 75.0F);
    }
    return table;
}

float SchneiderHuTable::density_g_per_cm3(const float hu) const noexcept {
    auto value = hu;
    const auto hu_min = static_cast<float>(density_correction_hu0);
    const auto hu_max =
        hu_min + static_cast<float>(density_correction.empty()
                                        ? 0
                                        : density_correction.size() - 1);
    if (value < hu_min) {
        value = hu_min;
    }
    if (value > hu_max && hu_max > hu_min) {
        value = hu_max;
    }
    const auto section = find_section(density_hu_edges, value);
    const auto offset = section < static_cast<int>(density_offset.size())
                            ? density_offset[static_cast<std::size_t>(section)]
                            : 1.0;
    const auto factor = section < static_cast<int>(density_factor.size())
                            ? density_factor[static_cast<std::size_t>(section)]
                            : 0.0;
    const auto factor_offset =
        section < static_cast<int>(density_factor_offset.size())
            ? density_factor_offset[static_cast<std::size_t>(section)]
            : 0.0;
    auto density = offset + factor * (factor_offset + static_cast<double>(value));
    if (!density_correction.empty()) {
        auto index = static_cast<int>(std::lround(value)) - density_correction_hu0;
        if (index < 0) {
            index = 0;
        }
        if (index >= static_cast<int>(density_correction.size())) {
            index = static_cast<int>(density_correction.size()) - 1;
        }
        density *= density_correction[static_cast<std::size_t>(index)];
    }
    return static_cast<float>(density > 1.0e-6 ? density : 1.0e-6);
}

std::uint8_t SchneiderHuTable::section_id(const float hu) const noexcept {
    const auto section = find_section(material_hu_edges, hu);
    if (section < 0) {
        return 0;
    }
    if (section > 255) {
        return 255;
    }
    return static_cast<std::uint8_t>(section);
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

CtGrid CtGrid::from_config(const TransportConfig& config) {
    RuntimeScope runtime_ct("ct_from_config_including_load");
    return load(config.ct_grid_file, config.ct_schneider_file,
                config.ct_dicom_origin_mode);
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

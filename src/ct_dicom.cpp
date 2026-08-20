#include "carbon/ct_grid.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace carbon {
namespace {

constexpr std::uint16_t k_file_meta_group = 0x0002;
constexpr std::uint32_t k_undefined_length = 0xFFFFFFFFU;

struct DicomTag {
    std::uint16_t group{0};
    std::uint16_t element{0};

    [[nodiscard]] std::uint32_t key() const noexcept {
        return (static_cast<std::uint32_t>(group) << 16U) | element;
    }
};

struct DicomElement {
    DicomTag tag{};
    std::string vr{};
    std::vector<std::uint8_t> value{};
};

class BinaryReader {
public:
    explicit BinaryReader(std::vector<std::uint8_t> bytes) : bytes_(std::move(bytes)) {}

    [[nodiscard]] std::size_t size() const noexcept { return bytes_.size(); }
    [[nodiscard]] std::size_t offset() const noexcept { return offset_; }
    [[nodiscard]] bool remaining(const std::size_t count) const noexcept {
        return offset_ + count <= bytes_.size();
    }

    void seek(const std::size_t position) {
        if (position > bytes_.size()) {
            throw std::runtime_error("DICOM seek past end of file");
        }
        offset_ = position;
    }

    template <typename T>
    T read_le() {
        if (!remaining(sizeof(T))) {
            throw std::runtime_error("Truncated DICOM value");
        }
        T value{};
        std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    std::vector<std::uint8_t> read_bytes(const std::size_t count) {
        if (!remaining(count)) {
            throw std::runtime_error("Truncated DICOM byte run");
        }
        std::vector<std::uint8_t> out(bytes_.data() + offset_,
                                      bytes_.data() + offset_ + count);
        offset_ += count;
        return out;
    }

    std::string read_string(const std::size_t count) {
        const auto bytes = read_bytes(count);
        std::string text(bytes.begin(), bytes.end());
        while (!text.empty() && (text.back() == ' ' || text.back() == '\0')) {
            text.pop_back();
        }
        return text;
    }

private:
    std::vector<std::uint8_t> bytes_{};
    std::size_t offset_{0};
};

const std::unordered_map<std::uint32_t, const char*>& implicit_vr_map() {
    static const std::unordered_map<std::uint32_t, const char*> table = {
        {0x00080016U, "UI"}, {0x00080018U, "UI"}, {0x00080060U, "CS"},
        {0x00180050U, "DS"}, {0x0020000DU, "UI"}, {0x0020000EU, "UI"},
        {0x00200032U, "DS"}, {0x00200037U, "DS"}, {0x00280010U, "US"},
        {0x00280011U, "US"}, {0x00280030U, "DS"}, {0x00280100U, "US"},
        {0x00280101U, "US"}, {0x00280103U, "US"}, {0x00281052U, "DS"},
        {0x00281053U, "DS"}, {0x7FE00010U, "OW"},
    };
    return table;
}

bool even_vr(const std::string& vr) {
    return vr == "OB" || vr == "OD" || vr == "OF" || vr == "OL" || vr == "OW" ||
           vr == "OV" || vr == "SQ" || vr == "SV" || vr == "UC" || vr == "UN" ||
           vr == "UR" || vr == "UT" || vr == "UV";
}

DicomElement read_element(BinaryReader& reader, const bool explicit_vr) {
    DicomElement element;
    element.tag.group = reader.read_le<std::uint16_t>();
    element.tag.element = reader.read_le<std::uint16_t>();
    if (element.tag.group == 0xFFFE) {
        const auto length = reader.read_le<std::uint32_t>();
        element.vr = "DL";
        if (length != k_undefined_length && length > 0) {
            element.value = reader.read_bytes(length);
        }
        return element;
    }

    std::uint32_t length = 0;
    if (explicit_vr) {
        char vr_bytes[3] = {};
        vr_bytes[0] = static_cast<char>(reader.read_le<std::uint8_t>());
        vr_bytes[1] = static_cast<char>(reader.read_le<std::uint8_t>());
        element.vr = vr_bytes;
        if (even_vr(element.vr)) {
            reader.read_le<std::uint16_t>();
            length = reader.read_le<std::uint32_t>();
        } else {
            length = reader.read_le<std::uint16_t>();
        }
    } else {
        const auto found = implicit_vr_map().find(element.tag.key());
        element.vr = found == implicit_vr_map().end() ? "UN" : found->second;
        length = reader.read_le<std::uint32_t>();
    }
    if (length == k_undefined_length) {
        if (element.vr == "SQ" || element.tag.key() == 0x7FE00010U) {
            throw std::runtime_error(
                "Undefined-length DICOM sequences/pixel data are not supported");
        }
        throw std::runtime_error("Undefined-length DICOM element");
    }
    if (length > 0) {
        element.value = reader.read_bytes(length);
    }
    return element;
}

std::string as_string(const DicomElement& element) {
    std::string text(element.value.begin(), element.value.end());
    while (!text.empty() && (text.back() == ' ' || text.back() == '\0')) {
        text.pop_back();
    }
    return text;
}

std::vector<double> as_ds(const DicomElement& element) {
    std::vector<double> values;
    std::stringstream input(as_string(element));
    std::string token;
    while (std::getline(input, token, '\\')) {
        if (token.empty()) {
            continue;
        }
        values.push_back(std::stod(token));
    }
    return values;
}

std::uint16_t as_us(const DicomElement& element) {
    if (element.value.size() < 2) {
        throw std::runtime_error("Expected US DICOM value");
    }
    std::uint16_t value = 0;
    std::memcpy(&value, element.value.data(), sizeof(value));
    return value;
}

struct DicomSlice {
    std::string series_uid;
    std::string modality;
    std::array<double, 3> ipp{0.0, 0.0, 0.0};
    std::array<double, 6> iop{1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    std::uint16_t rows{0};
    std::uint16_t cols{0};
    double spacing_x{0.0};
    double spacing_y{0.0};
    double slope{1.0};
    double intercept{0.0};
    std::uint16_t bits_allocated{16};
    std::uint16_t pixel_representation{1};
    std::vector<std::uint8_t> pixel_bytes;
};

bool is_axial_lps(const std::array<double, 6>& iop) {
    const auto close = [](const double value, const double expected) {
        return std::abs(value - expected) < 1.0e-3;
    };
    return close(iop[0], 1.0) && close(iop[1], 0.0) && close(iop[2], 0.0) &&
           close(iop[3], 0.0) && close(iop[4], 1.0) && close(iop[5], 0.0);
}

DicomSlice parse_dicom_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open DICOM file: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(input.tellg());
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(size);
    input.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(size));
    if (size < 132 || std::memcmp(bytes.data() + 128, "DICM", 4) != 0) {
        throw std::runtime_error("Not a DICOM Part 10 file: " + path.string());
    }

    BinaryReader reader(std::move(bytes));
    reader.seek(132);
    std::string transfer_syntax = "1.2.840.10008.1.2.1";
    while (reader.remaining(8)) {
        const auto mark = reader.offset();
        const auto group = reader.read_le<std::uint16_t>();
        reader.seek(mark);
        if (group != k_file_meta_group) {
            break;
        }
        const auto element = read_element(reader, true);
        if (element.tag.group == 0x0002 && element.tag.element == 0x0010) {
            transfer_syntax = as_string(element);
        }
    }
    if (transfer_syntax != "1.2.840.10008.1.2" &&
        transfer_syntax != "1.2.840.10008.1.2.1") {
        throw std::runtime_error("Unsupported DICOM transfer syntax " +
                                 transfer_syntax + " in " + path.string());
    }
    const auto explicit_vr = transfer_syntax != "1.2.840.10008.1.2";

    DicomSlice slice;
    while (reader.remaining(8)) {
        const auto element = read_element(reader, explicit_vr);
        switch (element.tag.key()) {
            case 0x00080060U:
                slice.modality = as_string(element);
                break;
            case 0x0020000EU:
                slice.series_uid = as_string(element);
                break;
            case 0x00200032U: {
                const auto values = as_ds(element);
                if (values.size() == 3) {
                    slice.ipp = {values[0], values[1], values[2]};
                }
                break;
            }
            case 0x00200037U: {
                const auto values = as_ds(element);
                if (values.size() == 6) {
                    slice.iop = {values[0], values[1], values[2],
                                 values[3], values[4], values[5]};
                }
                break;
            }
            case 0x00280010U:
                slice.rows = as_us(element);
                break;
            case 0x00280011U:
                slice.cols = as_us(element);
                break;
            case 0x00280030U: {
                const auto values = as_ds(element);
                if (values.size() >= 2) {
                    slice.spacing_y = values[0];
                    slice.spacing_x = values[1];
                }
                break;
            }
            case 0x00280100U:
                slice.bits_allocated = as_us(element);
                break;
            case 0x00280103U:
                slice.pixel_representation = as_us(element);
                break;
            case 0x00281052U: {
                const auto values = as_ds(element);
                if (!values.empty()) {
                    slice.intercept = values[0];
                }
                break;
            }
            case 0x00281053U: {
                const auto values = as_ds(element);
                if (!values.empty()) {
                    slice.slope = values[0];
                }
                break;
            }
            case 0x7FE00010U:
                slice.pixel_bytes = element.value;
                break;
            default:
                break;
        }
    }
    return slice;
}

float hu_from_pixel(const DicomSlice& slice, const std::size_t index) {
    if (slice.bits_allocated != 16 ||
        slice.pixel_bytes.size() < (index + 1U) * 2U) {
        throw std::runtime_error("DICOM pixel data is not 16-bit");
    }
    std::uint16_t raw = 0;
    std::memcpy(&raw, slice.pixel_bytes.data() + index * 2U, sizeof(raw));
    const auto stored = slice.pixel_representation == 1
                            ? static_cast<double>(static_cast<std::int16_t>(raw))
                            : static_cast<double>(raw);
    return static_cast<float>(slice.slope * stored + slice.intercept);
}

std::filesystem::path default_schneider_file() {
    return "data/HUtoMaterialSchneider.txt";
}

}  // namespace

CtGrid CtGrid::from_dicom_directory(const std::filesystem::path& directory,
                                    const std::filesystem::path& schneider_file,
                                    const std::string_view origin_mode) {
    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error("CT DICOM path is not a directory: " +
                                 directory.string());
    }
    std::vector<std::filesystem::path> candidates;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto ext = entry.path().extension().string();
        auto lower = ext;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (lower == ".dcm" || lower == ".ima" || lower.empty() || lower == ".dicom") {
            candidates.push_back(entry.path());
        }
    }
    if (candidates.empty()) {
        throw std::runtime_error("No DICOM files in " + directory.string());
    }

    std::unordered_map<std::string, std::vector<DicomSlice>> series;
    for (const auto& path : candidates) {
        try {
            auto slice = parse_dicom_file(path);
            if (slice.modality != "CT" || slice.rows == 0 || slice.cols == 0 ||
                slice.pixel_bytes.empty()) {
                continue;
            }
            if (!is_axial_lps(slice.iop)) {
                throw std::runtime_error(
                    "Only axial HFS DICOM (IOP 1\\0\\0\\0\\1\\0) is supported: " +
                    path.string());
            }
            series[slice.series_uid.empty() ? path.parent_path().string()
                                            : slice.series_uid]
                .push_back(std::move(slice));
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            if (message.find("Not a DICOM") != std::string::npos) {
                continue;
            }
            throw;
        }
    }
    if (series.empty()) {
        throw std::runtime_error("No CT Image slices in " + directory.string());
    }
    auto* chosen = &series.begin()->second;
    for (auto& [uid, slices] : series) {
        if (slices.size() > chosen->size()) {
            chosen = &slices;
        }
    }
    auto& slices = *chosen;
    std::sort(slices.begin(), slices.end(), [](const DicomSlice& a, const DicomSlice& b) {
        return a.ipp[2] < b.ipp[2];
    });

    const auto& first = slices.front();
    for (const auto& slice : slices) {
        if (slice.rows != first.rows || slice.cols != first.cols ||
            std::abs(slice.spacing_x - first.spacing_x) > 1.0e-4 ||
            std::abs(slice.spacing_y - first.spacing_y) > 1.0e-4) {
            throw std::runtime_error("Inconsistent DICOM slice geometry in " +
                                     directory.string());
        }
    }
    double spacing_z = first.spacing_y;
    if (slices.size() > 1) {
        spacing_z = slices[1].ipp[2] - slices[0].ipp[2];
        if (!(spacing_z > 0.0)) {
            throw std::runtime_error("Non-increasing DICOM slice positions in " +
                                     directory.string());
        }
        for (std::size_t index = 1; index < slices.size(); ++index) {
            const auto dz = slices[index].ipp[2] - slices[index - 1].ipp[2];
            if (std::abs(dz - spacing_z) > 1.0e-2) {
                throw std::runtime_error("Uneven DICOM slice spacing in " +
                                         directory.string());
            }
        }
    }

    auto table = SchneiderHuTable::builtin();
    std::filesystem::path table_path = schneider_file;
    if (table_path.empty()) {
        table_path = default_schneider_file();
    }
    if (!table_path.empty() && std::filesystem::is_regular_file(table_path)) {
        table = SchneiderHuTable::from_topas_file(table_path);
    } else if (!schneider_file.empty()) {
        throw std::runtime_error("Cannot open Schneider HU table: " +
                                 schneider_file.string());
    }

    CtGrid grid;
    grid.file_version = version_value;
    grid.nx = first.cols;
    grid.ny = first.rows;
    grid.nz = static_cast<std::uint32_t>(slices.size());
    grid.spacing_x_mm = static_cast<float>(first.spacing_x);
    grid.spacing_y_mm = static_cast<float>(first.spacing_y);
    grid.spacing_z_mm = static_cast<float>(spacing_z);
    auto mode = std::string(origin_mode);
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (mode == "dicom") {
        grid.origin_x_mm =
            static_cast<float>(first.ipp[0] - 0.5 * first.spacing_x);
        grid.origin_y_mm =
            static_cast<float>(first.ipp[1] - 0.5 * first.spacing_y);
        grid.origin_z_mm = static_cast<float>(first.ipp[2] - 0.5 * spacing_z);
    } else if (mode == "centered") {
        grid.origin_x_mm =
            -0.5F * static_cast<float>(grid.nx) * grid.spacing_x_mm;
        grid.origin_y_mm =
            -0.5F * static_cast<float>(grid.ny) * grid.spacing_y_mm;
        grid.origin_z_mm = -0.5F * grid.spacing_z_mm;
    } else {
        throw std::invalid_argument(
            "ct_dicom_origin_mode must be centered or dicom");
    }

    const auto count = grid.number_of_voxels();
    grid.density_g_per_cm3.resize(count);
    grid.material_id.resize(count);
    for (std::uint32_t iz = 0; iz < grid.nz; ++iz) {
        const auto& slice = slices[iz];
        for (std::uint32_t iy = 0; iy < grid.ny; ++iy) {
            for (std::uint32_t ix = 0; ix < grid.nx; ++ix) {
                const auto pixel = static_cast<std::size_t>(iy) * grid.nx + ix;
                const auto dest = ct_linear_index(ix, iy, iz, grid.nx, grid.ny);
                const auto hu = hu_from_pixel(slice, pixel);
                grid.density_g_per_cm3[dest] = table.density_g_per_cm3(hu);
                grid.material_id[dest] = table.section_id(hu);
            }
        }
    }
    grid.mass_sp_za_rel = table.za_rel;
    grid.mass_sp_I_eV = table.I_eV;
    grid.mass_sp_factor = table.za_rel;
    return grid;
}

CtGrid CtGrid::load(const std::filesystem::path& path,
                    const std::filesystem::path& schneider_file,
                    const std::string_view origin_mode) {
    if (std::filesystem::is_directory(path)) {
        return from_dicom_directory(path, schneider_file, origin_mode);
    }
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("CT grid path does not exist: " + path.string());
    }
    const auto ext = path.extension().string();
    auto lower = ext;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lower == ".dcm" || lower == ".ima" || lower == ".dicom") {
        return from_dicom_directory(path.parent_path(), schneider_file, origin_mode);
    }
    std::ifstream peek(path, std::ios::binary);
    std::uint32_t magic = 0;
    peek.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (peek && magic == magic_value) {
        return from_binary(path);
    }
    return from_dicom_directory(path.parent_path(), schneider_file, origin_mode);
}

}  // namespace carbon

#include "carbon/tps_source.hpp"

#include "carbon/ct_grid.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace carbon {
namespace {

constexpr double k_pi = 3.141592653589793238462643383279502884;

struct Vec3 {
    double x{0.0}, y{0.0}, z{0.0};
};

struct Matrix3 {
    double m[3][3]{};
};

std::string trim(std::string value) {
    const auto not_space = [](const unsigned char character) {
        return !std::isspace(character);
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string normalized_header(std::string value) {
    value = trim(std::move(value));
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::vector<std::string> split_csv_line(const std::string& line) {
    // TPS spot tables are deliberately numeric and do not need quoted commas.
    std::vector<std::string> fields;
    std::stringstream input(line);
    std::string field;
    while (std::getline(input, field, ',')) {
        fields.push_back(trim(field));
    }
    if (!line.empty() && line.back() == ',') {
        fields.emplace_back();
    }
    return fields;
}

double parse_required(const std::vector<std::string>& fields,
                      const std::unordered_map<std::string, std::size_t>& columns,
                      const std::string& name,
                      const std::filesystem::path& path,
                      const std::size_t line_number) {
    const auto iterator = columns.find(name);
    if (iterator == columns.end() || iterator->second >= fields.size() ||
        fields[iterator->second].empty()) {
        throw std::runtime_error("TPS CSV missing '" + name + "' at " + path.string() +
                                 ":" + std::to_string(line_number));
    }
    std::size_t parsed = 0;
    const auto value = std::stod(fields[iterator->second], &parsed);
    if (parsed != fields[iterator->second].size() || !std::isfinite(value)) {
        throw std::runtime_error("Invalid TPS CSV number for '" + name + "' at " +
                                 path.string() + ":" + std::to_string(line_number));
    }
    return value;
}

double parse_optional(const std::vector<std::string>& fields,
                      const std::unordered_map<std::string, std::size_t>& columns,
                      const std::string& name) {
    const auto iterator = columns.find(name);
    if (iterator == columns.end() || iterator->second >= fields.size() ||
        fields[iterator->second].empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::size_t parsed = 0;
    const auto& token = fields[iterator->second];
    const auto value = std::stod(token, &parsed);
    if (parsed != token.size() || !std::isfinite(value)) {
        throw std::runtime_error("Invalid optional TPS CSV number for '" + name + "'");
    }
    return value;
}

Vec3 operator+(const Vec3& left, const Vec3& right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

Vec3 operator-(const Vec3& left, const Vec3& right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

Vec3 operator*(const double scale, const Vec3& value) {
    return {scale * value.x, scale * value.y, scale * value.z};
}

Vec3 multiply(const Matrix3& matrix, const Vec3& value) {
    return {
        matrix.m[0][0] * value.x + matrix.m[0][1] * value.y +
            matrix.m[0][2] * value.z,
        matrix.m[1][0] * value.x + matrix.m[1][1] * value.y +
            matrix.m[1][2] * value.z,
        matrix.m[2][0] * value.x + matrix.m[2][1] * value.y +
            matrix.m[2][2] * value.z,
    };
}

Matrix3 multiply(const Matrix3& left, const Matrix3& right) {
    Matrix3 result{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            for (int inner = 0; inner < 3; ++inner) {
                result.m[row][column] += left.m[row][inner] * right.m[inner][column];
            }
        }
    }
    return result;
}

Matrix3 rotation_y(const double angle_deg) {
    const auto angle = angle_deg * k_pi / 180.0;
    const auto c = std::cos(angle);
    const auto s = std::sin(angle);
    return {{{c, 0.0, s}, {0.0, 1.0, 0.0}, {-s, 0.0, c}}};
}

Matrix3 rotation_z(const double angle_deg) {
    const auto angle = angle_deg * k_pi / 180.0;
    const auto c = std::cos(angle);
    const auto s = std::sin(angle);
    return {{{c, -s, 0.0}, {s, c, 0.0}, {0.0, 0.0, 1.0}}};
}

Matrix3 topas_patient_rot_z_frame(const double gantry_angle_deg,
                                  const double couch_angle_deg,
                                  const double collimator_angle_deg) {
    // Beam-local axes at TOPAS/TPS 0 degrees:
    //   u=patient +X (TOPAS TransX), v=patient +Z (TOPAS TransZ),
    //   w=patient +Y (central propagation direction).
    // A passive Patient/RotZ=theta is equivalent, in a fixed patient CT, to
    // actively rotating the source frame by +theta around patient +Z.
    const auto collimator = collimator_angle_deg * k_pi / 180.0;
    const auto cc = std::cos(collimator);
    const auto sc = std::sin(collimator);
    const Vec3 u0{cc, 0.0, -sc};
    const Vec3 v0{sc, 0.0, cc};
    const Vec3 w0{0.0, 1.0, 0.0};
    const auto patient_z_rotation =
        rotation_z(gantry_angle_deg + couch_angle_deg);
    const auto u = multiply(patient_z_rotation, u0);
    const auto v = multiply(patient_z_rotation, v0);
    const auto w = multiply(patient_z_rotation, w0);
    // central_pose applies the matrix to local propagation (0,0,-1), so the
    // third column is -w.
    return {{{u.x, v.x, -w.x}, {u.y, v.y, -w.y}, {u.z, v.z, -w.z}}};
}

Matrix3 orientation_matrix(const std::string& patient_position) {
    if (patient_position == "HFS") {
        return {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
    }
    if (patient_position == "HFP") {
        return {{{-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 1.0}}};
    }
    if (patient_position == "FFS") {
        return {{{-1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, -1.0}}};
    }
    if (patient_position == "FFP") {
        return {{{1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, -1.0}}};
    }
    throw std::invalid_argument("tps_patient_position must be HFS, HFP, FFS, or FFP");
}

TpsSourcePose central_pose(const TransportConfig& config, const TpsSpot& spot) {
    // The historical IEC path is unchanged. The TOPAS path uses the same
    // Patient/RotZ convention as the validated CT plans, but rotates the beam
    // in a fixed patient CT instead of rotating/repacking the CT.
    Matrix3 machine{};
    const auto gantry_angle_deg = std::isfinite(spot.gantry_angle_deg)
                                      ? spot.gantry_angle_deg
                                      : config.tps_gantry_angle_deg;
    const auto couch_angle_deg = std::isfinite(spot.couch_angle_deg)
                                     ? spot.couch_angle_deg
                                     : config.tps_couch_angle_deg;
    const auto collimator_angle_deg =
        std::isfinite(spot.collimator_angle_deg)
            ? spot.collimator_angle_deg
            : config.tps_collimator_angle_deg;
    if (config.tps_angle_convention == "topas_patient_rot_z") {
        machine = topas_patient_rot_z_frame(
            gantry_angle_deg, couch_angle_deg, collimator_angle_deg);
    } else {
        const auto collimator = rotation_z(-collimator_angle_deg);
        const auto gantry = rotation_y(gantry_angle_deg);
        const auto couch = rotation_z(couch_angle_deg);
        machine = multiply(couch, multiply(gantry, collimator));
    }
    const auto patient = orientation_matrix(config.tps_patient_position);
    const auto rotation = multiply(patient, machine);
    const auto u = multiply(rotation, Vec3{1.0, 0.0, 0.0});
    const auto v = multiply(rotation, Vec3{0.0, 1.0, 0.0});
    const auto w = multiply(rotation, Vec3{0.0, 0.0, -1.0});
    const Vec3 isocenter{config.tps_isocenter_x_mm, config.tps_isocenter_y_mm,
                         config.tps_isocenter_z_mm};
    const auto central_source = isocenter - config.tps_sad_mm * w;
    const auto origin = central_source + spot.x_mm * u + spot.y_mm * v;
    return {
        origin.x, origin.y, origin.z,
        u.x, u.y, u.z,
        v.x, v.y, v.z,
        w.x, w.y, w.z,
    };
}

double inherited(const double value, const double fallback) {
    return std::isfinite(value) ? value : fallback;
}

void validate_source_parameters(const double energy_spread_percent,
                                const double sigma_x_mm,
                                const double sigma_y_mm,
                                const double sigma_x_prime,
                                const double sigma_y_prime,
                                const double correlation_x,
                                const double correlation_y,
                                const std::string& context) {
    if (energy_spread_percent < 0.0 || energy_spread_percent > 20.0) {
        throw std::invalid_argument(
            context + ": energy_spread_percent must be in [0, 20]");
    }
    if (sigma_x_mm < 0.0 || sigma_y_mm < 0.0 || sigma_x_prime < 0.0 ||
        sigma_y_prime < 0.0) {
        throw std::invalid_argument(context + ": emittance sigmas must be nonnegative");
    }
    if (correlation_x < -1.0 || correlation_x > 1.0 || correlation_y < -1.0 ||
        correlation_y > 1.0) {
        throw std::invalid_argument(
            context + ": emittance correlations must be in [-1, 1]");
    }
}

}  // namespace

TpsSourcePlan TpsSourcePlan::from_csv(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open TPS spots CSV: " + path.string());
    }
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        if (!trim(line).empty()) {
            break;
        }
    }
    if (trim(line).empty()) {
        throw std::runtime_error("TPS spots CSV is empty: " + path.string());
    }
    const auto header = split_csv_line(line);
    std::unordered_map<std::string, std::size_t> columns;
    for (std::size_t index = 0; index < header.size(); ++index) {
        columns[normalized_header(header[index])] = index;
    }
    for (const auto* required : {"energy_mevu", "x_mm", "y_mm", "mu_weight"}) {
        if (!columns.contains(required)) {
            throw std::runtime_error("TPS spots CSV missing required column '" +
                                     std::string(required) + "': " + path.string());
        }
    }

    TpsSourcePlan plan;
    while (std::getline(input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        if (trim(line).empty()) {
            continue;
        }
        const auto fields = split_csv_line(line);
        TpsSpot spot;
        const auto id = parse_optional(fields, columns, "spot_id");
        spot.spot_id = std::isfinite(id) ? static_cast<int>(std::llround(id))
                                        : static_cast<int>(plan.spots.size() + 1);
        spot.energy_MeVu = parse_required(fields, columns, "energy_mevu", path, line_number);
        spot.x_mm = parse_required(fields, columns, "x_mm", path, line_number);
        spot.y_mm = parse_required(fields, columns, "y_mm", path, line_number);
        spot.mu_weight = parse_required(fields, columns, "mu_weight", path, line_number);
        spot.energy_spread_percent =
            parse_optional(fields, columns, "energy_spread_percent");
        spot.sigma_x_mm = parse_optional(fields, columns, "sigma_x_mm");
        spot.sigma_y_mm = parse_optional(fields, columns, "sigma_y_mm");
        spot.sigma_x_prime = parse_optional(fields, columns, "sigma_x_prime");
        spot.sigma_y_prime = parse_optional(fields, columns, "sigma_y_prime");
        spot.correlation_x = parse_optional(fields, columns, "correlation_x");
        spot.correlation_y = parse_optional(fields, columns, "correlation_y");
        spot.gantry_angle_deg =
            parse_optional(fields, columns, "gantry_angle_deg");
        spot.couch_angle_deg =
            parse_optional(fields, columns, "couch_angle_deg");
        spot.collimator_angle_deg =
            parse_optional(fields, columns, "collimator_angle_deg");
        if (!(spot.energy_MeVu > 0.0) || spot.mu_weight < 0.0) {
            throw std::runtime_error(
                "TPS spot energy must be positive and MU nonnegative at " +
                path.string() + ":" + std::to_string(line_number));
        }
        const auto context = "TPS spot at " + path.string() + ":" +
                             std::to_string(line_number);
        validate_source_parameters(
            inherited(spot.energy_spread_percent, 0.0),
            inherited(spot.sigma_x_mm, 0.0), inherited(spot.sigma_y_mm, 0.0),
            inherited(spot.sigma_x_prime, 0.0),
            inherited(spot.sigma_y_prime, 0.0),
            inherited(spot.correlation_x, 0.0),
            inherited(spot.correlation_y, 0.0), context);
        plan.total_mu += spot.mu_weight;
        plan.spots.push_back(spot);
    }
    if (plan.spots.empty() || !(plan.total_mu > 0.0)) {
        throw std::runtime_error("TPS spots CSV must contain a positive-MU spot: " +
                                 path.string());
    }
    return plan;
}

TpsSourcePlan TpsSourcePlan::from_config(const TransportConfig& config) {
    if (!config.tps_spots_file.empty()) {
        return from_csv(config.tps_spots_file);
    }
    TpsSourcePlan plan;
    TpsSpot spot;
    spot.spot_id = 1;
    spot.energy_MeVu = config.initial_energy_MeVu;
    spot.mu_weight = 1.0;
    plan.spots.push_back(spot);
    plan.total_mu = 1.0;
    return plan;
}

TpsSourcePose TpsSourcePlan::pose_for_spot(const TransportConfig& config,
                                           const TpsSpot& spot) const {
    return central_pose(config, spot);
}

std::size_t TpsSourcePlan::active_spot_count() const noexcept {
    return static_cast<std::size_t>(
        std::count_if(spots.begin(), spots.end(), [](const TpsSpot& spot) {
            return spot.mu_weight > 0.0;
        }));
}

std::vector<std::size_t> TpsSourcePlan::allocate_histories(
    const std::size_t total_histories) const {
    const auto active = active_spot_count();
    if (!(total_mu > 0.0) || active == 0) {
        throw std::invalid_argument("TPS plan must contain positive MU");
    }
    std::vector<std::size_t> allocation(spots.size(), 0);
    std::vector<std::pair<double, std::size_t>> remainders;
    remainders.reserve(active);
    std::size_t assigned = 0;
    for (std::size_t index = 0; index < spots.size(); ++index) {
        if (spots[index].mu_weight == 0.0) {
            continue;
        }
        const auto exact = static_cast<double>(total_histories) *
                           spots[index].mu_weight / total_mu;
        const auto base = static_cast<std::size_t>(std::floor(exact));
        allocation[index] = base;
        assigned += base;
        remainders.emplace_back(exact - static_cast<double>(base), index);
    }
    std::stable_sort(remainders.begin(), remainders.end(),
                     [](const auto& left, const auto& right) {
                         return left.first > right.first;
                     });
    for (std::size_t index = 0; assigned < total_histories; ++index, ++assigned) {
        ++allocation[remainders[index].second];
    }
    return allocation;
}

std::vector<PrimarySpotBatchEntry> TpsSourcePlan::make_primary_batch(
    const TransportConfig& config) const {
    const auto allocation = allocate_histories(config.number_of_histories);
    // The transport kernel historically uses z=[0, phantom_length] for its
    // dense scorer. A patient-coordinate CT may have a non-zero z low edge.
    // Rebase only the internal transport z coordinate; output MHD metadata
    // remains in the original patient coordinates.
    double transport_z_shift_mm = 0.0;
    if (config.enable_ct_grid) {
        const auto grid = CtGrid::from_binary(config.ct_grid_file);
        const auto number_of_bins = config.number_of_bins();
        const auto close = [](const double left, const double right) {
            return std::abs(left - right) <=
                   1.0e-5 * std::max({1.0, std::abs(left), std::abs(right)});
        };
        if (grid.nx != config.voxel_bins_x || grid.ny != config.voxel_bins_y ||
            grid.nz != number_of_bins ||
            !close(grid.spacing_x_mm, config.voxel_size_x_mm) ||
            !close(grid.spacing_y_mm, config.voxel_size_y_mm) ||
            !close(grid.spacing_z_mm, config.depth_bin_width_mm) ||
            !close(static_cast<double>(grid.nz) * grid.spacing_z_mm,
                   config.phantom_length_mm)) {
            throw std::invalid_argument(
                "tpsSource with a CT requires the voxel scorer dimensions, "
                "spacing, depth bins, and phantom length to match the CT grid");
        }
        transport_z_shift_mm = -static_cast<double>(grid.origin_z_mm);
    }
    std::vector<PrimarySpotBatchEntry> batch;
    batch.reserve(active_spot_count());
    std::uint64_t history_begin = 0;
    for (std::size_t index = 0; index < spots.size(); ++index) {
        if (allocation[index] == 0) {
            continue;
        }
        const auto& spot = spots[index];
        const auto pose = pose_for_spot(config, spot);
        const auto energy_spread_percent =
            inherited(spot.energy_spread_percent, config.beam_energy_spread * 100.0);
        const auto sigma_x_mm = inherited(spot.sigma_x_mm, config.emittance_sigma_x_mm);
        const auto sigma_y_mm = inherited(spot.sigma_y_mm, config.emittance_sigma_y_mm);
        const auto sigma_x_prime =
            inherited(spot.sigma_x_prime, config.emittance_sigma_x_prime);
        const auto sigma_y_prime =
            inherited(spot.sigma_y_prime, config.emittance_sigma_y_prime);
        const auto correlation_x =
            inherited(spot.correlation_x, config.emittance_correlation_x);
        const auto correlation_y =
            inherited(spot.correlation_y, config.emittance_correlation_y);
        validate_source_parameters(
            energy_spread_percent, sigma_x_mm, sigma_y_mm, sigma_x_prime,
            sigma_y_prime, correlation_x, correlation_y,
            "TPS spot " + std::to_string(spot.spot_id));
        PrimarySpotBatchEntry entry{};
        entry.history_begin = history_begin;
        entry.history_end = history_begin + allocation[index];
        entry.random_seed = config.random_seed + index * 1'000'003ULL;
        entry.initial_energy_MeV() = static_cast<float>(
            spot.energy_MeVu * static_cast<double>(config.mass_number));
        entry.beam_energy_spread() =
            static_cast<float>(energy_spread_percent / 100.0);
        entry.emittance_sigma_x_mm() = static_cast<float>(sigma_x_mm);
        entry.emittance_sigma_y_mm() = static_cast<float>(sigma_y_mm);
        entry.emittance_sigma_x_prime() = static_cast<float>(sigma_x_prime);
        entry.emittance_sigma_y_prime() = static_cast<float>(sigma_y_prime);
        entry.emittance_correlation_x() = static_cast<float>(correlation_x);
        entry.emittance_correlation_y() = static_cast<float>(correlation_y);
        entry.source_origin_x_mm() = static_cast<float>(pose.origin_x_mm);
        entry.source_origin_y_mm() = static_cast<float>(pose.origin_y_mm);
        entry.source_origin_z_mm() =
            static_cast<float>(pose.origin_z_mm + transport_z_shift_mm);
        entry.beam_ux_x() = static_cast<float>(pose.ux_x);
        entry.beam_ux_y() = static_cast<float>(pose.ux_y);
        entry.beam_ux_z() = static_cast<float>(pose.ux_z);
        entry.beam_uy_x() = static_cast<float>(pose.uy_x);
        entry.beam_uy_y() = static_cast<float>(pose.uy_y);
        entry.beam_uy_z() = static_cast<float>(pose.uy_z);
        entry.beam_uz_x() = static_cast<float>(pose.uz_x);
        entry.beam_uz_y() = static_cast<float>(pose.uz_y);
        entry.beam_uz_z() = static_cast<float>(pose.uz_z);
        batch.push_back(entry);
        history_begin = entry.history_end;
    }
    return batch;
}

}  // namespace carbon

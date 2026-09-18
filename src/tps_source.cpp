#include "carbon/tps_source.hpp"

#include "carbon/ct_grid.hpp"
#include "carbon/topas_spots.hpp"
#include "carbon/upstream_air_scattering.hpp"
#include "carbon/sha256.hpp"
#include <optional>

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

Matrix3 dicom_lps_beam_frame(const double gantry_angle_deg,
                             const double couch_angle_deg,
                             const double collimator_angle_deg) {
    // DICOM LPS patient frame: +X left, +Y posterior, +Z superior.
    // Beam-local axes at TPS 0 degrees are u=+X, v=+Z and w=+Y. Angles rotate
    // this source frame in the fixed patient CT; no CT permutation is involved.
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
    if (config.tps_angle_convention == "dicom_lps" ||
        config.tps_angle_convention == "topas_patient_rot_z") {
        machine = dicom_lps_beam_frame(
            gantry_angle_deg, couch_angle_deg, collimator_angle_deg);
    } else {
        const auto collimator = rotation_z(-collimator_angle_deg);
        const auto gantry = rotation_y(gantry_angle_deg);
        const auto couch = rotation_z(couch_angle_deg);
        machine = multiply(couch, multiply(gantry, collimator));
    }
    const auto magnet_x = config.tps_virtual_scanning_magnet_x_mm;
    const auto magnet_y = config.tps_virtual_scanning_magnet_y_mm;
    const auto uses_virtual_magnets = magnet_x > 0.0 && magnet_y > 0.0;
    const auto source_distance_mm =
        config.tps_virtual_source_to_isocenter_mm > 0.0
            ? config.tps_virtual_source_to_isocenter_mm
            : config.tps_sad_mm;
    const auto apply_topas_patient =
        config.tps_apply_topas_patient_placement ||
        (uses_virtual_magnets && config.spots_patient_rot_z_deg != 0.0);

    auto magnet_pose = [&](const Vec3& u0, const Vec3& v0, const Vec3& w0,
                           const Vec3& isocenter) {
        const auto x_src =
            spot.x_mm * (magnet_x - source_distance_mm) / magnet_x;
        const auto y_src =
            spot.y_mm * (magnet_y - source_distance_mm) / magnet_y;
        const auto origin =
            isocenter + x_src * u0 + y_src * v0 - source_distance_mm * w0;
        const auto tan_x = spot.x_mm / magnet_x;
        const auto tan_y = spot.y_mm / magnet_y;
        auto w = tan_x * u0 + tan_y * v0 + w0;
        const auto w_norm = std::sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
        w = (1.0 / w_norm) * w;
        auto u = u0 - (u0.x * w.x + u0.y * w.y + u0.z * w.z) * w;
        const auto u_norm = std::sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
        u = (1.0 / u_norm) * u;
        const Vec3 v{w.y * u.z - w.z * u.y, w.z * u.x - w.x * u.z,
                     w.x * u.y - w.y * u.x};
        return TpsSourcePose{
            origin.x, origin.y, origin.z, u.x, u.y, u.z,
            v.x, v.y, v.z, w.x, w.y, w.z,
        };
    };

    if (uses_virtual_magnets && apply_topas_patient) {
        // World TPS 0°: +X scan X, +Y beam, +Z scan Y. TOPAS applies Patient
        // RotZ first, then Trans; the same tps_90 packing as Time Feature.
        const Vec3 u0{1.0, 0.0, 0.0};
        const Vec3 v0{0.0, 0.0, 1.0};
        const Vec3 w0{0.0, 1.0, 0.0};
        const auto world = magnet_pose(u0, v0, w0, Vec3{0.0, 0.0, 0.0});
        SpotSourcePose world_pose{};
        world_pose.origin_x_mm = world.origin_x_mm;
        world_pose.origin_y_mm = world.origin_y_mm;
        world_pose.origin_z_mm = world.origin_z_mm;
        world_pose.ux_x = world.ux_x;
        world_pose.ux_y = world.ux_y;
        world_pose.ux_z = world.ux_z;
        world_pose.uy_x = world.uy_x;
        world_pose.uy_y = world.uy_y;
        world_pose.uy_z = world.uy_z;
        world_pose.uz_x = world.uz_x;
        world_pose.uz_y = world.uz_y;
        world_pose.uz_z = world.uz_z;
        const auto ct = transform_tps_90_pose_to_ct(
            world_pose, config.spots_patient_trans_x_mm,
            config.spots_patient_trans_y_mm, config.spots_patient_trans_z_mm,
            config.spots_patient_rot_z_deg, config.spots_ct_axis_min_mm);
        return {ct.origin_x_mm, ct.origin_y_mm, ct.origin_z_mm,
                ct.ux_x,        ct.ux_y,        ct.ux_z,
                ct.uy_x,        ct.uy_y,        ct.uy_z,
                ct.uz_x,        ct.uz_y,        ct.uz_z};
    }

    const auto patient = orientation_matrix(config.tps_patient_position);
    const auto rotation = multiply(patient, machine);
    const auto u0 = multiply(rotation, Vec3{1.0, 0.0, 0.0});
    const auto v0 = multiply(rotation, Vec3{0.0, 1.0, 0.0});
    const auto w0 = multiply(rotation, Vec3{0.0, 0.0, -1.0});
    const Vec3 isocenter{config.tps_isocenter_x_mm, config.tps_isocenter_y_mm,
                         config.tps_isocenter_z_mm};
    if (!uses_virtual_magnets) {
        const auto central_source = isocenter - source_distance_mm * w0;
        const auto origin = central_source + spot.x_mm * u0 + spot.y_mm * v0;
        return {
            origin.x, origin.y, origin.z,
            u0.x, u0.y, u0.z,
            v0.x, v0.y, v0.z,
            w0.x, w0.y, w0.z,
        };
    }
    return magnet_pose(u0, v0, w0, isocenter);
}

TpsSourcePose explicit_pose(const TpsSpot& spot) {
    Vec3 w{spot.direction_x, spot.direction_y, spot.direction_z};
    const auto w_norm = std::sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
    if (!(w_norm > 0.0) || !std::isfinite(w_norm)) {
        throw std::invalid_argument("TPS explicit source direction must be finite and nonzero");
    }
    w = (1.0 / w_norm) * w;

    // The transverse axes do not affect deterministic replay, but keeping a
    // stable right-handed frame also makes explicit poses safe with emittance.
    const Vec3 reference = std::abs(w.z) < 0.9 ? Vec3{0.0, 0.0, 1.0}
                                                : Vec3{0.0, 1.0, 0.0};
    Vec3 u{reference.y * w.z - reference.z * w.y,
           reference.z * w.x - reference.x * w.z,
           reference.x * w.y - reference.y * w.x};
    const auto u_norm = std::sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
    u = (1.0 / u_norm) * u;
    const Vec3 v{w.y * u.z - w.z * u.y,
                 w.z * u.x - w.x * u.z,
                 w.x * u.y - w.y * u.x};
    return {spot.source_x_mm, spot.source_y_mm, spot.source_z_mm,
            u.x, u.y, u.z, v.x, v.y, v.z, w.x, w.y, w.z};
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
    const auto has_energy_mevu = columns.contains("energy_mevu");
    const auto has_energy_mev =
        columns.contains("energy_mev") || columns.contains("energy");
    const auto has_x = columns.contains("x_mm") || columns.contains("x");
    const auto has_y = columns.contains("y_mm") || columns.contains("y");
    const auto has_mu = columns.contains("mu_weight");
    const auto has_weight = columns.contains("weight");
    constexpr std::array<const char*, 6> explicit_pose_columns{
        "source_x_mm", "source_y_mm", "source_z_mm",
        "direction_x", "direction_y", "direction_z"};
    const auto explicit_pose_count = static_cast<std::size_t>(std::count_if(
        explicit_pose_columns.begin(), explicit_pose_columns.end(),
        [&](const char* name) { return columns.contains(name); }));
    if (explicit_pose_count != 0 &&
        explicit_pose_count != explicit_pose_columns.size()) {
        throw std::runtime_error(
            "TPS spots CSV explicit source pose requires source_x/y/z_mm and "
            "direction_x/y/z together: " + path.string());
    }
    if ((!has_energy_mevu && !has_energy_mev) || !has_x || !has_y ||
        (!has_mu && !has_weight)) {
        throw std::runtime_error(
            "TPS spots CSV needs energy (energy_MeVu or energy_MeV), x, y, "
            "and weight (mu_weight or weight): " +
            path.string());
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
        if (has_energy_mevu) {
            spot.energy_MeVu =
                parse_required(fields, columns, "energy_mevu", path, line_number);
        } else if (columns.contains("energy_mev")) {
            spot.energy_total_MeV =
                parse_required(fields, columns, "energy_mev", path, line_number);
        } else {
            spot.energy_total_MeV =
                parse_required(fields, columns, "energy", path, line_number);
        }
        spot.x_mm = parse_required(
            fields, columns, columns.contains("x_mm") ? "x_mm" : "x", path,
            line_number);
        spot.y_mm = parse_required(
            fields, columns, columns.contains("y_mm") ? "y_mm" : "y", path,
            line_number);
        if (has_mu) {
            spot.mu_weight =
                parse_required(fields, columns, "mu_weight", path, line_number);
        } else {
            spot.mu_weight =
                parse_required(fields, columns, "weight", path, line_number);
        }
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
        if (explicit_pose_count != 0) {
            spot.source_x_mm = parse_required(
                fields, columns, "source_x_mm", path, line_number);
            spot.source_y_mm = parse_required(
                fields, columns, "source_y_mm", path, line_number);
            spot.source_z_mm = parse_required(
                fields, columns, "source_z_mm", path, line_number);
            spot.direction_x = parse_required(
                fields, columns, "direction_x", path, line_number);
            spot.direction_y = parse_required(
                fields, columns, "direction_y", path, line_number);
            spot.direction_z = parse_required(
                fields, columns, "direction_z", path, line_number);
            const auto direction_norm = std::sqrt(
                spot.direction_x * spot.direction_x +
                spot.direction_y * spot.direction_y +
                spot.direction_z * spot.direction_z);
            if (!(direction_norm > 0.0) || !std::isfinite(direction_norm)) {
                throw std::runtime_error(
                    "TPS spot explicit source direction must be nonzero at " +
                    path.string() + ":" + std::to_string(line_number));
            }
        }
        const auto energy_ok = std::isfinite(spot.energy_total_MeV)
                                   ? spot.energy_total_MeV > 0.0
                                   : spot.energy_MeVu > 0.0;
        if (!energy_ok || spot.mu_weight < 0.0) {
            throw std::runtime_error(
                "TPS spot energy must be positive and weight nonnegative at " +
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
        auto plan = from_csv(config.tps_spots_file);
        if (!config.tps_beam_model_file.empty()) {
            plan.apply_beam_model(config.tps_beam_model_file);
        }
        return plan;
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

void TpsSourcePlan::apply_beam_model(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open TPS beam model: " + path.string());
    }
    struct Optics {
        double energy_MeV{0.0};
        double sigma_x_mm{0.0};
        double sigma_x_prime{0.0};
        double correlation_x{0.0};
        double sigma_y_mm{0.0};
        double sigma_y_prime{0.0};
        double correlation_y{0.0};
        double energy_spread_percent{0.0};
    };
    std::string line;
    std::size_t line_number = 0;
    std::unordered_map<std::string, std::size_t> columns;
    std::vector<Optics> rows;
    while (std::getline(input, line)) {
        ++line_number;
        if (!trim(line).empty() && trim(line).front() != '#') {
            const auto header = split_csv_line(line);
            for (std::size_t index = 0; index < header.size(); ++index) {
                columns[normalized_header(header[index])] = index;
            }
            break;
        }
    }
    const auto energy_key = columns.contains("energy_mev") ? "energy_mev" : "energy";
    for (const char* required :
         {energy_key, "sigma_x_mm", "sigma_xp_rad", "corr_x", "sigma_y_mm",
          "sigma_yp_rad", "corr_y", "energy_spread_percent"}) {
        if (!columns.contains(required)) {
            throw std::runtime_error("TPS beam model missing '" +
                                     std::string(required) + "': " + path.string());
        }
    }
    while (std::getline(input, line)) {
        ++line_number;
        if (trim(line).empty() || trim(line).front() == '#') {
            continue;
        }
        const auto fields = split_csv_line(line);
        Optics row;
        row.energy_MeV = parse_required(fields, columns, energy_key, path, line_number);
        row.sigma_x_mm =
            parse_required(fields, columns, "sigma_x_mm", path, line_number);
        row.sigma_x_prime =
            parse_required(fields, columns, "sigma_xp_rad", path, line_number);
        row.correlation_x =
            parse_required(fields, columns, "corr_x", path, line_number);
        row.sigma_y_mm =
            parse_required(fields, columns, "sigma_y_mm", path, line_number);
        row.sigma_y_prime =
            parse_required(fields, columns, "sigma_yp_rad", path, line_number);
        row.correlation_y =
            parse_required(fields, columns, "corr_y", path, line_number);
        row.energy_spread_percent = parse_required(
            fields, columns, "energy_spread_percent", path, line_number);
        rows.push_back(row);
    }
    if (rows.size() < 2) {
        throw std::runtime_error("TPS beam model needs at least two energies: " +
                                 path.string());
    }
    std::sort(rows.begin(), rows.end(), [](const Optics& left, const Optics& right) {
        return left.energy_MeV < right.energy_MeV;
    });
    const auto interpolate = [&rows](const double energy) {
        if (energy <= rows.front().energy_MeV) {
            return rows.front();
        }
        if (energy >= rows.back().energy_MeV) {
            return rows.back();
        }
        const auto upper = std::lower_bound(
            rows.begin(), rows.end(), energy,
            [](const Optics& row, const double value) {
                return row.energy_MeV < value;
            });
        const auto& b = *upper;
        const auto& a = *std::prev(upper);
        const auto t = (energy - a.energy_MeV) / (b.energy_MeV - a.energy_MeV);
        Optics out = a;
        out.energy_MeV = energy;
        out.sigma_x_mm += t * (b.sigma_x_mm - a.sigma_x_mm);
        out.sigma_x_prime += t * (b.sigma_x_prime - a.sigma_x_prime);
        out.correlation_x += t * (b.correlation_x - a.correlation_x);
        out.sigma_y_mm += t * (b.sigma_y_mm - a.sigma_y_mm);
        out.sigma_y_prime += t * (b.sigma_y_prime - a.sigma_y_prime);
        out.correlation_y += t * (b.correlation_y - a.correlation_y);
        out.energy_spread_percent +=
            t * (b.energy_spread_percent - a.energy_spread_percent);
        return out;
    };
    for (auto& spot : spots) {
        const auto energy = std::isfinite(spot.energy_total_MeV)
                                ? spot.energy_total_MeV
                                : spot.energy_MeVu * 12.0;
        const auto optics = interpolate(energy);
        if (!std::isfinite(spot.sigma_x_mm)) {
            spot.sigma_x_mm = optics.sigma_x_mm;
        }
        if (!std::isfinite(spot.sigma_y_mm)) {
            spot.sigma_y_mm = optics.sigma_y_mm;
        }
        if (!std::isfinite(spot.sigma_x_prime)) {
            spot.sigma_x_prime = optics.sigma_x_prime;
        }
        if (!std::isfinite(spot.sigma_y_prime)) {
            spot.sigma_y_prime = optics.sigma_y_prime;
        }
        if (!std::isfinite(spot.correlation_x)) {
            spot.correlation_x = optics.correlation_x;
        }
        if (!std::isfinite(spot.correlation_y)) {
            spot.correlation_y = optics.correlation_y;
        }
        if (!std::isfinite(spot.energy_spread_percent)) {
            spot.energy_spread_percent = optics.energy_spread_percent;
        }
    }
}

TpsSourcePose TpsSourcePlan::pose_for_spot(const TransportConfig& config,
                                           const TpsSpot& spot) const {
    if (std::isfinite(spot.source_x_mm)) {
        return explicit_pose(spot);
    }
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
    const TransportConfig& config,
    const StoppingPowerTable* const upstream_air_stopping_power) const {
    if (config.tps_spot_weight_mode != "mu" &&
        config.tps_spot_weight_mode != "histories") {
        throw std::invalid_argument(
            "tps_spot_weight_mode must be mu or histories");
    }
    std::vector<std::size_t> allocation;
    if (config.tps_spot_weight_mode == "histories") {
        if (!(config.tps_histories_scale > 0.0) ||
            !std::isfinite(config.tps_histories_scale)) {
            throw std::invalid_argument("tps_histories_scale must be positive");
        }
        allocation.assign(spots.size(), 0);
        for (std::size_t index = 0; index < spots.size(); ++index) {
            if (spots[index].mu_weight <= 0.0) {
                continue;
            }
            const auto scaled = spots[index].mu_weight * config.tps_histories_scale;
            allocation[index] = static_cast<std::size_t>(std::llround(scaled));
            if (allocation[index] == 0 && scaled > 0.0) {
                allocation[index] = 1;
            }
        }
    } else {
        allocation = allocate_histories(config.number_of_histories);
    }
    // CCTG uses conventional patient coordinates: x-y is axial and z is the
    // slice direction. The kernel stores that same orientation but rebases the
    // z low edge to zero internally. Output metadata restores the CT origin.
    double transport_z_shift_mm = 0.0;
    double ct_min_x_mm = 0.0;
    double ct_max_x_mm = 0.0;
    double ct_min_y_mm = 0.0;
    double ct_max_y_mm = 0.0;
    double ct_max_z_mm = 0.0;
    if (config.enable_ct_grid) {
        const auto grid = CtGrid::from_config(config);
        const auto number_of_bins = config.number_of_bins();
        const auto close = [](const double left, const double right) {
            return std::abs(left - right) <=
                   1.0e-5 * std::max({1.0, std::abs(left), std::abs(right)});
        };
        if (grid.nx != config.voxel_bins_x || grid.ny != config.voxel_bins_y ||
            grid.nz != number_of_bins ||
            !close(grid.spacing_x_mm, config.voxel_size_x_mm) ||
            !close(grid.spacing_y_mm, config.voxel_size_y_mm) ||
            !close(grid.spacing_z_mm, config.scorer_spacing_z_mm()) ||
            !close(static_cast<double>(grid.nz) * grid.spacing_z_mm,
                   config.phantom_length_mm)) {
            throw std::invalid_argument(
                "tpsSource with a CT requires the voxel scorer dimensions, "
                "spacing, depth bins, and phantom length to match the CT grid");
        }
        transport_z_shift_mm = -static_cast<double>(grid.origin_z_mm);
        ct_min_x_mm = grid.origin_x_mm;
        ct_max_x_mm = grid.origin_x_mm + grid.extent_x_mm();
        ct_min_y_mm = grid.origin_y_mm;
        ct_max_y_mm = grid.origin_y_mm + grid.extent_y_mm();
        ct_max_z_mm = grid.extent_z_mm();
    } else if (upstream_air_stopping_power != nullptr) {
        throw std::invalid_argument(
            "TPS upstream air energy loss requires a CT grid");
    }

    const auto distance_to_ct_entry = [&](const TpsSourcePose& pose) {
        auto enter = 0.0;
        auto exit = std::numeric_limits<double>::infinity();
        const auto intersect = [&](const double position, const double direction,
                                   const double lower, const double upper) {
            if (std::abs(direction) < 1.0e-12) {
                return position >= lower && position < upper;
            }
            auto first = (lower - position) / direction;
            auto second = (upper - position) / direction;
            if (first > second) std::swap(first, second);
            enter = std::max(enter, first);
            exit = std::min(exit, second);
            return exit >= enter;
        };
        const auto transport_z = pose.origin_z_mm + transport_z_shift_mm;
        if (!intersect(pose.origin_x_mm, pose.uz_x, ct_min_x_mm, ct_max_x_mm) ||
            !intersect(pose.origin_y_mm, pose.uz_y, ct_min_y_mm, ct_max_y_mm) ||
            !intersect(transport_z, pose.uz_z, 0.0, ct_max_z_mm) ||
            !std::isfinite(enter)) {
            throw std::invalid_argument(
                "TPS central spot ray does not intersect the CT grid");
        }
        return enter;
    };
    std::vector<PrimarySpotBatchEntry> batch;
    std::optional<AirMomentTable> air_moments;
    if (config.spots_enable_upstream_air_mcs) {
        if (!file_sha256_matches(config.spots_upstream_air_mcs_file,config.spots_upstream_air_mcs_sha256))
            throw std::invalid_argument("Air moment table SHA mismatch");
        air_moments=AirMomentTable::read(config.spots_upstream_air_mcs_file);
    }
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
        const auto total_energy_MeV = std::isfinite(spot.energy_total_MeV)
            ? spot.energy_total_MeV
            : spot.energy_MeVu * static_cast<double>(config.primary_mass_number);
        const auto entry_energy_MeV =
            upstream_air_stopping_power == nullptr
                ? total_energy_MeV
                : spot_entry_total_energy_after_optional_upstream_loss(
                      total_energy_MeV, config.primary_mass_number,
                      distance_to_ct_entry(pose), upstream_air_stopping_power);
        entry.initial_energy_MeV() = static_cast<float>(entry_energy_MeV);
        entry.beam_energy_spread() =
            static_cast<float>(energy_spread_percent / 100.0);
        entry.emittance_sigma_x_mm() = static_cast<float>(sigma_x_mm);
        entry.emittance_sigma_y_mm() = static_cast<float>(sigma_y_mm);
        entry.emittance_sigma_x_prime() = static_cast<float>(sigma_x_prime);
        entry.emittance_sigma_y_prime() = static_cast<float>(sigma_y_prime);
        entry.emittance_correlation_x() = static_cast<float>(correlation_x);
        entry.emittance_correlation_y() = static_cast<float>(correlation_y);
        if (config.spots_enable_upstream_air_mcs) {
            if (upstream_air_stopping_power == nullptr)
                throw std::invalid_argument("Air MCS requires the upstream stopping table");
            const auto length = distance_to_ct_entry(pose);
            if (config.primary_atomic_number!=6 || config.primary_mass_number!=12)
                throw std::invalid_argument("Measured air moment table supports C12 only");
            const auto moments=air_moments->sample(total_energy_MeV/12.,length);
            const auto x = add_measured_air_covariance({sigma_x_mm, sigma_x_prime, correlation_x}, length, moments);
            const auto y = add_measured_air_covariance({sigma_y_mm, sigma_y_prime, correlation_y}, length, moments);
            entry.emittance_sigma_x_mm() = static_cast<float>(x.sigma_position_mm);
            entry.emittance_sigma_y_mm() = static_cast<float>(y.sigma_position_mm);
            entry.emittance_sigma_x_prime() = static_cast<float>(x.sigma_angle_rad);
            entry.emittance_sigma_y_prime() = static_cast<float>(y.sigma_angle_rad);
            entry.emittance_correlation_x() = static_cast<float>(x.correlation);
            entry.emittance_correlation_y() = static_cast<float>(y.correlation);
        }
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

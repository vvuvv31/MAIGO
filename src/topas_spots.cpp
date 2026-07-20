#include "carbon/topas_spots.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace carbon {
namespace {

struct ChannelData {
    std::vector<double> times_ms{};
    std::vector<double> values{};
};

std::string strip_comment(std::string line) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) {
        line.resize(hash);
    }
    return line;
}

// Parse TOPAS "type:Path = body" lines. type is s/d/u/i/b + optional v.
bool parse_assignment(const std::string& line, std::string& type, std::string& path,
                      std::string& body) {
    static const std::regex re(
        R"(^\s*([sduib]v?):([A-Za-z0-9_./]+)\s*=\s*(.+?)\s*$)", std::regex::icase);
    std::smatch match;
    if (!std::regex_match(line, match, re)) {
        return false;
    }
    type = match[1].str();
    path = match[2].str();
    body = match[3].str();
    return true;
}

std::vector<double> parse_numeric_vector(const std::string& body) {
    std::istringstream input(body);
    std::vector<std::string> tokens;
    std::string token;
    while (input >> token) {
        // Drop trailing unit tokens.
        const auto lower = token;
        // Keep pure numbers / scientific.
        bool numeric = false;
        try {
            size_t idx = 0;
            (void)std::stod(token, &idx);
            numeric = idx > 0;
        } catch (...) {
            numeric = false;
        }
        if (numeric) {
            tokens.push_back(token);
        }
    }
    if (tokens.empty()) {
        return {};
    }
    // TOPAS vector: first entry is count, remainder are values (may omit count mismatch).
    std::vector<double> values;
    values.reserve(tokens.size());
    for (const auto& t : tokens) {
        values.push_back(std::stod(t));
    }
    if (values.size() >= 2) {
        const auto declared = static_cast<std::size_t>(std::llround(values.front()));
        if (declared + 1 == values.size()) {
            return std::vector<double>(values.begin() + 1, values.end());
        }
    }
    // Single-value form without count is rare; keep all.
    return values;
}

// Extract L index from "Tf/Scatterer1/L12/Values"
bool parse_channel_path(const std::string& path, std::string& scatterer, int& layer,
                        std::string& field) {
    static const std::regex re(
        R"(^Tf/([A-Za-z0-9_]+)/L(\d+)/(Times|Values|Function)$)", std::regex::icase);
    std::smatch match;
    if (!std::regex_match(path, match, re)) {
        return false;
    }
    scatterer = match[1].str();
    layer = std::stoi(match[2].str());
    field = match[3].str();
    return true;
}

}  // namespace

std::size_t TopasSpotPlan::total_histories() const noexcept {
    std::size_t total = 0;
    for (const auto& spot : spots) {
        total += spot.number_of_histories;
    }
    return total;
}

void rotate_rx_ry(double rx_deg, double ry_deg, double& x, double& y, double& z) noexcept {
    constexpr double deg2rad = 3.14159265358979323846 / 180.0;
    const double cx = std::cos(rx_deg * deg2rad);
    const double sx = std::sin(rx_deg * deg2rad);
    const double cy = std::cos(ry_deg * deg2rad);
    const double sy = std::sin(ry_deg * deg2rad);
    // Rx
    {
        const double y2 = cx * y - sx * z;
        const double z2 = sx * y + cx * z;
        y = y2;
        z = z2;
    }
    // Ry
    {
        const double x2 = cy * x + sy * z;
        const double z2 = -sy * x + cy * z;
        x = x2;
        z = z2;
    }
}

SpotSourcePose TopasSpotPlan::pose_for_spot(const TopasSpot& spot) const noexcept {
    // Match TOPAS BeamPosition2: origin at (TransX, -SAD, TransZ) then RotX/RotY of the
    // volume. Local beam axis is +Z of that volume; emittance lives in local X/Y.
    SpotSourcePose pose{};
    double ox = spot.trans_x_mm;
    double oy = -sad_mm;
    double oz = spot.trans_z_mm;
    rotate_rx_ry(spot.rot_x_deg, spot.rot_y_deg, ox, oy, oz);
    pose.origin_x_mm = ox;
    pose.origin_y_mm = oy;
    pose.origin_z_mm = oz;

    double ux = 1.0, uy = 0.0, uz = 0.0;
    double vx = 0.0, vy = 1.0, vz = 0.0;
    double wx = 0.0, wy = 0.0, wz = 1.0;
    rotate_rx_ry(spot.rot_x_deg, spot.rot_y_deg, ux, uy, uz);
    rotate_rx_ry(spot.rot_x_deg, spot.rot_y_deg, vx, vy, vz);
    rotate_rx_ry(spot.rot_x_deg, spot.rot_y_deg, wx, wy, wz);
    pose.ux_x = ux;
    pose.ux_y = uy;
    pose.ux_z = uz;
    pose.uy_x = vx;
    pose.uy_y = vy;
    pose.uy_z = vz;
    pose.uz_x = wx;
    pose.uz_y = wy;
    pose.uz_z = wz;
    return pose;
}

SpotSourcePose TopasSpotPlan::tps_zero_beam_pose_for_spot(
    const TopasSpot& spot) const noexcept {
    SpotSourcePose pose{};
    // Component translations are expressed in the parent (World) frame.
    pose.origin_x_mm = spot.trans_x_mm;
    pose.origin_y_mm = -sad_mm;
    pose.origin_z_mm = spot.trans_z_mm;

    double ux = 1.0, uy = 0.0, uz = 0.0;
    double vx = 0.0, vy = 1.0, vz = 0.0;
    double wx = 0.0, wy = 0.0, wz = 1.0;
    rotate_rx_ry(-spot.rot_x_deg, -spot.rot_y_deg, ux, uy, uz);
    rotate_rx_ry(-spot.rot_x_deg, -spot.rot_y_deg, vx, vy, vz);
    rotate_rx_ry(-spot.rot_x_deg, -spot.rot_y_deg, wx, wy, wz);
    pose.ux_x = ux;
    pose.ux_y = uy;
    pose.ux_z = uz;
    pose.uy_x = vx;
    pose.uy_y = vy;
    pose.uy_z = vz;
    pose.uz_x = wx;
    pose.uz_y = wy;
    pose.uz_z = wz;
    return pose;
}

SpotSourcePose transform_tps_90_pose_to_ct(
    const SpotSourcePose& world_pose,
    const double patient_trans_x_mm,
    const double patient_trans_y_mm,
    const double patient_trans_z_mm,
    const double patient_rot_z_deg,
    const double ct_axis_min_mm) noexcept {
    constexpr double deg2rad = 3.14159265358979323846 / 180.0;
    const auto angle = -patient_rot_z_deg * deg2rad;
    const auto c = std::cos(angle);
    const auto s = std::sin(angle);

    auto point_to_ct = [&](double x, double y, double z) {
        x -= patient_trans_x_mm;
        y -= patient_trans_y_mm;
        z -= patient_trans_z_mm;
        const auto patient_x = c * x - s * y;
        const auto patient_y = s * x + c * y;
        // Reoriented grid axes: (patient Y, patient Z, patient X).
        return std::tuple{patient_y, z, patient_x - ct_axis_min_mm};
    };
    auto vector_to_ct = [&](const double x, const double y, const double z) {
        const auto patient_x = c * x - s * y;
        const auto patient_y = s * x + c * y;
        return std::tuple{patient_y, z, patient_x};
    };

    SpotSourcePose pose{};
    std::tie(pose.origin_x_mm, pose.origin_y_mm, pose.origin_z_mm) =
        point_to_ct(world_pose.origin_x_mm, world_pose.origin_y_mm,
                    world_pose.origin_z_mm);
    std::tie(pose.ux_x, pose.ux_y, pose.ux_z) =
        vector_to_ct(world_pose.ux_x, world_pose.ux_y, world_pose.ux_z);
    std::tie(pose.uy_x, pose.uy_y, pose.uy_z) =
        vector_to_ct(world_pose.uy_x, world_pose.uy_y, world_pose.uy_z);
    std::tie(pose.uz_x, pose.uz_y, pose.uz_z) =
        vector_to_ct(world_pose.uz_x, world_pose.uz_y, world_pose.uz_z);
    return pose;
}

TopasSpotPlan TopasSpotPlan::from_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open TOPAS spots file: " + path.string());
    }

    std::unordered_map<int, ChannelData> channels;
    std::string scatterer = "Scatterer1";
    std::string line;
    while (std::getline(input, line)) {
        line = strip_comment(std::move(line));
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        std::string type;
        std::string path_token;
        std::string body;
        if (!parse_assignment(line, type, path_token, body)) {
            continue;
        }
        std::string sc;
        int layer = -1;
        std::string field;
        if (!parse_channel_path(path_token, sc, layer, field)) {
            // Allow Times alias lines: dv:Tf/.../L1/Times = Tf/.../L0/Times ms
            continue;
        }
        scatterer = sc;
        if (field == "Function") {
            continue;
        }
        if (field == "Times") {
            // Only honor explicit numeric Times on L0 (others alias L0).
            if (layer == 0 && type.find('v') != std::string::npos) {
                channels[0].times_ms = parse_numeric_vector(body);
            }
            continue;
        }
        if (field == "Values") {
            channels[layer].values = parse_numeric_vector(body);
        }
    }

    if (channels.find(0) == channels.end() || channels[0].values.empty()) {
        throw std::runtime_error("TOPAS spots file missing L0 Values: " + path.string());
    }
    // Prefer L0 Times; else synthesize unit steps.
    auto times = channels[0].times_ms;
    const auto n = channels[0].values.size();
    if (times.size() != n) {
        times.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            times[i] = static_cast<double>(i + 1);
        }
    }

    auto value_at = [&](int layer, std::size_t index, double fallback) {
        const auto it = channels.find(layer);
        if (it == channels.end() || it->second.values.size() <= index) {
            return fallback;
        }
        return it->second.values[index];
    };

    TopasSpotPlan plan;
    plan.scatterer_name = scatterer;
    plan.spots.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        TopasSpot spot;
        spot.spot_id = static_cast<int>(std::llround(value_at(0, i, static_cast<double>(i))));
        // L2 is BeamEnergy (total MeV); L1 often duplicates energy in plan dumps.
        spot.energy_MeV = value_at(2, i, value_at(1, i, 0.0));
        spot.energy_spread_percent = value_at(3, i, 0.0);
        spot.number_of_histories = static_cast<std::size_t>(
            std::max(0LL, std::llround(value_at(4, i, 0.0))));
        spot.trans_x_mm = value_at(5, i, 0.0);
        spot.trans_z_mm = value_at(6, i, 0.0);
        spot.rot_x_deg = value_at(7, i, 0.0);
        spot.rot_y_deg = value_at(8, i, 0.0);
        spot.sigma_x_mm = value_at(9, i, 0.0);
        spot.sigma_x_prime = value_at(10, i, 0.0);
        spot.correlation_x = value_at(11, i, 0.0);
        spot.sigma_y_mm = value_at(12, i, 0.0);
        spot.sigma_y_prime = value_at(13, i, 0.0);
        spot.correlation_y = value_at(14, i, 0.0);
        spot.time_end_ms = times[i];
        if (spot.energy_MeV <= 0.0) {
            throw std::runtime_error("TOPAS spot has non-positive BeamEnergy in " +
                                     path.string());
        }
        if (spot.number_of_histories == 0) {
            throw std::runtime_error("TOPAS spot has zero histories in " + path.string());
        }
        plan.spots.push_back(spot);
    }
    if (plan.spots.empty()) {
        throw std::runtime_error("TOPAS spots file contains no spots: " + path.string());
    }
    return plan;
}

TopasSpotPlan TopasSpotPlan::from_files(
    const std::vector<std::filesystem::path>& paths) {
    if (paths.empty()) {
        throw std::invalid_argument("TOPAS spots file list is empty");
    }
    TopasSpotPlan combined;
    combined.spots.clear();
    for (const auto& path : paths) {
        auto part = from_file(path);
        if (combined.spots.empty()) {
            combined.scatterer_name = part.scatterer_name;
        }
        combined.spots.insert(combined.spots.end(), part.spots.begin(), part.spots.end());
    }
    return combined;
}

std::size_t TopasSpotPlan::apply_weights_from_csv(
    const std::filesystem::path& path,
    const std::size_t requested_total_histories) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open spot weights file: " + path.string());
    }
    std::vector<double> weights;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line_number == 1 && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEFU &&
            static_cast<unsigned char>(line[1]) == 0xBBU &&
            static_cast<unsigned char>(line[2]) == 0xBFU) {
            line.erase(0, 3);
        }
        const auto comma = line.find(',');
        if (comma != std::string::npos) {
            line.resize(comma);
        }
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            continue;
        }
        const auto last = line.find_last_not_of(" \t\r\n");
        const auto token = line.substr(first, last - first + 1);
        std::size_t parsed = 0;
        double weight = 0.0;
        try {
            weight = std::stod(token, &parsed);
        } catch (...) {
            throw std::runtime_error("Invalid spot weight at " + path.string() + ":" +
                                     std::to_string(line_number));
        }
        if (parsed != token.size() || !std::isfinite(weight) || weight < 0.0) {
            throw std::runtime_error("Spot weight must be a finite nonnegative number at " +
                                     path.string() + ":" +
                                     std::to_string(line_number));
        }
        weights.push_back(weight);
    }
    if (weights.size() != spots.size()) {
        throw std::runtime_error("Spot weight count " + std::to_string(weights.size()) +
                                 " does not match concatenated spot count " +
                                 std::to_string(spots.size()));
    }
    total_plan_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    const auto positive_count = static_cast<std::size_t>(
        std::count_if(weights.begin(), weights.end(), [](double value) { return value > 0.0; }));
    if (!(total_plan_weight > 0.0) || positive_count == 0) {
        throw std::runtime_error("Spot weights must contain at least one positive value");
    }
    if (requested_total_histories < positive_count) {
        throw std::runtime_error("Total plan histories must be at least the number of "
                                 "positive-weight spots (" +
                                 std::to_string(positive_count) + ")");
    }

    // Give every active spot one history, then use Hamilton/largest-remainder
    // allocation for the remainder. This preserves the exact requested total.
    const auto distributable = requested_total_histories - positive_count;
    std::vector<std::size_t> allocation(weights.size(), 0);
    std::vector<std::pair<double, std::size_t>> remainders;
    remainders.reserve(positive_count);
    std::size_t assigned = positive_count;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        if (weights[i] == 0.0) {
            continue;
        }
        const auto exact = static_cast<double>(distributable) * weights[i] /
                           total_plan_weight;
        const auto base = static_cast<std::size_t>(std::floor(exact));
        allocation[i] = 1 + base;
        assigned += base;
        remainders.emplace_back(exact - static_cast<double>(base), i);
    }
    std::stable_sort(remainders.begin(), remainders.end(),
                     [](const auto& left, const auto& right) {
                         return left.first > right.first;
                     });
    for (std::size_t i = 0; assigned < requested_total_histories; ++i, ++assigned) {
        ++allocation[remainders[i].second];
    }

    std::vector<TopasSpot> active;
    active.reserve(positive_count);
    for (std::size_t i = 0; i < spots.size(); ++i) {
        if (allocation[i] == 0) {
            continue;
        }
        spots[i].number_of_histories = allocation[i];
        active.push_back(spots[i]);
    }
    const auto removed = spots.size() - active.size();
    spots = std::move(active);
    return removed;
}

}  // namespace carbon

#include "carbon/topas_spots.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
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

}  // namespace carbon

// Host-exact replica of kernel primary sampling for the TPS spot-batch
// source (src/transport_sycl.cpp: sample energy dims 40/41, emittance dims
// 30-33, pose transform, phantom-box entry clip). Emits per-history phase
// space at box entry in TOPAS world coordinates plus per-spot JSON moments.
// No transport is performed. Used for GPU-vs-TOPAS source truth comparison.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "carbon/ct_grid.hpp"
#include "carbon/rng.hpp"
#include "carbon/topas_spots.hpp"
#include "carbon/tps_source.hpp"
#include "carbon/transport_config.hpp"

namespace {

constexpr float kTwoPi = 6.2831853071795864769F;

struct WorldState {
    double x{0}, y{0}, z{0}, dx{0}, dy{0}, dz{0}, e{0};
    bool hit{false};
};

// Inverse of transform_tps_90_pose_to_ct for points and vectors.
// Round-trip validated at startup against the library forward map.
struct InverseTps90 {
    double tx{0}, ty{0}, tz{0};
    double c{0}, s{0};
    double ct_axis_min_mm{0};
    bool minus_x_branch{false};  // beam along -patient X (uzx < 0)

    void to_patient(double gx, double gy, double gz, bool is_vector,
                    double& ox, double& oy, double& oz) const {
        if (!minus_x_branch) {
            ox = gx + ct_axis_min_mm;  // not exercised by RT06423; kept symmetric
            oy = gx;
            oz = gy;
            (void)gz;
            (void)is_vector;
            ox = 0;  // unreachable guard
            throw std::runtime_error("plus-X branch not implemented");
        }
        const double ct_max = -ct_axis_min_mm;
        oy = gx;
        oz = gy;
        ox = ct_max - gz;
        if (is_vector) {
            // Forward (-X branch): ux=(uxy,uxz,-uxx) etc. Invert per-axis below
            // via the entry-frame reconstruction in to_world_vector.
        }
    }

    // Inverse rotation R(-RotZ) applied to patient coords -> world offset.
    void patient_to_world(double px, double py, double pz, double& wx, double& wy,
                          double& wz) const {
        wx = c * px + s * py + tx;
        wy = -s * px + c * py + ty;
        wz = pz + tz;
    }
};

int run(const std::string& config_path, const std::string& spots_path,
        const std::string& out_prefix, bool allocation_only,
        const std::vector<double>& world_planes, double scale_override) {
    carbon::TransportConfig config = carbon::load_config(config_path);
    config.tps_spots_file = spots_path;
    if (scale_override > 0.0) config.tps_histories_scale = scale_override;
    carbon::TpsSourcePlan plan = carbon::TpsSourcePlan::from_config(config);
    std::vector<carbon::PrimarySpotBatchEntry> batch =
        plan.make_primary_batch(config);

    if (allocation_only) {
        // Mirrors TpsSourcePlan::make_primary_batch histories-mode rounding.
        std::ofstream out(out_prefix + "_allocation.csv");
        out << "spot_index,spot_id,histories\n";
        std::size_t kept = 0;
        for (std::size_t i = 0; i < plan.spots.size(); ++i) {
            const auto& s = plan.spots[i];
            std::size_t n = 0;
            if (s.mu_weight > 0.0) {
                const double scaled = s.mu_weight * config.tps_histories_scale;
                n = static_cast<std::size_t>(std::llround(scaled));
                if (n == 0 && scaled > 0.0) n = 1;
            }
            if (n == 0) continue;
            out << i << "," << s.spot_id << "," << n << "\n";
            ++kept;
        }
        std::cout << "spots: " << kept << "/" << plan.spots.size() << "\n";
        if (kept != batch.size()) {
            std::cerr << "allocation mismatch vs make_primary_batch\n";
            return 1;
        }
        return 0;
    }

    const bool enable_emittance =
        std::any_of(batch.begin(), batch.end(), [](const auto& e) {
            return e.emittance_sigma_x_mm() > 0.0F ||
                   e.emittance_sigma_y_mm() > 0.0F ||
                   e.emittance_sigma_x_prime() > 0.0F ||
                   e.emittance_sigma_y_prime() > 0.0F;
        });

    const carbon::CtGrid grid = carbon::CtGrid::from_config(config);
    const double transport_z_shift_mm = -static_cast<double>(grid.origin_z_mm);
    const float phantom_length = static_cast<float>(config.phantom_length_mm);
    const float cutoff = static_cast<float>(config.energy_cutoff_MeV);
    float voxel_min_x = -0.5F * static_cast<float>(config.voxel_bins_x) *
                        static_cast<float>(config.voxel_size_x_mm);
    float voxel_max_x =
        voxel_min_x + static_cast<float>(config.voxel_bins_x) *
                          static_cast<float>(config.voxel_size_x_mm);
    float voxel_min_y = -0.5F * static_cast<float>(config.voxel_bins_y) *
                        static_cast<float>(config.voxel_size_y_mm);
    float voxel_max_y =
        voxel_min_y + static_cast<float>(config.voxel_bins_y) *
                          static_cast<float>(config.voxel_size_y_mm);
    if (config.enable_ct_grid &&
        std::abs(grid.spacing_x_mm - static_cast<float>(config.voxel_size_x_mm)) <
            1.0e-5F &&
        std::abs(grid.spacing_y_mm - static_cast<float>(config.voxel_size_y_mm)) <
            1.0e-5F) {
        voxel_min_x = grid.origin_x_mm;
        voxel_min_y = grid.origin_y_mm;
        voxel_max_x = voxel_min_x + static_cast<float>(config.voxel_bins_x) *
                                          static_cast<float>(config.voxel_size_x_mm);
        voxel_max_y = voxel_min_y + static_cast<float>(config.voxel_bins_y) *
                                          static_cast<float>(config.voxel_size_y_mm);
    }

    // Inverse map setup. By construction the transported beam always travels
    // +CT-Z, and +CT-Z is -patient-X in the tps_90 packing, so the -X shuffle
    // branch applies to every entry. Verify per entry below.
    InverseTps90 inv;
    inv.tx = config.spots_patient_trans_x_mm;
    inv.ty = config.spots_patient_trans_y_mm;
    inv.tz = config.spots_patient_trans_z_mm;
    {
        const double ang =
            config.spots_patient_rot_z_deg * 3.14159265358979323846 / 180.0;
        inv.c = std::cos(ang);
        inv.s = std::sin(ang);
    }
    inv.ct_axis_min_mm = config.spots_ct_axis_min_mm;
    inv.minus_x_branch = true;
    std::cout << "shuffle branch: -X (by beam-packing construction)\n";

    // Round-trip self-test: inv(fwd(P)) == P for beam-like world poses
    // (direction +Y takes the -X shuffle branch, as all plan entries do).
    for (std::size_t bi = 0; bi < batch.size(); ++bi) {
        carbon::SpotSourcePose world_pose{};
        world_pose.origin_x_mm = 100.0 + 3.0 * bi;
        world_pose.origin_y_mm = -200.0 - bi;
        world_pose.origin_z_mm = 50.0;
        world_pose.ux_x = 1; world_pose.uy_z = 1; world_pose.uz_y = 1;
        const carbon::SpotSourcePose fwd =
            carbon::transform_tps_90_pose_to_ct(world_pose, inv.tx, inv.ty, inv.tz,
                                                config.spots_patient_rot_z_deg,
                                                inv.ct_axis_min_mm);
        double qx, qy, qz, rx, ry, rz;
        inv.to_patient(fwd.origin_x_mm, fwd.origin_y_mm, fwd.origin_z_mm, false,
                       qx, qy, qz);
        inv.patient_to_world(qx, qy, qz, rx, ry, rz);
        const double err = std::abs(rx - world_pose.origin_x_mm) +
                           std::abs(ry - world_pose.origin_y_mm) +
                           std::abs(rz - world_pose.origin_z_mm);
        if (!(err < 1.0e-9)) {
            std::cerr << "inverse map round-trip failed: " << err << "\n";
            return 1;
        }
    }
    std::cout << "inverse map round-trip OK (" << batch.size() << " poses)\n";

    std::ofstream states(out_prefix + "_states.csv");
    states << "spot_id,hist_in_spot,x_mm,y_mm,z_mm,dx,dy,dz,E_MeV,hit\n";
    std::ofstream summary(out_prefix + "_summary.json");
    summary << "[\n";
    struct PlaneAcc {
        double y0{0};
        std::ofstream* out{nullptr};
        double n{0}, sx{0}, sy{0}, sz{0}, sdx{0}, sdy{0}, sdz{0}, se{0};
        double sxx{0}, szz{0}, stx{0}, sty{0}, stxtx{0}, stzty{0}, see{0};
        double sxtx{0}, szty{0};
    };
    std::vector<PlaneAcc> planes;
    std::vector<std::ofstream> plane_files;
    plane_files.reserve(world_planes.size());
    for (const double y0 : world_planes) {
        plane_files.emplace_back(out_prefix + "_planeY" +
                                 std::to_string(static_cast<int>(y0)) + ".csv");
        plane_files.back()
            << "spot_id,hist_in_spot,x_mm,y_mm,z_mm,dx,dy,dz,E_MeV\n";
        planes.push_back(PlaneAcc{y0, &plane_files.back()});
    }

    auto intersect_slab = [](float pos, float dir, float lo, float hi, float& t_enter,
                             float& t_exit, bool& hit) {
        if (std::fabs(dir) < 1.0e-8F) {
            if (pos < lo || pos >= hi) hit = false;
            return;
        }
        float first = (lo - pos) / dir;
        float second = (hi - pos) / dir;
        if (first > second) std::swap(first, second);
        t_enter = std::fmax(t_enter, first);
        t_exit = std::fmin(t_exit, second);
        if (t_exit < t_enter) hit = false;
    };

    for (std::size_t bi = 0; bi < batch.size(); ++bi) {
        const auto& spot = batch[bi];
        // batch preserves plan order with zero-allocation spots skipped
        std::size_t pi = 0, seen = 0;
        for (; pi < plan.spots.size(); ++pi) {
            const auto& s = plan.spots[pi];
            std::size_t n = 0;
            if (s.mu_weight > 0.0) {
                n = static_cast<std::size_t>(
                    std::llround(s.mu_weight * config.tps_histories_scale));
                if (n == 0) n = 1;
            }
            if (n == 0) continue;
            if (seen++ == bi) break;
        }
        const int spot_id = plan.spots[pi].spot_id;
        const std::uint64_t spot_seed = spot.random_seed;
        const std::size_t count =
            static_cast<std::size_t>(spot.history_end - spot.history_begin);
        // Accumulators in world frame.
        double n = 0, sx = 0, sy = 0, sz = 0, sdx = 0, sdy = 0, sdz = 0, se = 0;
        double sxx = 0, szz = 0, stx = 0, sty = 0, stxtx = 0, stzty = 0, see = 0;
        double sxtx = 0, szty = 0;
        std::size_t hits = 0;
        for (std::size_t h = 0; h < count; ++h) {
            const auto rng_history = static_cast<std::uint64_t>(h);
            float energy = spot.initial_energy_MeV();
            const float spread = spot.beam_energy_spread();
            if (spread > 0.0F) {
                const float u0 = std::fmax(
                    carbon::rng::uniform01(spot_seed, rng_history, 0, 40), 1.0e-12F);
                const float u1 =
                    carbon::rng::uniform01(spot_seed, rng_history, 0, 41);
                const float gauss =
                    std::sqrt(-2.0F * std::log(u0)) * std::cos(kTwoPi * u1);
                energy = spot.initial_energy_MeV() * (1.0F + spread * gauss);
                if (energy < cutoff) energy = cutoff;
            }
            float lx = 0, ly = 0, ldx = 0, ldy = 0, ldz = 1;
            if (enable_emittance) {
                const float u0 = std::fmax(
                    carbon::rng::uniform01(spot_seed, rng_history, 0, 30), 1.0e-12F);
                const float u1 =
                    carbon::rng::uniform01(spot_seed, rng_history, 0, 31);
                const float u2 = std::fmax(
                    carbon::rng::uniform01(spot_seed, rng_history, 0, 32), 1.0e-12F);
                const float u3 =
                    carbon::rng::uniform01(spot_seed, rng_history, 0, 33);
                const float g0 =
                    std::sqrt(-2.0F * std::log(u0)) * std::cos(kTwoPi * u1);
                const float g1 =
                    std::sqrt(-2.0F * std::log(u0)) * std::sin(kTwoPi * u1);
                const float g2 =
                    std::sqrt(-2.0F * std::log(u2)) * std::cos(kTwoPi * u3);
                const float g3 =
                    std::sqrt(-2.0F * std::log(u2)) * std::sin(kTwoPi * u3);
                const float sxm = spot.emittance_sigma_x_mm();
                const float sym = spot.emittance_sigma_y_mm();
                const float sxp = spot.emittance_sigma_x_prime();
                const float syp = spot.emittance_sigma_y_prime();
                lx = sxm * g0;
                ly = sym * g2;
                const float rhox =
                    std::clamp(spot.emittance_correlation_x(), -0.9999F, 0.9999F);
                const float rhoy =
                    std::clamp(spot.emittance_correlation_y(), -0.9999F, 0.9999F);
                const float xp =
                    sxp * (rhox * g0 + std::sqrt(1.0F - rhox * rhox) * g1);
                const float yp =
                    syp * (rhoy * g2 + std::sqrt(1.0F - rhoy * rhoy) * g3);
                const float inv_norm = 1.0F / std::sqrt(1.0F + xp * xp + yp * yp);
                ldx = xp * inv_norm;
                ldy = yp * inv_norm;
                ldz = inv_norm;
            }
            float pxm = spot.source_origin_x_mm() + spot.beam_ux_x() * lx +
                        spot.beam_uy_x() * ly;
            float pym = spot.source_origin_y_mm() + spot.beam_ux_y() * lx +
                        spot.beam_uy_y() * ly;
            float pzm = spot.source_origin_z_mm() + spot.beam_ux_z() * lx +
                        spot.beam_uy_z() * ly;
            float dx = spot.beam_ux_x() * ldx + spot.beam_uy_x() * ldy +
                       spot.beam_uz_x() * ldz;
            float dy = spot.beam_ux_y() * ldx + spot.beam_uy_y() * ldy +
                       spot.beam_uz_y() * ldz;
            float dz = spot.beam_ux_z() * ldx + spot.beam_uy_z() * ldy +
                       spot.beam_uz_z() * ldz;
            {
                const float inv_n =
                    1.0F / std::sqrt(std::fmax(1.0e-20F, dx * dx + dy * dy + dz * dz));
                dx *= inv_n;
                dy *= inv_n;
                dz *= inv_n;
            }
            float t_enter = 0.0F, t_exit = 1.0e30F;
            bool hit = true;
            const float src_x = pxm, src_y = pym, src_z = pzm;
            if (config.enable_voxel_scoring) {
                intersect_slab(pxm, dx, voxel_min_x, voxel_max_x, t_enter, t_exit,
                               hit);
                intersect_slab(pym, dy, voxel_min_y, voxel_max_y, t_enter, t_exit,
                               hit);
            }
            intersect_slab(pzm, dz, 0.0F, phantom_length, t_enter, t_exit, hit);
            if (hit && t_exit >= t_enter) {
                const float entry = t_enter + 1.0e-4F;
                pxm += entry * dx;
                pym += entry * dy;
                pzm += entry * dz;
            }
            // transport -> CT -> patient -> world (shared for pre/post-clip)
            auto to_world = [&](float qx, float qy, float qz, float qdx, float qdy,
                                float qdz, double& wx, double& wy, double& wz,
                                double& wdx, double& wdy, double& wdz) {
                const double ct_x = qx, ct_y = qy,
                             ct_z = static_cast<double>(qz) - transport_z_shift_mm;
                double ox, oy, oz;
                inv.to_patient(ct_x, ct_y, ct_z, false, ox, oy, oz);
                inv.patient_to_world(ox, oy, oz, wx, wy, wz);
                const double pdx = -static_cast<double>(qdz);
                const double pdy = static_cast<double>(qdx);
                const double pdz = static_cast<double>(qdy);
                wdx = inv.c * pdx + inv.s * pdy;
                wdy = -inv.s * pdx + inv.c * pdy;
                wdz = pdz;
            };
            double sx0, sy0, sz0, sdx0, sdy0, sdz0;
            to_world(src_x, src_y, src_z, dx, dy, dz, sx0, sy0, sz0, sdx0, sdy0,
                     sdz0);
            for (auto& pl : planes) {
                const double t = (pl.y0 - sy0) / sdy0;
                const double qx = sx0 + t * sdx0, qz = sz0 + t * sdz0;
                *pl.out << spot_id << "," << h << "," << qx << "," << pl.y0 << ","
                        << qz << "," << sdx0 << "," << sdy0 << "," << sdz0 << ","
                        << energy << "\n";
                pl.n += 1;
                pl.sx += qx;
                pl.sy += pl.y0;
                pl.sz += qz;
                pl.sdx += sdx0;
                pl.sdy += sdy0;
                pl.sdz += sdz0;
                pl.se += energy;
                pl.sxx += qx * qx;
                pl.szz += qz * qz;
                const double txa = sdx0 / sdy0, tya = sdz0 / sdy0;
                pl.stx += txa;
                pl.sty += tya;
                pl.stxtx += txa * txa;
                pl.stzty += tya * tya;
                pl.see += static_cast<double>(energy) * energy;
                pl.sxtx += qx * txa;
                pl.szty += qz * tya;
            }
            if (!(static_cast<double>(dz) > 0.0)) {
                std::cerr << "entry beam does not travel +CT-Z\n";
                return 1;
            }
            double wx, wy, wz, wdx, wdy, wdz;
            to_world(pxm, pym, pzm, dx, dy, dz, wx, wy, wz, wdx, wdy, wdz);
            states << spot_id << "," << h << "," << wx << "," << wy
                   << "," << wz << "," << wdx << "," << wdy << "," << wdz << ","
                   << energy << "," << (hit ? 1 : 0) << "\n";
            n += 1;
            if (hit) ++hits;
            sx += wx;
            sy += wy;
            sz += wz;
            sdx += wdx;
            sdy += wdy;
            sdz += wdz;
            se += energy;
            sxx += wx * wx;
            szz += wz * wz;
            const double txa = wdx / wdy, tya = wdz / wdy;
            stx += txa;
            sty += tya;
            stxtx += txa * txa;
            stzty += tya * tya;
            see += static_cast<double>(energy) * energy;
            sxtx += wx * txa;
            szty += wz * tya;
        }
        const double e_mean = se / n, e_var = std::max(0.0, see / n - e_mean * e_mean);
        const double x_mean = sx / n, x_var = std::max(0.0, sxx / n - x_mean * x_mean);
        const double z_mean = sz / n, z_var = std::max(0.0, szz / n - z_mean * z_mean);
        const double tx_mean = stx / n, tx_var = std::max(0.0, stxtx / n - tx_mean * tx_mean);
        const double ty_mean = sty / n, ty_var = std::max(0.0, stzty / n - ty_mean * ty_mean);
        const double cov_x_tx = sxtx / n - x_mean * tx_mean;
        const double cov_z_ty = szty / n - z_mean * ty_mean;
        const double corr_x_tx =
            (x_var > 0 && tx_var > 0) ? cov_x_tx / std::sqrt(x_var * tx_var) : 0.0;
        const double corr_z_ty =
            (z_var > 0 && ty_var > 0) ? cov_z_ty / std::sqrt(z_var * ty_var) : 0.0;
        char buf[2048];
        std::snprintf(buf, sizeof(buf),
                      "  {\"spot_id\": %d, \"histories\": %zu, \"hits\": %zu,\n"
                      "   \"E_mean_MeV\": %.6f, \"E_std_MeV\": %.6f,\n"
                      "   \"x_mean\": %.6f, \"x_std\": %.6f, \"z_mean\": %.6f, "
                      "\"z_std\": %.6f,\n"
                      "   \"tx_mean\": %.8f, \"tx_std\": %.8f, \"ty_mean\": %.8f, "
                      "\"ty_std\": %.8f,\n"
                      "   \"corr_x_tx\": %.6f, \"corr_z_ty\": %.6f,\n"
                      "   \"dir_mean\": [%.8f, %.8f, %.8f]}%s\n",
                      spot_id, count, hits, e_mean, std::sqrt(e_var),
                      x_mean, std::sqrt(x_var), z_mean, std::sqrt(z_var),
                      tx_mean, std::sqrt(tx_var), ty_mean, std::sqrt(ty_var),
                      corr_x_tx, corr_z_ty,
                      sdx / n, sdy / n, sdz / n,
                      (!planes.empty() || bi + 1 < batch.size() ? "," : ""));
        summary << buf;
        for (std::size_t pli = 0; pli < planes.size(); ++pli) {
            const auto& pl = planes[pli];
            const double pe = pl.se / pl.n;
            const double pxm_ = pl.sx / pl.n, pxv = std::max(0.0, pl.sxx / pl.n - pxm_ * pxm_);
            const double pzm_ = pl.sz / pl.n, pzv = std::max(0.0, pl.szz / pl.n - pzm_ * pzm_);
            const double ptx = pl.stx / pl.n, ptxv = std::max(0.0, pl.stxtx / pl.n - ptx * ptx);
            const double pty = pl.sty / pl.n, ptyv = std::max(0.0, pl.stzty / pl.n - pty * pty);
            char pb[1024];
            std::snprintf(pb, sizeof(pb),
                          "  {\"spot_id\": %d, \"plane_y\": %.1f, \"n\": %.0f, "
                          "\"E_mean\": %.4f, \"E_std\": %.4f, \"x_mean\": %.5f, "
                          "\"x_std\": %.5f, \"z_mean\": %.5f, \"z_std\": %.5f, "
                          "\"tx_mean\": %.7f, \"tx_std\": %.7f, \"ty_mean\": %.7f, "
                          "\"ty_std\": %.7f}%s\n",
                          spot_id, pl.y0, pl.n, pe,
                          std::sqrt(std::max(0.0, pl.see / pl.n - pe * pe)),
                          pxm_, std::sqrt(pxv), pzm_, std::sqrt(pzv), ptx,
                          std::sqrt(ptxv), pty, std::sqrt(ptyv),
                          (pli + 1 < planes.size() || bi + 1 < batch.size() ? ","
                                                                            : ""));
            summary << pb;
        }
    }
    summary << "]\n";
    std::cout << "done\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: dump_tps_phase_space config.yaml spots.csv out_prefix "
                     "[--allocation-only] [--plane=Y0[,Y1...]] [--scale=S]\n";
        return 2;
    }
    bool allocation_only = false;
    std::vector<double> planes;
    double scale_override = 0.0;
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--allocation-only") {
            allocation_only = true;
        } else if (a.rfind("--scale=", 0) == 0) {
            scale_override = std::stod(a.substr(8));
        } else if (a.rfind("--plane=", 0) == 0) {
            std::string list = a.substr(8);
            std::size_t pos = 0;
            while (pos < list.size()) {
                auto comma = list.find(',', pos);
                planes.push_back(std::stod(list.substr(
                    pos, comma == std::string::npos ? comma : comma - pos)));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } else {
            std::cerr << "unknown arg: " << a << "\n";
            return 2;
        }
    }
    try {
        return run(argv[1], argv[2], argv[3], allocation_only, planes,
                   scale_override);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

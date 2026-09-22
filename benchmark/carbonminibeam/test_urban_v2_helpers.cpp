// Production regression for the urban_v2 shared helpers (Fix A/B).
//
// Calls the SAME implementation as production: src/detail/sycl_device_math.inc
// (copper_urban_v2_* helpers, copper_urban_v2_msc_step) and
// include/carbon/minibeam_collimator.hpp. No algorithm is re-implemented
// here; the double-precision numbers below are the independent reference.
//
// PASS -> exit 0.  FAIL -> exit nonzero (no "FAIL reproduced but exit 0").
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "carbon/minibeam_collimator.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/rng.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/water_electron_response.hpp"

namespace carbon {
namespace {
#include "detail/sycl_device_math.inc"
}  // namespace
}  // namespace carbon

namespace carbon {

namespace {

int failures = 0;

void test_propose_path();
void test_water_material();
void test_water_urban_segment_scaling();
void test_water_urban_low_energy_proposal();
void test_water_urban_subdivision_robustness();
void test_water_xsec_table();
void test_delta_relocation();

void check(bool cond, const char* label, double a = 0.0, double b = 0.0) {
    if (cond) {
        std::printf("PASS: %s\n", label);
    } else {
        ++failures;
        std::printf("FAIL: %s  (got %g want %g)\n", label, a, b);
    }
}

char label_buf[256];

void test_acceptance_gate() {
    // Exact G4VMultipleScattering gate: geomMin = 5e-8 mm,
    // minDisplacement2 = 2.5e-15 mm^2.
    float ax = 0.0F, ay = 0.0F;
    int branch = -1;
    // 1. raw_r < geomMin -> no position change (r2 <= minDisplacement2).
    copper_urban_v2_accept_displacement(1.0e-9F, 0.0F, 1.0F, ax, ay, branch);
    check(branch == 1 && ax == 0.0F && ay == 0.0F, "accept: raw_r<geomMin -> below-min, no change");
    // 2. safety = 0 -> cancel.
    copper_urban_v2_accept_displacement(1.0e-4F, 0.0F, 0.0F, ax, ay, branch);
    check(branch == 4 && ax == 0.0F && ay == 0.0F, "accept: safety=0 -> cancel");
    // 3. 0 < postSafety < raw_r, postSafety > geomMin -> reduce.
    copper_urban_v2_accept_displacement(1.0e-4F, 0.0F, 1.0e-5F, ax, ay, branch);
    const double acc3 = std::sqrt(double(ax) * ax + double(ay) * ay);
    check(branch == 3 && std::fabs(acc3 - 9.9e-6) < 1.0e-12, "accept: 0<post<raw -> reduce to postSafety",
          acc3, 9.9e-6);
    // 4. Far-field safety would fully accept WITHOUT the dispR cap, but
    // G4SafetyHelper::ComputeSafety(fNewPosition, dispR) caps at dispR, so
    // the reference scales every accepted displacement by 0.99 (fix B2):
    // raw=1e-4, safety=1.0 -> reduce to 0.99e-4, branch 3, not branch 2.
    copper_urban_v2_accept_displacement(1.0e-4F, 0.0F, 1.0F, ax, ay, branch);
    const double acc4 = std::sqrt(double(ax) * ax + double(ay) * ay);
    check(branch == 3 && std::fabs(acc4 - 9.9e-5) < 1.0e-10,
          "accept: far-field safety -> 0.99 reduce (dispR cap)", acc4,
          9.9e-5);
    // 5. constants are exactly the G4 reference, not placeholders.
    check(kUrbanV2GeomMinMm == 5.0e-8F, "accept: geomMin == 0.05nm");
    check(kUrbanV2MinDisplacement2Mm2 == 2.5e-15, "accept: minDisplacement2 == geomMin^2",
          kUrbanV2MinDisplacement2Mm2, 2.5e-15);
}

void test_stable_delta() {
    // C12, Cu, 250 MeV/u: total kinetic 3000 MeV. lambda0 from the SHARED
    // cross-section helper; reference delta in double (stable log1p form)
    // evaluated at the SAME float32 lambda0 the helper receives, so the
    // check isolates the delta algorithm (not the ~6e-8 float32 input
    // quantization of lambda0 itself, which is documented and negligible
    // against the pre-fix 88%/sign-flip error).
    const float lambda0 = copper_urban_transport_mfp_mm(3000.0F, 6, 12, 8.96F);
    check(lambda0 > 1.0e6F && lambda0 < 1.0e7F, "delta: shared lambda0 sane", lambda0, 1.97e6);
    const double lam = double(lambda0);
    for (const double g : {0.03, 0.05, 0.10, 0.25}) {
        const auto got = copper_urban_v2_true_path_and_delta(float(g), lambda0, 1.0e10F);
        const double r = g / lam;
        const double ref = -lam * (std::log1p(-r) + r);
        const double rel =
            std::fabs(double(got.delta_mm) - ref) / ref;
        std::snprintf(label_buf, sizeof(label_buf),
                      "delta: g=%.2f stable vs double (rel=%g)", g, rel);
        // 2e-7 admits the float32 OUTPUT quantization of delta itself
        // (up to 6e-8); the pre-fix error was cancellation to 0, sign
        // flip, and +88% at g=0.25.
        check(rel < 2.0e-7, label_buf, rel, 2.0e-7);
        check(got.delta_mm > 0.0F, "delta: strictly positive", got.delta_mm, 0.0);
        // rmax exactly as the device forms it (float ops) vs double.
        const float t = got.true_path_mm, d = got.delta_mm;
        const float rmax2_f = d * (2.0F * t - d);
        const double ref_t = g + ref;
        const double rmax2_ref = ref * (2.0 * ref_t - ref);
        const double rrel =
            std::fabs(double(rmax2_f) - rmax2_ref) / rmax2_ref;
        std::snprintf(label_buf, sizeof(label_buf),
                      "rmax2: g=%.2f device-float vs double (rel=%g)", g, rrel);
        check(rrel < 1.0e-6, label_buf, rrel, 1.0e-6);
    }
}

void test_slit_navigation_and_safety() {
    const float cos_a = 1.0F, sin_a = 0.0F, radius = 60.0F;
    const int slits = 15;
    const float w = 0.5F, p = 3.6F, hl = 25.0F;
    int s = -99;
    check(minibeam_point_in_slit(0.0F, 0.0F, cos_a, sin_a, radius, slits, w, p,
                                 hl, 0.05F, s) &&
              s == 0,
          "slit: x=0 inside +0.05mm-offset slit");
    int si = -99, so = -99;
    check(minibeam_point_in_slit(0.249F, 0.0F, cos_a, sin_a, radius, slits, w,
                                 p, hl, 0.0F, si) &&
              !minibeam_point_in_slit(0.251F, 0.0F, cos_a, sin_a, radius, slits,
                                      w, p, hl, 0.0F, so),
          "slit: 0.249 in / 0.251 out (d=x-0.25)");
    const float s_air = minibeam_isotropic_safety_at_point(
        0.0F, 0.0F, 0.0F, cos_a, sin_a, radius, slits, w, p, hl, 0.0F, -30.0F,
        30.0F);
    const float s_cu = minibeam_isotropic_safety_at_point(
        0.30F, 0.0F, 0.0F, cos_a, sin_a, radius, slits, w, p, hl, 0.0F, -30.0F,
        30.0F);
    check(std::fabs(s_air - 0.25F) < 1.0e-5F, "safety: slit air == 0.25", s_air, 0.25);
    check(std::fabs(s_cu - 0.05F) < 1.0e-5F, "safety: copper == 0.05", s_cu, 0.05);
}

void test_angular_small_quantity() {
    // C12 Cu 250 MeV/u, production g values. Calls the SHARED full sampler
    // (dedx=0 isolates the angular FP32 path; safety=0.25 is slit-scale).
    // Reports cth==1 fraction and checks the omcth-derived direction stays
    // sane; asserts determinism (RNG contract).
    const Direction3F dir{0.0F, 0.0F, 1.0F};
    for (const float g : {0.03F, 0.05F, 0.10F, 0.25F}) {
        constexpr int n = 20000;
        long cth_one = 0, bad = 0, zero_raw = 0;
        long hist[5] = {0, 0, 0, 0, 0};
        std::vector<double> theta;
        theta.reserve(n);
        double dir_norm_err = 0.0;
        for (int i = 0; i < n; ++i) {
            const auto s = copper_urban_v2_msc_step(
                dir, 3000.0F, 6, 12, g, 8.96F, 12.8628F, 1.0e10F, 0.0F, 0.25F,
                1.0F, 777ULL, uint64_t(i), uint64_t(1000 + i), 50);
            if (s.cth_rounded_to_one) ++cth_one;
            if (!(s.one_minus_cth >= 0.0F) || !(s.one_minus_cth <= 2.0F)) ++bad;
            if (s.displacement_branch < 0 || s.displacement_branch > 4) ++bad;
            ++hist[s.displacement_branch < 0 || s.displacement_branch > 4
                       ? 0
                       : s.displacement_branch];
            if (s.displacement_branch == 0) ++zero_raw;
            theta.push_back(std::sqrt(2.0 * double(s.one_minus_cth)));
            const double dn = std::sqrt(double(s.direction.x) * s.direction.x +
                                        double(s.direction.y) * s.direction.y +
                                        double(s.direction.z) * s.direction.z);
            dir_norm_err = std::fmax(dir_norm_err, std::fabs(dn - 1.0));
            if (!(dn == dn)) ++bad;  // NaN
        }
        // Determinism: identical call -> identical result.
        const auto a = copper_urban_v2_msc_step(dir, 3000.0F, 6, 12, g, 8.96F,
                                                12.8628F, 1.0e10F, 0.0F, 0.25F,
                                                1.0F, 777ULL, 7ULL, 1007ULL, 50);
        const auto b = copper_urban_v2_msc_step(dir, 3000.0F, 6, 12, g, 8.96F,
                                                12.8628F, 1.0e10F, 0.0F, 0.25F,
                                                1.0F, 777ULL, 7ULL, 1007ULL, 50);
        check(a.direction.x == b.direction.x && a.direction.y == b.direction.y &&
                  a.direction.z == b.direction.z &&
                  a.one_minus_cth == b.one_minus_cth &&
                  a.displacement_mm.x == b.displacement_mm.x,
              "angular: sampler deterministic");
        // Median space-angle vs Highland theta0: Rayleigh core gives
        // median ~= 1.177*theta0 (same band the python oracle asserts,
        // 1.15-1.35). Median is used because the Rutherford tail dominates
        // the MEAN (<1-cos> was ~3.3x theta0^2/2, tail-driven, not an FP32
        // bug). nth_element for the median.
        std::nth_element(theta.begin(), theta.begin() + n / 2, theta.end());
        const double med = theta[n / 2];
        const float th0 = copper_urban_theta0(
            g, 3000.0F, 10.0F * 12.8628F / 8.96F, 6, 12,
            copper_urban_coefficients(29.0F));
        const double ratio = med / double(th0);
        std::snprintf(label_buf, sizeof(label_buf),
                      "angular: g=%.2f cth==1 frac=%.3f med(theta)/th0=%.3f "
                      "zero_raw=%ld bad=%ld dirnormerr=%g",
                      double(g), double(cth_one) / n, ratio, zero_raw, bad,
                      dir_norm_err);
        check(bad == 0 && dir_norm_err < 1.0e-5 && ratio > 1.1 && ratio < 1.4,
              label_buf, ratio, 1.177);
        std::printf("  branch histogram g=%.2f: zero=%ld below=%ld accept=%ld "
                    "reduce=%ld cancel=%ld\n",
                    double(g), hist[0], hist[1], hist[2], hist[3], hist[4]);
    }
}

void test_loss_table_and_limiter() {
    // Loads the REAL extraction CSV (fails honest if missing) and checks
    // spot values against the TOPAS ntuple. Then drives the shared limiter
    // + full propose_and_sample path (production code, host-called).
    FILE* f = std::fopen(
        "data/urban/c12_copper_loss_range_g4_11_3_2.csv", "r");
    check(f != nullptr, "loss table: extraction CSV present");
    if (f == nullptr) return;
    std::fclose(f);
    // Minimal CSV read here (the carbon_core loader is covered by the
    // production config path); spot-check R(250MeV/u)=22.314mm.
    double r250 = 0.0;
    {
        char line[256];
        FILE* g = std::fopen(
            "data/urban/c12_copper_loss_range_g4_11_3_2.csv", "r");
        while (std::fgets(line, sizeof(line), g)) {
            if (line[0] == '#' || (line[0] >= 'a' && line[0] <= 'z') ||
                (line[0] >= 'A' && line[0] <= 'Z'))
                continue;
            double eu, et, r, d, res;
            if (std::sscanf(line, "%lf,%lf,%lf,%lf,%lf", &eu, &et, &r, &d,
                             &res) == 5 &&
                std::fabs(eu - 250.01) < 0.06) {
                r250 = r;
            }
        }
        std::fclose(g);
    }
    check(std::fabs(r250 - 22.314) < 0.01, "loss table: R(250MeV/u)=22.3mm",
          r250, 22.314);
    test_propose_path();
    test_water_material();
    test_water_urban_segment_scaling();
    test_water_urban_low_energy_proposal();
    test_water_urban_subdivision_robustness();
    test_water_xsec_table();
    test_delta_relocation();
}

void test_water_xsec_table() {
    // Prints GPU H/O cross sections + Bragg mfp at fixed energies for
    // diffing against urban_water_xsec_oracle.py (independent port).
    // Asserts positivity + H-extrapolation/O-interpolation branch sanity.
    for (const float eu : {0.1F, 0.2F, 0.5F, 1.0F, 10.0F, 50.0F, 250.0F}) {
        const float tot = eu * 12.0F;
        const float sh = urban_cross_section_per_atom_cm2(tot, 6, 12, 1.0F);
        const float so = urban_cross_section_per_atom_cm2(tot, 6, 12, 8.0F);
        const float mfp = urban_water_transport_mfp_mm(tot, 6, 12);
        char lab[192];
        std::snprintf(lab, sizeof(lab),
                      "xsec: E/u=%.1f H=%.6e O=%.6e mfp=%.4fmm", double(eu),
                      double(sh), double(so), double(mfp));
        check(sh > 0.0F && so > 0.0F && mfp > 0.0F && so > sh, lab, so,
              sh);
    }
}

void test_propose_path() {
    // Full production step (propose_and_sample) with the REAL loss table
    // and slit-block geometry ctx, host-called. C12 250MeV/u, Cu slab
    // interior point, both external ceilings.
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_copper_loss_range_g4_11_3_2.csv");
    } catch (...) {
        check(false, "propose: loss table loads via carbon_core");
        return;
    }
    check(host.ranges_mm().size() == 4001, "propose: 4001 nodes",
          double(host.ranges_mm().size()), 4001.0);
    std::vector<float> e(host.energies_total_mev().begin(),
                         host.energies_total_mev().end());
    std::vector<float> r(host.ranges_mm().begin(), host.ranges_mm().end());
    std::vector<float> d(host.dedx_values().begin(), host.dedx_values().end());
    UrbanV2LossTable table{e.data(), r.data(), d.data(),
                           static_cast<int>(e.size())};
    UrbanV2GeomCtx geom{};
    geom.cos_a = 1.0F;
    geom.sin_a = 0.0F;
    geom.radius_mm = 60.0F;
    geom.slit_count = 1;
    geom.slit_width_mm = 0.001F;
    geom.slit_pitch_mm = 3.6F;
    geom.slit_half_len_mm = 0.5F;
    geom.slit_offset_mm = 100.0F;
    geom.block_entrance_z_mm = -1.0F;
    geom.block_exit_z_mm = 0.0F;
    const Direction3F dir{0.0F, 0.0F, 1.0F};
    // (ceiling, boundary, expected reason): truncation by the 0.05 boundary
    // must report geometry(3); matched ceiling reports user(1).
    for (const auto [ceiling, boundary, want] :
         {std::tuple{0.25F, 0.05F, 3}, std::tuple{0.05F, 0.05F, 1}}) {
        UrbanV2TrackState state{};
        const auto s = copper_urban_v2_propose_and_sample(
            dir, 3000.0F, 6, 12, ceiling, boundary, 0.0F, 0.0F, -0.5F, 0.0F,
            0.0F, 1.0F, geom, true, state, table, 8.96F, 12.8628F, 1.0F,
            4242ULL, 17ULL, 3ULL, 50);
        char lab[160];
        std::snprintf(lab, sizeof(lab),
                      "propose: ceiling=%.2f valid reason=%d g=%.5f t=%.5f",
                      double(ceiling), s.limit_reason,
                      double(s.final_geom_path_mm),
                      double(s.final_true_path_mm));
        // Interior Cu step: scatters (reason 1 or 2), g<=ceiling, t>=g,
        // state consumed first_step, deterministic on repeat.
        const auto s2 = copper_urban_v2_propose_and_sample(
            dir, 3000.0F, 6, 12, ceiling, boundary, 0.0F, 0.0F, -0.5F, 0.0F,
            0.0F, 1.0F, geom, true, state, table, 8.96F, 12.8628F, 1.0F,
            4242ULL, 17ULL, 3ULL, 50);
        UrbanV2TrackState state0{};
        const auto s0 = copper_urban_v2_propose_and_sample(
            dir, 3000.0F, 6, 12, ceiling, boundary, 0.0F, 0.0F, -0.5F, 0.0F,
            0.0F, 1.0F, geom, true, state0, table, 8.96F, 12.8628F, 1.0F,
            4242ULL, 17ULL, 3ULL, 50);
        check(s.proposal_valid && s0.proposal_valid &&
                  s.limit_reason == want &&
                  s.final_geom_path_mm <= ceiling &&
                  s.final_true_path_mm >= s.final_geom_path_mm &&
                  !state.first_step &&
                  s.direction.x == s0.direction.x &&
                  s.displacement_mm.x == s0.displacement_mm.x,
              lab, double(s.limit_reason), 1.0);
        (void)s2;
    }
}

void test_water_material() {
    // Water_75eV couple: Bragg-sum mfp sane, cache from extracted Zeff,
    // box safety analytic, full propose smoke with the real water table.
    const float lam_hi = urban_water_transport_mfp_mm(3000.0F, 6, 12);
    const float lam_lo = urban_water_transport_mfp_mm(120.0F, 6, 12);
    // Urban transport mfp is enormous by construction (validated Cu value at
    // 250MeV/u is 1.97e6mm; water is ~30x larger by Z^2/volume scaling).
    // Assert only positivity + correct energy trend here.
    check(lam_hi > lam_lo && lam_lo > 0.0F, "water mfp: positive, falls",
          lam_lo, lam_hi);
    std::printf("  water mfp: 250MeV/u=%.3fmm 10MeV/u=%.4fmm\n",
                double(lam_hi), double(lam_lo));
    const auto cache = urban_v2_material_cache(3.3334F);
    check(cache.doverrb > 1.1F && cache.doverrb < 1.2F, "water cache doverrb",
          cache.doverrb, 1.147);
    UrbanV2SafetyCtx box{};
    box.kind = 1;
    box.box_x0 = -50.0F;
    box.box_x1 = 50.0F;
    box.box_y0 = -50.0F;
    box.box_y1 = 50.0F;
    box.box_z0 = 0.0F;
    box.box_z1 = 250.0F;
    check(std::fabs(urban_v2_safety_at(0.0F, 0.0F, 40.0F, box) - 40.0F) <
              1.0e-4F,
          "water safety: axial bound == 40");
    check(std::fabs(urban_v2_safety_at(0.0F, 0.0F, 249.0F, box) - 1.0F) <
              1.0e-4F,
          "water safety: near exit == 1");
    FILE* f = std::fopen("data/urban/c12_water75ev_urban_g4_11_3_2.csv", "r");
    check(f != nullptr, "water table: extraction CSV present");
    if (f == nullptr) return;
    std::fclose(f);
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
    } catch (...) {
        check(false, "water table: loads via carbon_core");
        return;
    }
    check(host.ranges_mm().size() == 4001, "water table: 4001 nodes",
          double(host.ranges_mm().size()), 4001.0);
    check(std::fabs(host.zeff() - 3.3334) < 1.0e-3, "water table: zeff",
          host.zeff(), 3.3334);
    check(std::fabs(host.radlen_mm() - 360.829) < 1.0e-2, "water table: radlen",
          host.radlen_mm(), 360.829);
    std::vector<float> e(host.energies_total_mev().begin(),
                         host.energies_total_mev().end());
    std::vector<float> r(host.ranges_mm().begin(), host.ranges_mm().end());
    std::vector<float> d(host.dedx_values().begin(), host.dedx_values().end());
    UrbanV2LossTable table{e.data(), r.data(), d.data(),
                           static_cast<int>(e.size())};
    UrbanV2Material mat{};
    mat.table = table;
    mat.zeff = static_cast<float>(host.zeff());
    mat.radlen_mm = static_cast<float>(host.radlen_mm());
    mat.projectile_z = 6;
    mat.projectile_a = 12;
    mat.mass_mev = 12.0F * 931.49410242F;
    mat.density_g_per_cm3 = 1.0F;
    mat.mfp_kind = 1;
    const Direction3F dir{0.0F, 0.0F, 1.0F};
    UrbanV2TrackState state{};
    const auto s = urban_v2_propose_and_sample(
        dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
        box, true, state, mat, 1.0F, 4242ULL, 17ULL, 3ULL, 70U);
    char lab[160];
    std::snprintf(lab, sizeof(lab),
                  "water propose: valid reason=%d g=%.5f t=%.5f range=%.2f",
                  s.limit_reason, double(s.final_geom_path_mm),
                  double(s.final_true_path_mm), double(s.current_range_mm));
    check(s.proposal_valid && s.final_geom_path_mm > 0.0F &&
              s.current_range_mm > 100.0F &&
              (s.limit_reason == 1 || s.limit_reason == 2),
          lab, double(s.limit_reason), 1.0);
}

void test_water_urban_segment_scaling() {
    // Unit-level pin of the full-chain s0025 finding: fixed-subdivision
    // Urban-v2 sampling is N-dependent. Compares per-unit-path sampled
    // angular variance of 1x0.05mm vs 2x0.025mm consecutive segments (second
    // segment reuses the persistent track state with first=false, exactly as
    // transport_sycl.cpp does; fresh state + first=true otherwise).
    // Metric is per-segment E[2(1-cos)] summed over the segments covering
    // 0.05mm, so no rotation composition is needed. Urban per-segment
    // variance scales ~(c1+c2*ln(h/X0))^2, hence the subdivided total is
    // SMALLER; the port has no consecutive-step renormalization. This is a
    // characterization pin (fast, deterministic), NOT an invariance
    // assertion: Geant4 Urban is itself step-limit dependent, so 0.05mm must
    // stay matched to the TOPAS water MaxStepSize.
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
    } catch (...) {
        check(false, "water scaling: loads table");
        return;
    }
    std::vector<float> e(host.energies_total_mev().begin(),
                         host.energies_total_mev().end());
    std::vector<float> r(host.ranges_mm().begin(), host.ranges_mm().end());
    std::vector<float> d(host.dedx_values().begin(), host.dedx_values().end());
    UrbanV2LossTable table{e.data(), r.data(), d.data(),
                           static_cast<int>(e.size())};
    UrbanV2Material mat{};
    mat.table = table;
    mat.zeff = static_cast<float>(host.zeff());
    mat.radlen_mm = static_cast<float>(host.radlen_mm());
    mat.projectile_z = 6;
    mat.projectile_a = 12;
    mat.mass_mev = 12.0F * 931.49410242F;
    mat.density_g_per_cm3 = 1.0F;
    mat.mfp_kind = 1;
    UrbanV2SafetyCtx box{};
    box.kind = 1;
    box.box_x0 = -50.0F;
    box.box_x1 = 50.0F;
    box.box_y0 = -50.0F;
    box.box_y1 = 50.0F;
    box.box_z0 = 0.0F;
    box.box_z1 = 250.0F;
    const Direction3F dir{0.0F, 0.0F, 1.0F};
    constexpr std::uint64_t kSeed = 777001ULL;
    constexpr int kSamples = 100000;
    double sum_single = 0.0, sum_pair = 0.0;
    long n_single = 0, n_pair = 0;
    for (int i = 0; i < kSamples; ++i) {
        const auto h = static_cast<std::uint64_t>(i);
        UrbanV2TrackState st{};
        const auto s = urban_v2_propose_and_sample(
            dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, st, mat, 1.0F, kSeed, h, 0ULL, 70U);
        if (s.proposal_valid) {
            sum_single += 2.0 * (1.0 - double(s.direction.z));
            ++n_single;
        }
        UrbanV2TrackState st2{};
        const auto a = urban_v2_propose_and_sample(
            dir, 3000.0F, 0.025F, 0.025F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, st2, mat, 1.0F, kSeed, h, 0ULL, 70U);
        const auto b = urban_v2_propose_and_sample(
            a.direction, 3000.0F, 0.025F, 0.025F, 0.0F, 0.0F, 40.0F,
            a.direction.x, a.direction.y, a.direction.z, box, false, st2, mat,
            1.0F, kSeed, h, 1ULL, 70U);
        if (a.proposal_valid && b.proposal_valid) {
            // Per-segment deflections about their own entry axes.
            const double t1 = 2.0 * (1.0 - double(a.direction.z));
            const double cosb =
                double(a.direction.x * b.direction.x +
                       a.direction.y * b.direction.y +
                       a.direction.z * b.direction.z);
            sum_pair += t1 + 2.0 * (1.0 - cosb);
            ++n_pair;
        }
    }
    const double m1 = sum_single / double(n_single);
    const double m2 = sum_pair / double(n_pair);
    const double ratio = m2 / m1;
    std::printf("  water scaling: E[th2] 1x0.05=%.6e 2x0.025=%.6e ratio=%.4f "
                "(n=%ld/%d)\n",
                m1, m2, ratio, n_pair, kSamples);
    check(n_single == kSamples && n_pair == kSamples,
          "water scaling: all proposals valid", double(n_pair),
          double(kSamples));
    // Measured 0.284 on 2026-09-22 (FP32, deterministic streams): far below
    // the Highland-like ~0.94 because Urban theta0 carries the
    // (coeffth1 + coeffth2*ln(t/X0)) correction, which is steep at
    // t/X0 ~ 1e-4. Band pins the behavior for regression; the exact value
    // still awaits a Geant4 two-step-limit reference (TOPAS MaxStepSize
    // 0.05 vs 0.025), so 0.05mm must stay matched to the TOPAS setting.
    check(ratio > 0.20 && ratio < 0.38, "water scaling: subdivided total < whole",
          ratio, 0.284);
}

void test_water_urban_low_energy_proposal() {
    // Diagnostic for the stalled 1.6M urban_v2 water replay: the entrance
    // file contains primaries down to 0.047 MeV total (table floor 0.12).
    // Reports proposal validity and final geom path at/near/below the floor.
    // A zero (or denormal-stalling) final path with valid=true would spin
    // the transport subdivision loop forever, since it advances traversed_mm
    // by final_geom_path_mm with no progress guard.
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
    } catch (...) {
        check(false, "water lowE: loads table");
        return;
    }
    std::vector<float> e(host.energies_total_mev().begin(),
                         host.energies_total_mev().end());
    std::vector<float> r(host.ranges_mm().begin(), host.ranges_mm().end());
    std::vector<float> d(host.dedx_values().begin(), host.dedx_values().end());
    UrbanV2LossTable table{e.data(), r.data(), d.data(),
                           static_cast<int>(e.size())};
    UrbanV2Material mat{};
    mat.table = table;
    mat.zeff = static_cast<float>(host.zeff());
    mat.radlen_mm = static_cast<float>(host.radlen_mm());
    mat.projectile_z = 6;
    mat.projectile_a = 12;
    mat.mass_mev = 12.0F * 931.49410242F;
    mat.density_g_per_cm3 = 1.0F;
    mat.mfp_kind = 1;
    UrbanV2SafetyCtx box{};
    box.kind = 1;
    box.box_x0 = -50.0F;
    box.box_x1 = 50.0F;
    box.box_y0 = -50.0F;
    box.box_y1 = 50.0F;
    box.box_z0 = 0.0F;
    box.box_z1 = 250.0F;
    const Direction3F dir{0.0F, 0.0F, 1.0F};
    for (const float epre : {0.05F, 0.1F, 1.2F, 12.0F}) {
        UrbanV2TrackState st{};
        const auto s = urban_v2_propose_and_sample(
            dir, epre, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, st, mat, 1.0F, 999ULL, 7ULL, 0ULL, 70U);
        std::printf("  water lowE: epre=%.3f valid=%d reason=%d g=%.6g t=%.6g\n",
                    double(epre), int(s.proposal_valid), s.limit_reason,
                    double(s.final_geom_path_mm), double(s.final_true_path_mm));
    }
    check(true, "water lowE: proposals reported");
}

void test_water_urban_subdivision_robustness() {
    // Host mirror of the transport water-urban subdivision loop: cover a
    // 0.1 mm transport step by repeated propose_and_sample calls with a
    // persistent track state (first=true only on segment 0), advancing by
    // final_geom_path_mm exactly as transport_sycl.cpp does. Reports the
    // segment count and flags zero/negative progress, which would spin the
    // device loop forever (no progress guard there). Sweep energies from the
    // replay-entrance floor (0.05 MeV total) to 3000 MeV.
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
    } catch (...) {
        check(false, "water subdiv: loads table");
        return;
    }
    std::vector<float> e(host.energies_total_mev().begin(),
                         host.energies_total_mev().end());
    std::vector<float> r(host.ranges_mm().begin(), host.ranges_mm().end());
    std::vector<float> d(host.dedx_values().begin(), host.dedx_values().end());
    UrbanV2LossTable table{e.data(), r.data(), d.data(),
                           static_cast<int>(e.size())};
    UrbanV2Material mat{};
    mat.table = table;
    mat.zeff = static_cast<float>(host.zeff());
    mat.radlen_mm = static_cast<float>(host.radlen_mm());
    mat.projectile_z = 6;
    mat.projectile_a = 12;
    mat.mass_mev = 12.0F * 931.49410242F;
    mat.density_g_per_cm3 = 1.0F;
    mat.mfp_kind = 1;
    UrbanV2SafetyCtx box{};
    box.kind = 1;
    box.box_x0 = -50.0F;
    box.box_x1 = 50.0F;
    box.box_y0 = -50.0F;
    box.box_y1 = 50.0F;
    box.box_z0 = 0.0F;
    box.box_z1 = 250.0F;
    const Direction3F dir{0.0F, 0.0F, 1.0F};
    bool all_ok = true;
    for (const float epre : {0.05F, 0.1F, 1.0F, 12.0F, 120.0F, 3000.0F}) {
        int worst_segs = 0, zero_runs = 0;
        for (int h = 0; h < 200; ++h) {
            UrbanV2TrackState st{};
            Direction3F sd = dir;
            float traversed = 0.0F;
            int segs = 0;
            bool stuck = false;
            while (traversed < 0.1F && segs < 100000) {
                const float segmm =
                    0.05F < 0.1F - traversed ? 0.05F : 0.1F - traversed;
                const auto s = urban_v2_propose_and_sample(
                    sd, epre, 0.05F, segmm, 0.0F, 0.0F, 40.0F, sd.x, sd.y,
                    sd.z, box, segs == 0, st, mat, 1.0F, 555ULL,
                    std::uint64_t(h), std::uint64_t(segs), 70U);
                if (!(s.final_geom_path_mm > 0.0F)) {
                    stuck = true;
                    break;
                }
                traversed += s.final_geom_path_mm;
                sd = s.direction;
                ++segs;
            }
            worst_segs = segs > worst_segs ? segs : worst_segs;
            if (stuck || segs >= 100000) {
                ++zero_runs;
                all_ok = false;
            }
        }
        std::printf("  water subdiv: epre=%.3f worst_segs=%d stuck=%d/200\n",
                    double(epre), worst_segs, zero_runs);
    }
    check(all_ok, "water subdiv: always progresses");
}

void test_delta_relocation() {
    // Shared-helper logic with SYNTHETIC channels (mechanics only, never
    // physics): one channel, fraction 0.6, unresolved 0.1, single sample at
    // local point (0.05,0,0.02), trivial path node at origin.
    carbon::WaterElectronChannel ch{};
    ch.low = 0.0;
    ch.high = 400.0;
    ch.fraction = 0.6;
    ch.unresolved = 0.1;
    ch.offset = 0;
    ch.count = 1;
    carbon::WaterElectronSample sm{};
    sm.cdf = 1.0;
    sm.pre = {0.0, 0.0, 0.0};
    sm.post = {0.05, 0.0, 0.02};
    std::uint32_t head = 0;
    carbon::WaterElectronPathNode nd{};
    nd.previous = carbon::kWaterPathNone;
    nd.point = {0.0, 0.0, 0.0};
    const auto pkt = carbon::sample_water_delta_packet(
        10.0, 250.0, 0.3, 0.7, &ch, &sm, &head, 1);
    check(pkt.valid && std::fabs(pkt.packet_mev - 6.0) < 1e-12 &&
              std::fabs(pkt.unresolved_mev - 1.0) < 1e-12,
          "relocate: phase-1 partition 6.0/1.0 of 10.0", pkt.packet_mev, 6.0);
    // Orthonormal basis for +z: ex=(1,0,0), ey=(0,1,0).
    const auto pl = carbon::place_water_delta_packet(
        pkt, 0.0, 0.0, 40.0, 0.0, 0.0, 1.0, 0.05, 1, 0, 0, 0, 1, 0, 0.5, &nd,
        1, nullptr, 250.0, 3.6);
    check(pl.valid && pl.path_status == 0, "relocate: contained placement");
    // draw.point = pre+0.7*(post-pre) = (0.035,0,0.014); birth=(0,0,40.025).
    check(std::fabs(pl.point_x - 0.035) < 1e-12 &&
              std::fabs(pl.point_z - 40.039) < 1e-12,
          "relocate: world point", pl.point_x, 0.035);
    check(pl.birth_roi == 0 && pl.deposit_roi == 0, "relocate: peak->peak");
    // Invalid draw propagates without placement.
    const auto bad = carbon::sample_water_delta_packet(
        10.0, 5000.0, 0.3, 0.7, &ch, &sm, &head, 1);
    check(!bad.valid, "relocate: out-of-domain energy rejected");
    // Zero deposit refuses immediately.
    const auto zero = carbon::sample_water_delta_packet(
        0.0, 250.0, 0.3, 0.7, &ch, &sm, &head, 1);
    check(!zero.valid, "relocate: zero deposit refused");
}

int run_urban_v2_helper_tests() {
    test_acceptance_gate();
    test_stable_delta();
    test_slit_navigation_and_safety();
    test_angular_small_quantity();
    test_loss_table_and_limiter();
    if (failures == 0) {
        std::printf("urban_v2_helpers: ALL PASS\n");
        return 0;
    }
    std::printf("urban_v2_helpers: %d FAILURES\n", failures);
    return 1;
}

}  // namespace
}  // namespace carbon

int main() { return carbon::run_urban_v2_helper_tests(); }

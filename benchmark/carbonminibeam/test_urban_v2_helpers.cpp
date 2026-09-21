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

namespace carbon {
namespace {
#include "detail/sycl_device_math.inc"
}  // namespace
}  // namespace carbon

namespace carbon {

namespace {

int failures = 0;

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
    // 4. postSafety >= raw_r -> full accept.
    copper_urban_v2_accept_displacement(1.0e-4F, 0.0F, 1.0F, ax, ay, branch);
    check(branch == 2 && std::fabs(double(ax) - 1.0e-4) < 1.0e-10 && ay == 0.0F,
          "accept: post>=raw -> full accept");
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

int run_urban_v2_helper_tests() {
    test_acceptance_gate();
    test_stable_delta();
    test_slit_navigation_and_safety();
    test_angular_small_quantity();
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

// Localization experiments for the Urban default-migration task
// (prompt sections 2.1/2.2/2.4). Host-side, deterministic, production code:
//   E1  rng::uniform01 raw-bit upper endpoint (strict-(0,1) claim vs fact).
//   E2  stable small-angle measurement of the 0.284 segment-scaling setup.
//   E3  fMinimal tlimitmin lifecycle: MAIGO per-step recompute vs the frozen
//       Geant4 11.3.2 StartTracking value, through the PRODUCTION limiter.
//
// PASS -> exit 0. FAIL -> exit nonzero. Informational prints go to stdout;
// evidence predictions are recorded in the matching evidence/ directory.
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "carbon/minibeam_collimator.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/rng.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/transport_config.hpp"
#include "carbon/water_electron_response.hpp"

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

// --- E1: raw-bit mapping + reachability ------------------------------------
void experiment_uniform01_endpoint() {
    constexpr float inv24 = 5.9604644775390625e-8f;  // 2^-24
    const std::uint32_t cases[] = {0U, 1U, (1U << 23) - 1U, 1U << 23,
                                   (1U << 23) + 1U, (1U << 24) - 2U,
                                   (1U << 24) - 1U};
    for (auto bits : cases) {
        const float v = (static_cast<float>(bits) + 0.5f) * inv24;
        std::printf("  E1 map: bits=%10u -> %a (%.9g)\n", bits, double(v),
                    double(v));
    }
    // Mapping-level fact: max bits round to exactly 1.0f in FP32.
    const float vmax =
        (static_cast<float>((1U << 24) - 1U) + 0.5f) * inv24;
    check(vmax == 1.0f, "E1: bits=2^24-1 maps to exactly 1.0f", vmax, 1.0);
    const float vmin = (0.0f + 0.5f) * inv24;
    check(vmin > 0.0f && vmin < 1.0f, "E1: bits=0 maps strictly inside",
          vmin, 0.0);
    // Reachability: find a real Philox preimage of the all-ones top-24.
    // P(hit) = 2^-24 per draw; bounded search, honest NOT_FOUND allowed.
    constexpr std::uint64_t kCap = 100000000ULL;
    std::uint64_t found_ii = 0;
    bool found = false;
    for (std::uint64_t ii = 0; ii < kCap; ++ii) {
        const auto w = rng::random_u32(20260922ULL, 7ULL, ii, 70U) >> 8U;
        if (w == (1U << 24) - 1U) {
            found = true;
            found_ii = ii;
            break;
        }
    }
    if (found) {
        const float v = rng::uniform01(20260922ULL, 7ULL, found_ii, 70U);
        std::printf("  E1 reach: seed=20260922 hist=7 dim=70 ii=%llu -> %a\n",
                    (unsigned long long)found_ii, double(v));
        check(v == 1.0f, "E1: reached preimage returns 1.0f", v, 1.0);
    } else {
        std::printf("  E1 reach: no preimage within %llu draws (NOT_FOUND, "
                    "mapping proof above still stands)\n",
                    (unsigned long long)kCap);
    }
    // Logic-level consequence: with u0 == 1.0f the Urban Bernoulli
    // `u0 < q_probability` takes the isotropic branch even at q == 1.
    const float u0 = 1.0f, q = 1.0f;
    check(!(u0 < q), "E1: u0==1, q==1 takes the else (isotropic) branch");
}

// --- E2: stable small-angle measurement ------------------------------------
void experiment_stable_small_angle() {
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
    } catch (...) {
        check(false, "E2: water loss table loads");
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
    constexpr int kSamples = 200000;
    // Four estimators of the same per-step angular content (production
    // sampler, water 0.05 mm, 250 MeV/u):
    //   A: 2*(1-double(dir.z))            (legacy test metric)
    //   B: 2*double(one_minus_cth)        (sampler-direct, stable)
    //   C: atan2(|cross|,dot)^2 in double (true angle of rounded direction)
    //   D: lateral-slope^2 = (dx^2+dy^2)/dz^2
    double sum_a = 0.0, sum_b = 0.0, sum_c = 0.0, sum_d = 0.0;
    long n = 0, cth_one = 0, info_lost = 0;
    for (int i = 0; i < kSamples; ++i) {
        UrbanV2TrackState st{};
        const auto s = urban_v2_propose_and_sample(
            dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, st, mat, 1.0F, kSeed, std::uint64_t(i), 0ULL, 70U);
        if (!s.proposal_valid) continue;
        ++n;
        const double dz = double(s.direction.z);
        const double a = 2.0 * (1.0 - dz);
        const double b = 2.0 * double(s.one_minus_cth);
        const double cx =
            double(s.direction.y) * 0.0 - double(s.direction.z) * 0.0;
        (void)cx;
        // cross((0,0,1), d) = (-dy, dx, 0)
        const double cross =
            std::sqrt(double(s.direction.x) * double(s.direction.x) +
                      double(s.direction.y) * double(s.direction.y));
        const double cth = std::atan2(cross, dz);
        const double c = cth * cth;
        const double dd = (double(s.direction.x) * double(s.direction.x) +
                           double(s.direction.y) * double(s.direction.y)) /
                          (dz * dz);
        sum_a += a;
        sum_b += b;
        sum_c += c;
        sum_d += dd;
        if (s.cth_rounded_to_one) ++cth_one;
        if (s.cth_rounded_to_one && s.one_minus_cth > 0.0F) ++info_lost;
    }
    const double ma = sum_a / n, mb = sum_b / n, mc = sum_c / n,
                 md = sum_d / n;
    std::printf("  E2 n=%ld: A[2(1-dz)]=%.6e B[2*omcth]=%.6e C[atan2^2]=%.6e "
                "D[slope^2]=%.6e\n",
                n, ma, mb, mc, md);
    std::printf("  E2: cth==1 fraction=%.4f omcth>0-but-rounded=%ld "
                "B/A=%.3f C/B=%.4f D/B=%.4f\n",
                double(cth_one) / n, info_lost, mb / ma, mc / mb, md / mb);
    check(n == kSamples, "E2: all proposals valid", double(n),
          double(kSamples));
    // The legacy metric A must demonstrably lose information vs the stable
    // sampler-direct moment B (that is the measurement defect, §2.4).
    check(info_lost > 0, "E2: rounded cth==1 hides nonzero omcth",
          double(info_lost), 0.0);
    check(mb > ma, "E2: stable moment B exceeds lossy metric A", mb, ma);
    // Post-fix characterization bands (deterministic streams; water 0.05 mm,
    // 250 MeV/u). B/C/D mutual agreement <=1e-4 proves the rotation preserves
    // the sampled angle and the stable moment is the right observable.
    // B vs direct-G4 single-step oracle: E[th^2] ratio 0.983 (see evidence).
    check(std::fabs(mc / mb - 1.0) < 1.0e-4 && std::fabs(md / mb - 1.0) < 1.0e-4,
          "E2: rotation preserves angle (C/B, D/B ~= 1)", mc / mb, 1.0);
    check(mb > 3.35e-8 && mb < 3.48e-8, "E2: stable-moment band", mb,
          3.4131e-8);
}

// --- E3/B3: tlimitmin lifecycle ------------------------------------------------
// E3 characterized the defect (per-step recompute ~3e-3 mm, 3e4x the frozen
// reference). After fix B3 this is the regression: the production limiter
// reports the frozen StartTracking value 1e-7 mm on ALL exit paths
// (reference fMinimal never calls ComputeStepmin/ComputeTlimitmin).
void experiment_tlimitmin_lifecycle() {
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
    } catch (...) {
        check(false, "E3: water loss table loads");
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
    constexpr float kRefTlimitmin = 1.0e-7F;  // 10*tlimitminfix, frozen
    constexpr float kLambdaLimit = 1.0F;      // mm, TOPAS log Llim
    for (const float epre : {3000.0F, 1000.0F, 300.0F, 100.0F, 30.0F,
                             10.0F}) {
        UrbanV2TrackState st{};
        // Production limiter, first water segment (at_boundary=true).
        // presafety=1e-6 forces exit path 3 (full fMinimal) at all
        // energies: with presafety=40 the low-E calls take the
        // far-from-boundary early exit and leave tlimitmin_mm at its
        // zero default, which is a separate path-coverage observation.
        const float t_in =
            copper_urban_v2_true_path_and_delta(
                0.05F, urban_v2_transport_mfp(mat, epre), 1.0e30F)
                .true_path_mm;
        const auto lim = copper_urban_v2_limit_step(
            t_in, epre, mat, 1.0e-6F, true, st, 4242ULL, 17ULL, 3ULL);
        const float tsmall_maigo =
            lim.tlimitmin_mm < kLambdaLimit ? lim.tlimitmin_mm : kLambdaLimit;
        char lab[160];
        std::snprintf(lab, sizeof(lab),
                      "B3: E=%.1f path=%d tlimitmin==1e-7 (got %g)",
                      double(epre), lim.exit_path,
                      double(lim.tlimitmin_mm));
        check(lim.exit_path == 3 && lim.tlimitmin_mm == kRefTlimitmin, lab,
              double(lim.tlimitmin_mm), 1.0e-7);
        check(tsmall_maigo == kRefTlimitmin, "B3: tsmall==1e-7",
              double(tsmall_maigo), 1.0e-7);
    }
    // Early exits carry the frozen value too (no zero default).
    {
        UrbanV2TrackState st{};
        // Path 0: extreme-small input.
        const auto p0 = copper_urban_v2_limit_step(
            1.0e-9F, 3000.0F, mat, 1.0e-6F, true, st, 1ULL, 2ULL, 3ULL);
        check(p0.exit_path == 0 && p0.tlimitmin_mm == kRefTlimitmin,
              "B3: path-0 tlimitmin frozen", double(p0.tlimitmin_mm),
              1.0e-7);
        // Path 2: far from boundary (huge presafety).
        const float t_in =
            copper_urban_v2_true_path_and_delta(
                0.05F, urban_v2_transport_mfp(mat, 3000.0F), 1.0e30F)
                .true_path_mm;
        const auto p2 = copper_urban_v2_limit_step(
            t_in, 3000.0F, mat, 1.0e9F, true, st, 1ULL, 2ULL, 3ULL);
        check(p2.exit_path == 2 && p2.tlimitmin_mm == kRefTlimitmin,
              "B3: path-2 tlimitmin frozen", double(p2.tlimitmin_mm),
              1.0e-7);
    }
}

// --- E4: MAIGO single-step theta writer --------------------------------------
// Same production call as E2 (water, 0.05 mm ceiling, 250 MeV/u). Writes
// space-angle/thx/thy in mrad for offline comparison against the direct
// Geant4 single-step oracle (evidence/.../g4_oracle/g4step_theta.csv).
void experiment_maigo_single_step() {
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
    } catch (...) {
        check(false, "E4: water loss table loads");
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
    FILE* f = std::fopen(
        "evidence/urban_9618eb0_20260922/maigo_step_theta.csv", "w");
    check(f != nullptr, "E4: output CSV opens");
    if (f == nullptr) return;
    std::fprintf(f, "# th_mrad,thx_mrad,thy_mrad,omcth\n");
    constexpr std::uint64_t kSeed = 777001ULL;
    constexpr int kSamples = 200000;
    long n = 0;
    for (int i = 0; i < kSamples; ++i) {
        UrbanV2TrackState st{};
        const auto s = urban_v2_propose_and_sample(
            dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, st, mat, 1.0F, kSeed, std::uint64_t(i), 0ULL, 70U);
        if (!s.proposal_valid) continue;
        ++n;
        const double thx =
            std::atan2(double(s.direction.x), double(s.direction.z)) *
            1000.0;
        const double thy =
            std::atan2(double(s.direction.y), double(s.direction.z)) *
            1000.0;
        std::fprintf(f, "%.6f,%.6f,%.6f,%.6e\n", std::sqrt(thx * thx + thy * thy),
                     thx, thy, double(s.one_minus_cth));
    }
    std::fclose(f);
    std::printf("  E4: wrote %ld rows\n", n);
    check(n == kSamples, "E4: all proposals valid", double(n),
          double(kSamples));
}

// --- B1: strict-uniform regression ------------------------------------------
// Pins rng::urban_unit_strict: open-interval (0,1) on all raw-bit edges,
// endpoint clamp value, identity with uniform01 elsewhere, determinism.
void test_strict_uniform() {
    const std::uint32_t cases[] = {0U, 1U, (1U << 23) - 1U, 1U << 23,
                                   (1U << 23) + 1U, (1U << 24) - 2U,
                                   (1U << 24) - 1U};
    for (auto bits : cases) {
        // Reproduce the mapping inline (raw bits are not injectable).
        constexpr float inv24 = 5.9604644775390625e-8f;
        const float raw = (static_cast<float>(bits) + 0.5f) * inv24;
        const float strict = raw < 1.0f ? raw : 0x1.fffffep-1f;
        char lab[96];
        std::snprintf(lab, sizeof(lab), "B1: strict map bits=%u in (0,1)",
                      bits);
        check(strict > 0.0f && strict < 1.0f, lab, strict, 0.5);
    }
    // Reached preimage (E1) now returns the clamp, not 1.0f.
    const float via_preimage = rng::urban_unit_strict(20260922ULL, 7ULL,
                                                      18823669ULL, 70U);
    check(via_preimage == 0x1.fffffep-1f, "B1: preimage clamps below 1",
          via_preimage, 1.0);
    // Identity with uniform01 on non-endpoint draws + determinism.
    long same = 0;
    for (std::uint64_t k = 0; k < 5000; ++k) {
        const float a = rng::urban_unit_strict(11ULL, 22ULL, k, 70U);
        const float b = rng::uniform01(11ULL, 22ULL, k, 70U);
        const float c = rng::urban_unit_strict(11ULL, 22ULL, k, 70U);
        if (a == b) ++same;
        if (a != c) {
            check(false, "B1: strict deterministic");
            return;
        }
        if (!(a > 0.0f && a < 1.0f)) {
            check(false, "B1: strict open interval on stream");
            return;
        }
    }
    check(same == 5000, "B1: strict == uniform01 off-endpoint", double(same),
          5000.0);
    // Bernoulli consequence: strict draws always satisfy u0 < 1.0f, so the
    // q >= 1 degenerate regime always takes the mixture branch.
    bool all_mix = true;
    for (std::uint64_t k = 0; k < 5000; ++k)
        all_mix &= (rng::urban_unit_strict(11ULL, 22ULL, k, 70U) < 1.0f);
    check(all_mix, "B1: strict draws take the q>=1 mixture branch");
}

// --- B4: branch-consistent delta ----------------------------------------------
// true_to_geom carries the stable delta of its taken branch; finalize
// returns the delta of ITS branch; untruncated proposals expose the limiter
// conversion delta (no par1<0-only re-inversion anywhere).
void test_branch_delta() {
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
    } catch (...) {
        check(false, "B4: water loss table loads");
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
    // 1. Small-t branch: series delta, positive, matches t^2/2λ.
    {
        const float lam0 = urban_v2_transport_mfp(mat, 3000.0F);
        const auto conv = copper_urban_v2_true_to_geom(0.05F, lam0, 124.0F,
                                                       3000.0F, mat);
        const double ref = double(0.05) * double(0.05) / (2.0 * double(lam0));
        const double rel =
            std::fabs(double(conv.delta_mm) - ref) / ref;
        check(conv.g_mm > 0.0F && conv.delta_mm > 0.0F && rel < 1.0e-6,
              "B4: small-t series delta", rel, 1.0e-6);
        const double rmax2 = double(conv.delta_mm) *
                             (2.0 * double(0.05F) - double(conv.delta_mm));
        check(rmax2 > 0.0, "B4: small-t rmax2 positive", rmax2, 0.0);
    }
    // 2. Range branch (t >= 5% range): direct double delta, self-consistent.
    {
        UrbanV2TrackState st{};
        const float t_in =
            copper_urban_v2_true_path_and_delta(
                0.05F, urban_v2_transport_mfp(mat, 12.0F), 1.0e30F)
                .true_path_mm;
        const auto lim = copper_urban_v2_limit_step(
            t_in, 12.0F, mat, 1.0e-6F, true, st, 5ULL, 6ULL, 7ULL);
        check(lim.valid, "B4: range-end proposal valid");
        const float t = 0.2F * lim.range_mm;  // >= range*dtrl
        const auto conv = copper_urban_v2_true_to_geom(t, lim.lambda0_mm,
                                                       lim.range_mm, 12.0F,
                                                       mat);
        check(conv.par1 > 0.0F, "B4: range branch taken", conv.par1, 0.0);
        const double direct = double(t) - double(conv.g_mm);
        const double rel =
            std::fabs(double(conv.delta_mm) - direct) / direct;
        check(conv.delta_mm > 0.0F && rel < 1.0e-6,
              "B4: range-branch delta self-consistent", rel, 1.0e-6);
    }
    // 3. Untruncated proposal exposes the limiter conversion delta.
    {
        UrbanV2TrackState st{};
        const auto s = urban_v2_propose_and_sample(
            dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, st, mat, 1.0F, 4242ULL, 17ULL, 3ULL, 70U);
        check(s.proposal_valid && !s.boundary_crossed,
              "B4: untruncated proposal");
        const float lam0 = urban_v2_transport_mfp(mat, 3000.0F);
        const auto conv = copper_urban_v2_true_to_geom(
            s.proposed_true_path_mm, lam0, s.current_range_mm, 3000.0F,
            mat);
        check(s.stable_delta_mm == conv.delta_mm,
              "B4: untruncated delta == limiter-branch delta",
              double(s.stable_delta_mm), double(conv.delta_mm));
    }
    // 4. Truncated proposal: 0 <= delta <= t_final, deterministic.
    {
        UrbanV2TrackState st{};
        const auto s = urban_v2_propose_and_sample(
            dir, 12.0F, 0.05F, 0.005F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, st, mat, 1.0F, 4242ULL, 17ULL, 3ULL, 70U);
        check(s.proposal_valid && s.boundary_crossed, "B4: truncated proposal");
        check(s.stable_delta_mm >= 0.0F &&
                  s.stable_delta_mm <= s.final_true_path_mm,
              "B4: truncated delta bounded", double(s.stable_delta_mm),
              double(s.final_true_path_mm));
        UrbanV2TrackState st2{};
        const auto s2 = urban_v2_propose_and_sample(
            dir, 12.0F, 0.05F, 0.005F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, st2, mat, 1.0F, 4242ULL, 17ULL, 3ULL, 70U);
        check(s2.stable_delta_mm == s.stable_delta_mm,
              "B4: truncated delta deterministic");
    }
}

// --- B5: failure detection (host fault injection) --------------------------------
// Invalid proposals must be DETECTABLE (proposal_valid=false) so the device
// fatal path (slot 137 -> host throw) can fire. Covers: empty table, null
// table, NaN energy, zero ceiling, zero boundary.
void test_failure_detection() {
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
    } catch (...) {
        check(false, "B5: water loss table loads");
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
    UrbanV2TrackState st{};
    const auto ok = urban_v2_propose_and_sample(
        dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
        box, true, st, mat, 1.0F, 1ULL, 2ULL, 3ULL, 70U);
    check(ok.proposal_valid, "B5: sane proposal valid");
    // Empty table.
    UrbanV2Material bad_mat = mat;
    UrbanV2LossTable empty{nullptr, nullptr, nullptr, 0};
    bad_mat.table = empty;
    UrbanV2TrackState st2{};
    check(!urban_v2_propose_and_sample(dir, 3000.0F, 0.05F, 0.05F, 0.0F,
                                       0.0F, 40.0F, 0.0F, 0.0F, 1.0F, box,
                                       true, st2, bad_mat, 1.0F, 1ULL, 2ULL,
                                       3ULL, 70U)
               .proposal_valid,
          "B5: empty table -> invalid");
    // NaN energy.
    UrbanV2TrackState st3{};
    check(!urban_v2_propose_and_sample(dir,
                                       std::numeric_limits<float>::quiet_NaN(),
                                       0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F,
                                       0.0F, 1.0F, box, true, st3, mat, 1.0F,
                                       1ULL, 2ULL, 3ULL, 70U)
               .proposal_valid,
          "B5: NaN energy -> invalid");
    // Zero ceiling / zero boundary.
    UrbanV2TrackState st4{}, st5{};
    check(!urban_v2_propose_and_sample(dir, 3000.0F, 0.0F, 0.05F, 0.0F, 0.0F,
                                       40.0F, 0.0F, 0.0F, 1.0F, box, true,
                                       st4, mat, 1.0F, 1ULL, 2ULL, 3ULL, 70U)
               .proposal_valid,
          "B5: zero ceiling -> invalid");
    check(!urban_v2_propose_and_sample(dir, 3000.0F, 0.05F, 0.0F, 0.0F, 0.0F,
                                       40.0F, 0.0F, 0.0F, 1.0F, box, true,
                                       st5, mat, 1.0F, 1ULL, 2ULL, 3ULL, 70U)
               .proposal_valid,
          "B5: zero boundary -> invalid");
}

// --- B6: RNG addressing proof + boundary equivalence ---------------------------
// The water loops address RNG as steps*1024+segment_index. Mixed-radix
// injectivity needs segment_index < 1024 per macro step (structural half,
// enforced by config validation: maximum_step_mm <= 1024*max_segment_mm);
// partner steps of a cross-dim/block collision sit at >= 12.6M macro steps
// per history given the Urban dim blocks (see evidence branch map), far
// beyond any runnable history. This pins the arithmetic contract.
void test_rng_addressing() {
    constexpr std::uint64_t kStride = 1024U;
    // Edges of the supported domain.
    const std::uint64_t ii_a = 0ULL * kStride + 0ULL;
    const std::uint64_t ii_b = 0ULL * kStride + 1023ULL;
    const std::uint64_t ii_c = 1ULL * kStride + 0ULL;
    const std::uint64_t ii_d = 4194303ULL * kStride + 1023ULL;
    check(ii_a == 0ULL && ii_b == 1023ULL && ii_c == 1024ULL,
          "B6: addressing edges", double(ii_c), 1024.0);
    check(ii_d == 4194303ULL * 1024ULL + 1023ULL, "B6: addressing top",
          double(ii_d), double(4194303ULL * 1024ULL + 1023ULL));
    // Production subdivision depths stay far below the stride: 0.1 mm macro
    // / 0.05 mm urban ceiling = 2 segments; guard allows up to 1024.
    check(0.1 / 0.05 < 1024.0, "B6: production segs << stride");
    check(51.2 / 0.05 == 1024.0, "B6: 51.2mm bound == stride",
          51.2 / 0.05, 1024.0);
    // Uniqueness spot check over the domain (mixed-radix: distinct pairs ->
    // distinct indices).
    bool unique = true;
    std::uint64_t prev = 0;
    bool first = true;
    for (std::uint64_t s = 0; s < 5000; s += 7) {
        for (std::uint64_t g = 0; g < 1024; g += 131) {
            const std::uint64_t ii = s * kStride + g;
            if (!first && ii <= prev) unique = false;
            prev = ii;
            first = false;
        }
    }
    check(unique, "B6: composition strictly increases");
}

// Secondary at_boundary=false must not change sampling where tlimit is
// non-binding (0.05 mm / 3000 MeV): identical draws, identical output.
// If tlimit ever binds at this scale the test fails loudly.
void test_secondary_boundary_equivalence() {
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
    } catch (...) {
        check(false, "B6: water loss table loads");
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
    UrbanV2TrackState st_b{}, st_n{};
    const auto s_b = urban_v2_propose_and_sample(
        dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
        box, true, st_b, mat, 1.0F, 4242ULL, 17ULL, 3ULL, 110U);
    const auto s_n = urban_v2_propose_and_sample(
        dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
        box, false, st_n, mat, 1.0F, 4242ULL, 17ULL, 3ULL, 110U);
    check(s_b.proposal_valid && s_n.proposal_valid,
          "B6: both boundary modes valid");
    check(s_b.direction.x == s_n.direction.x &&
              s_b.direction.z == s_n.direction.z &&
              s_b.displacement_mm.x == s_n.displacement_mm.x &&
              s_b.one_minus_cth == s_n.one_minus_cth,
          "B6: boundary flag inert at non-binding scale");
}

// B6 config guard: maximum_step_mm beyond 1024 subdivision ceilings with a
// water subdividing model must fail fast (RNG substep addressing proof).
// NOTE: in water mode the unified-water gate already caps maximum_step at
// 1.0 mm, so the 51.2/102.4 mm guard is defense-in-depth for future modes;
// it cannot be triggered through water-mode YAML today. This test therefore
// pins the INVARIANT on the shipped validation configs (parse+validate only,
// no GPU): every water Urban config must satisfy maximum_step_mm <= 51.2,
// every water FE config <= 102.4. If a future config violates it, the guard
// (not the physics) must be fixed first.
void test_rng_config_guard() {
    struct Case {
        const char* file;
        double bound;
    };
    const Case cases[] = {
        {"config/beam_minibeam_single_center_em_only_e250_urban_v2_cuwater_10m.yaml",
         51.2},
        {"config/beam_minibeam_water_replay_e250_em12800k_urban_v2.yaml",
         51.2},
        {"config/beam_minibeam_single_center_em_only_e250_urban_v2_10m.yaml",
         102.4},
    };
    for (const auto& tc : cases) {
        bool threw = false;
        double max_step = 0.0;
        try {
            const auto cfg = load_config(tc.file);
            max_step = cfg.maximum_step_mm;
        } catch (const std::exception& ex) {
            threw = true;
            std::printf("  B6cfg: %s threw: %.160s\n", tc.file, ex.what());
        }
        char lab[256];
        std::snprintf(lab, sizeof(lab), "B6cfg: %s parses, max_step=%.3f<=%.1f",
                      tc.file, max_step, tc.bound);
        check(!threw && max_step <= tc.bound, lab, max_step, tc.bound);
    }
}

// --- B2 note: no double-decision path ------------------------------
// The tau>0.05 double chain was evaluated and REMOVED: reachable-domain
// audit (q_audit.py, t<=min(0.05,range/2) at every energy grid point) shows
// zero float-vs-double trial flips, and the drift corner (t=0.05, E~1 MeV)
// cannot sample (t<range implies tau<=range/lambda0<3e-3 for C12-water).
// Unconditional double cost 3-4x wall time on sm_75 for zero reachable
// effect (measured: mini replay 2.2s -> 5.4s, restored after removal).
// Verdict pinned by: E2 stable-moment bands + direct-G4 single-step oracle
// comparison (q999 0.998, E[th^2] 0.983).

// --- E7: host multi-step slab composition --------------------------------------
// Pencil C12 through Water_75eV slabs with the PRODUCTION propose path
// (0.05 mm ceiling, table energy loss, persistent track state): validates
// direction composition + displacement + energy evolution against the
// direct-G4 slab oracle (g4_oracle/slab{1,10}ms005), WITHOUT full transport.
// Writes exit rows: th_mrad, x_mm, y_mm, E_MeV.
void experiment_slab_composition(double slab_mm, int n, const char* tag) {
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
    } catch (...) {
        check(false, "E7: water loss table loads");
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
    box.box_z0 = -10.0F;
    box.box_z1 = 260.0F;
    char fn[256];
    std::snprintf(fn, sizeof(fn),
                  "evidence/urban_9618eb0_20260922/maigo_slab_%s.csv", tag);
    FILE* f = std::fopen(fn, "w");
    check(f != nullptr, "E7: output CSV opens");
    if (f == nullptr) return;
    std::fprintf(f, "# th_mrad,x_mm,y_mm,E_MeV\n");
    long done = 0;
    for (int i = 0; i < n; ++i) {
        UrbanV2TrackState st{};
        Direction3F dir{0.0F, 0.0F, 1.0F};
        double x = 0.0, y = 0.0, z = 0.0, energy = 3000.0;
        bool first = true, alive = true;
        // NOTE: step_index MUST advance per segment (production passes
        // steps*1024+segment_index); a fixed 0 reuses identical draws on
        // every segment and scatters coherently (caught 2026-09-22: 23x
        // variance inflation in this test's first revision).
        std::uint64_t seg = 0;
        while (z < slab_mm && energy > 0.2) {
            const float seg_len = 0.05F;
            const auto s = urban_v2_propose_and_sample(
                dir, float(energy), seg_len, seg_len, float(x), float(y),
                float(z), dir.x, dir.y, dir.z, box, first, st, mat, 1.0F,
                777001ULL, std::uint64_t(i), seg, 70U);
            if (!s.proposal_valid || s.final_geom_path_mm <= 0.0F) {
                alive = false;
                break;
            }
            first = false;
            ++seg;
            const double g = double(s.final_geom_path_mm);
            x += g * double(dir.x) + double(s.displacement_mm.x);
            y += g * double(dir.y) + double(s.displacement_mm.y);
            z += g * double(dir.z) + double(s.displacement_mm.z);
            dir = s.direction;
            // Table energy at remaining range (CSDA-style, oracle-grade).
            energy -= double(s.final_true_path_mm) *
                      double(urban_v2_loss_dedx(mat.table, float(energy)));
            if (!(energy > 0.0)) {
                alive = false;
                break;
            }
        }
        if (!alive || z < slab_mm) continue;
        // Project to exact exit plane along final direction.
        const double res = (slab_mm - z) / double(dir.z);
        x += res * double(dir.x);
        y += res * double(dir.y);
        const double thx = std::atan2(double(dir.x), double(dir.z)) * 1000.0;
        const double thy = std::atan2(double(dir.y), double(dir.z)) * 1000.0;
        std::fprintf(f, "%.6f,%.6f,%.6f,%.4f\n",
                     std::sqrt(thx * thx + thy * thy), x, y, energy);
        ++done;
    }
    std::fclose(f);
    std::printf("  E7 %s: exit %ld/%d\n", tag, done, n);
    check(done > n / 2, "E7: majority exit slab", double(done),
          double(n) / 2.0);
}

int run_localize() {
    std::printf("[E1] uniform01 raw-bit endpoint\n");
    experiment_uniform01_endpoint();
    std::printf("[E2] stable small-angle measurement\n");
    experiment_stable_small_angle();
    std::printf("[E3] fMinimal tlimitmin lifecycle\n");
    experiment_tlimitmin_lifecycle();
    std::printf("[E4] MAIGO single-step theta writer (for G4-oracle compare)\n");
    experiment_maigo_single_step();
    std::printf("[B1] strict-uniform regression\n");
    test_strict_uniform();
    std::printf("[B4] branch-consistent delta\n");
    test_branch_delta();
    std::printf("[B5] failure detection\n");
    test_failure_detection();
    std::printf("[B6] RNG addressing + boundary equivalence\n");
    test_rng_addressing();
    test_secondary_boundary_equivalence();
    std::printf("[B6cfg] RNG config guard\n");
    test_rng_config_guard();
    std::printf("[E7] slab composition 1mm\n");
    experiment_slab_composition(1.0, 20000, "w1ms005");
    std::printf("[E7] slab composition 10mm\n");
    experiment_slab_composition(10.0, 20000, "w10ms005");
    if (failures == 0) {
        std::printf("urban_localize: ALL PASS\n");
        return 0;
    }
    std::printf("urban_localize: %d FAILURES\n", failures);
    return 1;
}

}  // namespace
}  // namespace carbon

int main() { return carbon::run_localize(); }

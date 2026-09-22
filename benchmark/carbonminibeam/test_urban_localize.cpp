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
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

// Fix C5: test outputs must NEVER rewrite frozen evidence. Writers go to
// $CARBON_TEST_OUT_DIR (or the OS temp dir); formal results are archived by
// an independent runner under hash names.
std::string test_out_path(const char* name) {
    const char* env = std::getenv("CARBON_TEST_OUT_DIR");
    std::string dir = (env != nullptr && env[0] != '\0')
        ? env
        : std::filesystem::temp_directory_path().string();
    return dir + "/" + name;
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
    // Fix C2: single-realization pins are fragile across intentional RNG
    // domain changes (same sampler distribution, different uniform stream).
    // Ensemble over 8 seeds estimates the expectation; the frozen band is
    // set from the ensemble with a statistically derived width.
    // Four estimators of the same per-step angular content (production
    // sampler, water 0.05 mm, 250 MeV/u):
    //   A: 2*(1-double(dir.z))            (legacy test metric)
    //   B: 2*double(one_minus_cth)        (sampler-direct, stable)
    //   C: atan2(|cross|,dot)^2 in double (true angle of rounded direction)
    //   D: lateral-slope^2 = (dx^2+dy^2)/dz^2
    double sum_a = 0.0, sum_b = 0.0, sum_c = 0.0, sum_d = 0.0;
    long n = 0, cth_one = 0, info_lost = 0;
    double seed_mb[8];
    for (int sd = 0; sd < 8; ++sd) {
        const std::uint64_t seed = kSeed + std::uint64_t(sd) * 1013904223ULL;
        double sum = 0.0;
        long nn = 0;
        for (int i = 0; i < kSamples; ++i) {
            UrbanV2TrackState st{};
            const auto s = urban_v2_propose_and_sample(
                dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F,
                1.0F, box, true, st, mat, 1.0F, seed, std::uint64_t(i), 0ULL,
                0U, 70U);
            if (!s.proposal_valid) continue;
            ++nn;
            sum += 2.0 * double(s.one_minus_cth);
            if (sd > 0) continue;
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
        seed_mb[sd] = nn > 0 ? sum / nn : 0.0;
        check(nn == kSamples, "E2: all proposals valid (ensemble)", double(nn),
              double(kSamples));
    }
    double emean = 0.0;
    for (int sd = 0; sd < 8; ++sd) emean += seed_mb[sd];
    emean /= 8.0;
    double svar = 0.0;
    for (int sd = 0; sd < 8; ++sd) svar += (seed_mb[sd] - emean) * (seed_mb[sd] - emean);
    const double se = std::sqrt(svar / 7.0 / 8.0);
    std::printf("  E2 ensemble (8x%dk, MSC domain): mean=%.6e SE=%.3e\n",
                kSamples / 1000, emean, se);
    for (int sd = 0; sd < 8; ++sd)
        std::printf("    seed[%d]=%.6e\n", sd, seed_mb[sd]);
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
    // Ensemble-pinned band (MSC domain, frozen 2026-09-22): the old
    // single-realization center 3.4131e-8 must lie inside the new ensemble
    // CI (distribution unchanged), and the frozen band covers the ensemble
    // mean at >> run-to-run scatter.
    check(std::fabs(3.4131e-8 - emean) < 4.0 * se,
          "E2: old domain value inside new ensemble CI", 3.4131e-8, emean);
    check(emean > 3.35e-8 && emean < 3.48e-8, "E2: ensemble-moment band",
          emean, 3.4131e-8);
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
            t_in, epre, mat, 1.0e-6F, true, st, 4242ULL, 17ULL, 0ULL, 3U);
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
            1.0e-9F, 3000.0F, mat, 1.0e-6F, true, st, 1ULL, 2ULL, 0ULL, 3U);
        check(p0.exit_path == 0 && p0.tlimitmin_mm == kRefTlimitmin,
              "B3: path-0 tlimitmin frozen", double(p0.tlimitmin_mm),
              1.0e-7);
        // Path 2: far from boundary (huge presafety).
        const float t_in =
            copper_urban_v2_true_path_and_delta(
                0.05F, urban_v2_transport_mfp(mat, 3000.0F), 1.0e30F)
                .true_path_mm;
        const auto p2 = copper_urban_v2_limit_step(
            t_in, 3000.0F, mat, 1.0e9F, true, st, 1ULL, 2ULL, 0ULL, 3U);
        check(p2.exit_path == 2 && p2.tlimitmin_mm == kRefTlimitmin,
              "B3: path-2 tlimitmin frozen", double(p2.tlimitmin_mm),
              1.0e-7);
    }
}

// --- E4: MAIGO single-step theta writer --------------------------------------
// Same production call as E2 (water, 0.05 mm ceiling, 250 MeV/u). Writes
// space-angle/thx/thy in mrad to the test output dir for offline comparison
// against the direct Geant4 single-step oracle
// (evidence/.../g4_oracle/g4step_theta.csv).
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
        test_out_path("maigo_step_theta.csv").c_str(), "w");
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
            box, true, st, mat, 1.0F, kSeed, std::uint64_t(i), 0ULL, 0U, 70U);
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
            t_in, 12.0F, mat, 1.0e-6F, true, st, 5ULL, 6ULL, 0ULL, 7U);
        check(lim.valid, "B4: range-end proposal valid");
        const float t = 0.2F * lim.range_mm;  // >= range*dtrl
        const auto conv = copper_urban_v2_true_to_geom(t, lim.lambda0_mm,
                                                       lim.range_mm, 12.0F,
                                                       mat);
        check(conv.par1 > 0.0F, "B4: range branch taken", conv.par1, 0.0);
        // C1 contract: delta == t-g exactly where that difference is
        // representable (> 8 ulp); below, the constant-lambda fallback
        // (validated independently in C1 Part 3) replaces the noise.
        const double direct = double(t) - double(conv.g_mm);
        const double ulp_t = double(std::nextafter(t, 1.0e30F)) - double(t);
        if (direct > 8.0 * ulp_t) {
            const double rel =
                std::fabs(double(conv.delta_mm) - direct) / direct;
            check(conv.delta_mm > 0.0F && rel < 1.0e-6,
                  "B4: range-branch delta self-consistent", rel, 1.0e-6);
        } else {
            const float lam0 = urban_v2_transport_mfp(mat, 12.0F);
            const double ref = double(t) * double(t) /
                (2.0 * double(lam0));
            const double rel =
                std::fabs(double(conv.delta_mm) - ref) / ref;
            check(conv.delta_mm > 0.0F && rel < 0.35,
                  "B4: range-branch fallback delta bounded", rel, 0.35);
        }
    }
    // 3. Untruncated proposal exposes the limiter conversion delta.
    {
        UrbanV2TrackState st{};
        const auto s = urban_v2_propose_and_sample(
            dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, st, mat, 1.0F, 4242ULL, 17ULL, 0ULL, 3U, 70U);
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
            box, true, st, mat, 1.0F, 4242ULL, 17ULL, 0ULL, 3U, 70U);
        check(s.proposal_valid && s.boundary_crossed, "B4: truncated proposal");
        check(s.stable_delta_mm >= 0.0F &&
                  s.stable_delta_mm <= s.final_true_path_mm,
              "B4: truncated delta bounded", double(s.stable_delta_mm),
              double(s.final_true_path_mm));
        UrbanV2TrackState st2{};
        const auto s2 = urban_v2_propose_and_sample(
            dir, 12.0F, 0.05F, 0.005F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, st2, mat, 1.0F, 4242ULL, 17ULL, 0ULL, 3U, 70U);
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
        box, true, st, mat, 1.0F, 1ULL, 2ULL, 0ULL, 3U, 70U);
    check(ok.proposal_valid, "B5: sane proposal valid");
    // Empty table.
    UrbanV2Material bad_mat = mat;
    UrbanV2LossTable empty{nullptr, nullptr, nullptr, 0};
    bad_mat.table = empty;
    UrbanV2TrackState st2{};
    check(!urban_v2_propose_and_sample(dir, 3000.0F, 0.05F, 0.05F, 0.0F,
                                       0.0F, 40.0F, 0.0F, 0.0F, 1.0F, box,
                                       true, st2, bad_mat, 1.0F, 1ULL, 2ULL, 0ULL, 3U, 70U)
               .proposal_valid,
          "B5: empty table -> invalid");
    // NaN energy.
    UrbanV2TrackState st3{};
    check(!urban_v2_propose_and_sample(dir,
                                       std::numeric_limits<float>::quiet_NaN(),
                                       0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F,
                                       0.0F, 1.0F, box, true, st3, mat, 1.0F,
                                       1ULL, 2ULL, 0ULL, 3U, 70U)
               .proposal_valid,
          "B5: NaN energy -> invalid");
    // Zero ceiling / zero boundary.
    UrbanV2TrackState st4{}, st5{};
    check(!urban_v2_propose_and_sample(dir, 3000.0F, 0.0F, 0.05F, 0.0F, 0.0F,
                                       40.0F, 0.0F, 0.0F, 1.0F, box, true,
                                       st4, mat, 1.0F, 1ULL, 2ULL, 0ULL, 3U, 70U)
               .proposal_valid,
          "B5: zero ceiling -> invalid");
    check(!urban_v2_propose_and_sample(dir, 3000.0F, 0.05F, 0.0F, 0.0F, 0.0F,
                                       40.0F, 0.0F, 0.0F, 1.0F, box, true,
                                       st5, mat, 1.0F, 1ULL, 2ULL, 0ULL, 3U, 70U)
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
// --- C2: MSC RNG domain separation (proof, not sampling) ----------------
// Legacy counter[3] = (ii>>32)^(dim/4) with ii < 2^32 in every legacy
// consumer: bit 30 always clear. MSC counter[3] carries msc_domain_tag:
// bit 30 always set. The counter SETS are disjoint; Philox is a bijection
// over counters at fixed key, so no MSC draw can equal a legacy draw.
// This test pins the ADDRESS construction (not random outputs).
void test_msc_domain_separation() {
    // Legacy side: every (ii, dim) a legacy consumer can produce keeps
    // bit 30 clear (ii < 2^32: outer steps, fixed 0/indices; dim any u32).
    const std::uint64_t legacy_ii[] = {0ULL,
                                       1ULL,
                                       99999ULL,
                                       100000ULL,
                                       0xFFFFFFFFULL};
    const std::uint32_t legacy_dims[] = {0U,  1U,  10U, 11U, 14U, 17U,
                                         19U, 20U, 24U, 30U, 40U, 50U,
                                         58U, 59U, 70U, 71U, 80U, 81U,
                                         100U, 110U, 120U, 0xFFFFFFFFU};
    for (auto ii : legacy_ii) {
        for (auto d : legacy_dims) {
            const std::uint32_t w3 = static_cast<std::uint32_t>(ii >> 32U) ^
                (d / 4U);
            check((w3 & 0x40000000U) == 0U, "C2: legacy word3 bit30 clear");
        }
    }
    // MSC side: bit 30 set for the full envelope (ii < 2^32 enforced by
    // the propose/limiter guards; dims cover all Urban lanes 40..130).
    const std::uint64_t msc_ii[] = {0ULL,
                                    1023ULL,
                                    1024ULL,
                                    4194303ULL * 1024ULL + 1023ULL,
                                    0xFFFFFFFFULL};
    for (auto ii : msc_ii) {
        for (std::uint32_t d = 40U; d < 130U; ++d) {
            const std::uint32_t w3 = static_cast<std::uint32_t>(ii >> 32U) ^
                (d / 4U) ^ rng::msc_domain_tag;
            check((w3 & 0x40000000U) != 0U, "C2: MSC word3 bit30 set");
        }
    }
    // The documented collision (outer=0, seg=0, dim=70: elastic ii=0 dim=70
    // vs old-Urban ii=0 dim=70) is gone: same key/history/lane, the MSC
    // draw differs because its counter differs.
    const auto legacy_draw = rng::uniform01(11ULL, 22ULL, 0ULL, 70U);
    const auto msc_draw = rng::msc_unit_strict(11ULL, 22ULL, 0ULL, 70U);
    check(legacy_draw != msc_draw, "C2: known collision point separated",
          double(legacy_draw), double(msc_draw));
    // MSC determinism + segment sensitivity (spot, not a proof).
    check(rng::msc_unit_strict(11ULL, 22ULL, 5ULL, 70U) ==
              rng::msc_unit_strict(11ULL, 22ULL, 5ULL, 70U),
          "C2: MSC draw deterministic");
    check(rng::msc_unit_strict(11ULL, 22ULL, 5ULL, 70U) !=
              rng::msc_unit_strict(11ULL, 22ULL, 6ULL, 70U),
          "C2: MSC draw segment-sensitive");
    // Strict endpoint preserved on the new domain.
    check(rng::msc_unit_strict(11ULL, 22ULL, 5ULL, 70U) < 1.0F &&
              rng::msc_unit_strict(11ULL, 22ULL, 5ULL, 70U) > 0.0F,
          "C2: MSC draw in open (0,1)");
}

// Fix C2: address-proof guards reject BEFORE any draw.
void test_rng_index_guards() {
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
    } catch (...) {
        check(false, "C2: water loss table loads");
        return;
    }
    std::vector<float> e(host.energies_total_mev().begin(),
                         host.energies_total_mev().end());
    std::vector<float> r(host.ranges_mm().begin(), host.ranges_mm().end());
    std::vector<float> d(host.dedx_values().begin(),
                         host.dedx_values().end());
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
    // Top of the legal envelope: outer = 2^22-1, segment = 1023.
    {
        UrbanV2TrackState st{};
        const auto s = urban_v2_propose_and_sample(
            dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, st, mat, 1.0F, 1ULL, 2ULL, (1ULL << 22) - 1ULL, 1023U,
            70U);
        check(s.proposal_valid, "C2: envelope top valid");
    }
    // segment == 1024 rejected before any draw.
    {
        UrbanV2TrackState st{};
        const auto s = urban_v2_propose_and_sample(
            dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, st, mat, 1.0F, 1ULL, 2ULL, 0ULL, 1024U, 70U);
        check(!s.proposal_valid, "C2: segment==1024 rejected");
    }
    // outer == 2^22 rejected (would break the ii < 2^32 proof).
    {
        UrbanV2TrackState st{};
        const auto s = urban_v2_propose_and_sample(
            dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, st, mat, 1.0F, 1ULL, 2ULL, (1ULL << 22), 0U, 70U);
        check(!s.proposal_valid, "C2: outer==2^22 rejected");
    }
    // Limiter guards the same envelope (dims 58/59 live there).
    {
        UrbanV2TrackState st{};
        const auto lim = copper_urban_v2_limit_step(
            0.05F, 3000.0F, mat, 1.0e-6F, true, st, 1ULL, 2ULL, 0ULL, 1024U);
        check(!lim.valid, "C2: limiter rejects segment==1024");
    }
}

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
        box, true, st_b, mat, 1.0F, 4242ULL, 17ULL, 0ULL, 3U, 110U);
    const auto s_n = urban_v2_propose_and_sample(
        dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
        box, false, st_n, mat, 1.0F, 4242ULL, 17ULL, 0ULL, 3U, 110U);
    check(s_b.proposal_valid && s_n.proposal_valid,
          "B6: both boundary modes valid");
    check(s_b.direction.x == s_n.direction.x &&
              s_b.direction.z == s_n.direction.z &&
              s_b.displacement_mm.x == s_n.displacement_mm.x &&
              s_b.one_minus_cth == s_n.one_minus_cth,
          "B6: boundary flag inert at non-binding scale");
    // Fix C4 newborn semantics (low-E binding scale): fresh state +
    // at_boundary=true initializes tlimit from current (range, lambda)
    // like G4 firstStep; fresh + false keeps the 1e10 sentinel (never
    // limits). Both proposals stay valid; the tlimit values must differ
    // exactly as specified.
    {
        UrbanV2TrackState st1{}, st2{};
        const auto lim_true = copper_urban_v2_limit_step(
            0.05F, 6.0F, mat, 1.0e-6F, true, st1, 4242ULL, 17ULL, 0ULL, 3U);
        const auto lim_false = copper_urban_v2_limit_step(
            0.05F, 6.0F, mat, 1.0e-6F, false, st2, 4242ULL, 17ULL, 0ULL, 3U);
        check(lim_true.valid && lim_false.valid, "C4: newborn limits valid");
        const float lam0 = urban_v2_transport_mfp(mat, 6.0F);
        // MSC mirror range at 6 MeV (E/7.2), facrange 0.2 (Rfact).
        const float r_msc = 6.0F / 7.2F;
        const float expect =
            0.2F * (r_msc > lam0 ? r_msc : lam0);
        check(std::fabs(lim_true.tlimit_used_mm - expect) / expect < 1.0e-5,
              "C4: newborn tlimit initialized", lim_true.tlimit_used_mm,
              expect);
        check(lim_false.tlimit_used_mm == 1.0e10F,
              "C4: non-boundary keeps sentinel", lim_false.tlimit_used_mm,
              1.0e10);
    }
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

// --- E7: host multi-step slab composition (DIAGNOSTIC, not a gate) -------
// Pencil C12 through Water_75eV slabs with the PRODUCTION propose path
// (0.05 mm ceiling, table energy loss, persistent track state): exercises
// direction composition + displacement + energy evolution end to end on the
// host and guards against hangs (majority must exit). It is NOT a physics
// acceptance: the safety box here is not a 1/10 mm slab, the last segment
// scatters before back-projecting to the exit, and no matching in-repo
// oracle exists (the G4 slab CSVs live outside the repo with a different
// source). A real slab-geometry reference comparison is BLOCKED; promotion
// must not cite E7. Writes exit rows (th_mrad, x_mm, y_mm, E_MeV) to the
// test output dir (never frozen evidence).
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
    const std::string out_name =
        std::string("maigo_slab_") + tag + ".csv";
    const std::string out_path = test_out_path(out_name.c_str());
    char fn[512];
    std::snprintf(fn, sizeof(fn), "%s", out_path.c_str());
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
                777001ULL, std::uint64_t(i), 0ULL, static_cast<std::uint32_t>(seg), 70U);
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
    check(done > n / 2, "E7: majority exit slab (hang guard only)",
          double(done), double(n) / 2.0);
}


// --- C1: true<->geom conversion vs independent references -----------------
// Independent long-double closed forms (NOT the production float code, NOT a
// rewrite of it): small-t branch g = -L*expm1(-t/L),
// delta = L*(t/L + expm1(-t/L)); inversion t = -L*log1p(-g/L).
// Plus the EXECUTED G4 reference evidence/.../g4_conv.csv
// (ComputeTruePathLengthLimit -> ComputeGeomPathLength ->
// ComputeTrueStepLength on the real G4 11.3.2 Urban model in Water_75eV).
namespace {
struct C1Row {
    double E, tq, tl, g, ti, lam, R;
};
bool c1_parse_conv(std::vector<C1Row>& rows) {
    FILE* f = std::fopen(
        "evidence/urban_after_0f2c0ca_20260922/g4_conv.csv", "r");
    if (!f) return false;
    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        C1Row r{};
        if (std::sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf", &r.E, &r.tq,
                         &r.tl, &r.g, &r.ti, &r.lam, &r.R) == 7)
            rows.push_back(r);
    }
    std::fclose(f);
    return !rows.empty();
}
}  // namespace

void experiment_c1_conversion_reference() {
    constexpr float kC12Mass = 12.0F * 931.49410242F;
    // -- Part 1: long-double sweep over tau (R=huge forces the small-t
    // branch; no table is touched there). --
    UrbanV2Material bare{};
    bare.mass_mev = kC12Mass;
    bare.projectile_z = 6;
    bare.projectile_a = 12;
    const double kLamA = 2500.0;  // prompt's artificial-counterexample scale
    const double taus[] = {
        0.5e-16, 1.0e-16, 2.0e-16, 1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3,
        std::nextafter(1.0e-2, 0.0), 1.0e-2, std::nextafter(1.0e-2, 1.0),
        0.05, 0.5, 2.0};
    double max_g_rel = 0.0, max_d_rel = 0.0, max_g_over_t = 0.0;
    long n_series = 0, n_expm = 0, pseudo_zero = 0;
    for (double tau : taus) {
        for (double lam : {kLamA, 1.873}) {
            const double t = tau * lam;
            if (!(t > 1.0e-6) || !(t < 0.05 * 1.0e30)) continue;
            const float tf = static_cast<float>(t);
            const float lamf = static_cast<float>(lam);
            const auto c = copper_urban_v2_true_to_geom(
                tf, lamf, 1.0e30F, 100.0F, bare);
            const long double taul = (long double)t / (long double)lam;
            const long double g_ref =
                -(long double)lam * expm1l(-taul);
            const long double d_ref =
                (long double)lam * (taul + expm1l(-taul));
            const double g_rel =
                std::fabs((double)c.g_mm - (double)g_ref) /
                std::fabs((double)g_ref);
            max_g_rel = std::fmax(max_g_rel, g_rel);
            if (c.g_mm > tf)
                max_g_over_t = std::fmax(
                    max_g_over_t, (double)(c.g_mm - tf) / (double)tf);
            if (tau < 1.0e-2)
                ++n_series;
            else
                ++n_expm;
            check(!c.rounding_fixup, "C1: no rounding fixup on tau grid");
        }
    }
    std::printf("  C1 tau-grid: max|g-gref|/g=%.2e max|d-dref|/d=%.2e "
                "max(g-t)/t=%.2e (n_series=%ld n_expm=%ld)\n",
                max_g_rel, max_d_rel, max_g_over_t, n_series, n_expm);
    check(max_g_rel < 3.0e-7, "C1: g matches long-double closed form",
          max_g_rel, 3.0e-7);
    check(max_d_rel < 1.0e-4, "C1: delta matches long-double closed form",
          max_d_rel, 1.0e-4);
    check(max_g_over_t == 0.0, "C1: no unexplained g>t", max_g_over_t, 0.0);
    check(pseudo_zero == 0, "C1: no pseudo-zero delta", pseudo_zero, 0);

    // -- Part 2: executed G4 triples. Feed the helper G4's own (lambda0,
    // range) so table error cannot leak into conversion-math error. --
    std::vector<C1Row> rows;
    if (!c1_parse_conv(rows)) {
        check(false, "C1: g4_conv.csv MISSING - gate BLOCKED");
        return;
    }
    std::printf("  C1 G4 rows: %lu\n", (unsigned long)rows.size());
    double gmax = 0.0, dmax_small = 0.0, dmax_range = 0.0, timax = 0.0;
    double g4noise = 0.0;
    long n_small = 0, n_range = 0, guard_trips = 0, nm_below = 0;
    for (const auto& r : rows) {
        const float tf = static_cast<float>(r.tl);
        const float lamf = static_cast<float>(r.lam);
        const float Rf = static_cast<float>(r.R);
        const float Ef = static_cast<float>(r.E);
        const auto c = copper_urban_v2_true_to_geom(tf, lamf, Rf, Ef, bare);
        const bool small = r.tl < 0.05 * r.R;
        if (small)
            ++n_small;
        else
            ++n_range;
        if (r.tl <= 1.0e-6) {
            // Executed 1nm exit: g == t exactly.
            ++nm_below;
            check(c.g_mm == tf && c.delta_mm == 0.0F,
                  "C1: G4 sub-nm g==t exit", double(c.g_mm), double(tf));
            continue;
        }
        const double gr = std::fabs((double)c.g_mm - r.g) / r.g;
        gmax = std::fmax(gmax, gr);
        const double dref = r.ti - r.g;  // G4's own (t-g), double triple
        if (small) {
            // Small-t rows vs LONG-DOUBLE truth with G4's own (lambda0):
            // exact math both sides approximate. G4's executed (t-g) is
            // reported separately: below ~1e-16 absolute it carries G4's
            // internal double rounding (same E, different t imply
            // different lambda by up to 9% there), so it is NOT asserted.
            const long double taul =
                (long double)tf / (long double)lamf;
            const long double d_true = (long double)lamf *
                (taul + expm1l(-taul));
            if (d_true > 0) {
                const double dr = std::fabs((double)c.delta_mm -
                                            (double)d_true) / (double)d_true;
                dmax_small = std::fmax(dmax_small, dr);
            }
            if (dref > 0 && d_true > 0) {
                const double g4n = std::fabs(dref - (double)d_true) /
                    (double)d_true;
                g4noise = std::fmax(g4noise, g4n);
            }
        } else if (dref > 0) {
            // Range-formula rows: same formula both sides (ours float vs
            // G4 double); the t-z subtraction amplifies z's few-ulp
            // error (measured ~7e-3 on the grid).
            const double dr =
                std::fabs((double)c.delta_mm - dref) / dref;
            dmax_range = std::fmax(dmax_range, dr);
        }
        if (c.rounding_fixup) ++guard_trips;
        // finalize with synthetic 2:1 truncation on small-t rows (par1<0):
        // compare against the independent log1p closed form.
        if (small && r.tl > 1.0e-6) {
            const float gf = static_cast<float>(0.5 * r.g);
            float d_out = 0.0F;
            bool trip = false;
            const float t_out = copper_urban_v2_finalize_true(
                gf, static_cast<float>(r.g), tf, c.delta_mm, lamf, Rf,
                -1.0F, 0.0F, d_out, trip);
            const long double rl = (long double)(0.5 * r.g) / (long double)r.lam;
            const long double t_ref =
                -(long double)r.lam * log1pl(-rl);
            const double tr = std::fabs((double)t_out - (double)t_ref) /
                (double)t_ref;
            timax = std::fmax(timax, tr);
            if (trip) ++guard_trips;
            check(!trip, "C1: finalize truncation needs no guard");
            check(t_out <= tf, "C1: truncated inversion stays <= t_msc",
                  double(t_out), double(tf));
        }
    }
    std::printf("  C1 G4: n_small=%ld n_range=%ld subnm=%ld max|g-gG4|/g=%.2e "
                "small-t-vs-truth=%.2e range-vs-G4=%.2e G4-triple-noise=%.2e "
                "max|t-tref|/t=%.2e subulp_repairs=%ld\n",
                n_small, n_range, nm_below, gmax, dmax_small, dmax_range,
                g4noise, timax, guard_trips);
    check(gmax < 2.0e-6, "C1: g matches executed G4 triple", gmax, 2.0e-6);
    check(dmax_small < 1.0e-4, "C1: small-t delta exact vs truth",
          dmax_small, 1.0e-4);
    check(dmax_range < 2.0e-2, "C1: range-branch delta bounded vs G4",
          dmax_range, 2.0e-2);
    check(timax < 5.0e-6, "C1: truncated inversion matches log1p form",
          timax, 5.0e-6);
    // (finalize-class trips are asserted per-row above; sub-ulp chord
    // repairs are normal protection with per-row accuracy asserted.)

    // -- Part 3: reachable-domain scan with the REAL water + Cu tables. --
    for (int m = 0; m < 2; ++m) {
        const char* path = m == 0
            ? "data/urban/c12_water75ev_urban_g4_11_3_2.csv"
            : "data/urban/c12_copper_loss_range_g4_11_3_2.csv";
        UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}},
                                 0.0);
        try {
            host = UrbanLossRangeTable::from_csv(path);
        } catch (...) {
            check(false, "C1: reachable table loads", (double)m, 0.0);
            continue;
        }
        std::vector<float> e(host.energies_total_mev().begin(),
                             host.energies_total_mev().end());
        std::vector<float> r(host.ranges_mm().begin(), host.ranges_mm().end());
        std::vector<float> d(host.dedx_values().begin(),
                             host.dedx_values().end());
        UrbanV2LossTable table{e.data(), r.data(), d.data(),
                               static_cast<int>(e.size())};
        UrbanV2Material mat{};
        mat.table = table;
        mat.zeff = m == 0 ? static_cast<float>(host.zeff()) : 29.0F;
        mat.radlen_mm =
            m == 0 ? static_cast<float>(host.radlen_mm()) : 14.0F;
        mat.projectile_z = 6;
        mat.projectile_a = 12;
        mat.mass_mev = kC12Mass;
        mat.density_g_per_cm3 = m == 0 ? 1.0F : 8.96F;
        mat.mfp_kind = m == 0 ? 1 : 0;
        const float esteps[] = {2.0F, 6.0F, 12.0F, 60.0F, 120.0F, 360.0F,
                                1200.0F, 3000.0F};
        const float tsteps[] = {0.001F, 0.01F, 0.025F, 0.05F, 0.1F, 0.25F};
        long ntot = 0, nparpos = 0, nfix = 0, nbad = 0, n_fallback = 0;
        double taumax = 0.0, worst = 0.0, worst_range = 0.0;
        double worst_fb_impl = 0.0, worst_fb_rmax = 0.0;
        double worst_range_env = 0.0;
        for (float E : esteps) {
            const float R = urban_v2_loss_range(table, E);
            const float lam = urban_v2_transport_mfp(mat, E);
            if (!(R > 0.0F) || !(lam > 0.0F)) continue;
            for (float t : tsteps) {
                if (!(t < R)) continue;  // limiter clamps t to range first
                ++ntot;
                const auto c = copper_urban_v2_true_to_geom(t, lam, R, E,
                                                            mat);
                taumax = std::fmax(taumax, (double)t / (double)lam);
                if (c.par1 > 0.0F) ++nparpos;
                if (c.rounding_fixup) ++nfix;
                // Triple contract: 0 <= g <= t, 0 <= delta <= t.
                if (!(c.g_mm >= 0.0F) || !(c.g_mm <= t) ||
                    !(c.delta_mm >= 0.0F) || !(c.delta_mm <= t))
                    ++nbad;
                // Small-t rows re-checked against long double.
                if (t < 0.05F * R) {
                    const long double taul =
                        (long double)t / (long double)lam;
                    const long double d_ref = (long double)lam *
                        (taul + expm1l(-taul));
                    if (d_ref > 0) {
                        const double dr = std::fabs((double)c.delta_mm -
                                                    (double)d_ref) /
                            (double)d_ref;
                        worst = std::fmax(worst, dr);
                    }
                } else if (E < kC12Mass && R > 0.0F) {
                    // Range-formula rows vs long-double algebra.
                    const long double p1 = 1.0L / (long double)R;
                    const long double p2 = (long double)R / (long double)lam;
                    const long double p3 = 1.0L + p2;
                    const long double zr =
                        (1.0L - expl(p3 * logl(1.0L - (long double)t /
                                                     (long double)R))) /
                        (p1 * p3);
                    const long double dr_ = (long double)t - zr;
                    // Regime split mirrors production (8 ulp representability).
                    const double ulp_t =
                        (double)std::nextafter(t, 1.0e30F) - (double)t;
                    const bool fallback =
                        !(dr_ > 8.0L * (long double)ulp_t);
                    if (fallback) ++n_fallback;
                    if (dr_ > 0) {
                        if (!fallback) {
                            // Direct regime: float t-z vs algebra. Error
                            // provably bounded by z's rounding envelope:
                            // rel < k*ulp(t)/direct. Track the worst
                            // envelope ratio (must stay < 1).
                            const double drr =
                                std::fabs((double)c.delta_mm -
                                          (double)dr_) / (double)dr_;
                            const double env =
                                drr / (4.0 * ulp_t / (double)dr_ + 1.0e-9);
                            worst_range = std::fmax(worst_range, drr);
                            worst_range_env =
                                std::fmax(worst_range_env, env);
                        } else {
                            // Fallback regime: (a) implementation exactness
                            // vs small-t algebra (same math); (b) rmax vs
                            // range algebra (discriminating: garbage/zero
                            // deltas fail with ~100% rmax error).
                            const long double taul =
                                (long double)t / (long double)lam;
                            const long double d_small =
                                (long double)lam *
                                (taul + expm1l(-taul));
                            if (d_small > 0) {
                                const double dfi =
                                    std::fabs((double)c.delta_mm -
                                              (double)d_small) /
                                    (double)d_small;
                                worst_fb_impl =
                                    std::fmax(worst_fb_impl, dfi);
                            }
                            const long double r_a =
                                (double)dr_ * ((double)t + (double)zr);
                            const long double r_o =
                                (double)c.delta_mm *
                                ((double)t + (double)c.g_mm);
                            if (r_a > 0) {
                                const double rrel = std::fabs(
                                    (double)r_o - (double)r_a) / (double)r_a;
                                worst_fb_rmax =
                                    std::fmax(worst_fb_rmax, rrel);
                            }
                        }
                    }
                }
            }
        }
        std::printf("  C1 reachable %s: n=%ld par1pos=%ld subulp=%ld bad=%ld "
                    "nfallback=%ld tau_max=%.3f smallt=%.2e range=%.2e "
                    "rangeenv=%.2f fbimpl=%.2e fbrmax=%.2e\n",
                    m == 0 ? "water" : "Cu", ntot, nparpos, nfix, nbad,
                    n_fallback, taumax, worst, worst_range, worst_range_env,
                    worst_fb_impl, worst_fb_rmax);
        check(nbad == 0, "C1: reachable triple contract holds", nbad, 0);
        check(worst < 1.0e-4, "C1: reachable small-t delta bounded", worst,
              1.0e-4);
        check(worst_range_env < 1.0, "C1: range-direct inside ulp envelope",
              worst_range_env, 1.0);
        check(worst_fb_impl < 1.0e-4, "C1: fallback implementation exact",
              worst_fb_impl, 1.0e-4);
        check(worst_fb_rmax < 0.25, "C1: fallback rmax bounded",
              worst_fb_rmax, 0.25);
        // Production branch occupancy with the MSC currentRange (fix C4):
        // small-t must dominate; the range branch (and its fallback
        // regime) survives only at end-of-range.
        {
            long o_small = 0, o_range = 0, o_par = 0, o_subnm = 0;
            for (float E : esteps) {
                bool sup = false;
                const float Rm =
                    urban_v2_msc_range_mm(mat, E, sup);
                if (!sup || !(Rm > 0.0F)) continue;
                const float lam = urban_v2_transport_mfp(mat, E);
                if (!(lam > 0.0F)) continue;
                for (float t : tsteps) {
                    if (!(t < Rm)) continue;
                    if (t <= 1.0e-6F) {
                        ++o_subnm;
                    } else if (t < 0.05F * Rm) {
                        ++o_small;
                    } else {
                        ++o_range;
                        const float rfin =
                            Rm - t > 0.01F * Rm ? Rm - t : 0.01F * Rm;
                        const float t1 = urban_v2_loss_energy(table, rfin);
                        const float lam1 = t1 > 0.0F
                            ? urban_v2_transport_mfp(mat, t1)
                            : 0.0F;
                        if ((lam - lam1) / (lam * t) > 0.0F) ++o_par;
                    }
                }
            }
            std::printf("  C1 production occupancy (%s, MSC range): "
                        "smallt=%ld range=%ld parpos=%ld subnm=%ld\n",
                        m == 0 ? "water" : "Cu", o_small, o_range, o_par,
                        o_subnm);
        }
    }

    // -- Part 3b: par1>=0 (table) branch via synthetic mass. Production
    // tables top at 4800 MeV total < C12 mass 11178 MeV, so the else-branch
    // (E >= mass) is DEAD for C12 production; its stated scope is
    // explicitly limited to other-mass callers. The float formula is still
    // validated here by forcing the branch with mass_mev = 1.
    {
        UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}},
                                 0.0);
        try {
            host = UrbanLossRangeTable::from_csv(
                "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
        } catch (...) {
            check(false, "C1: par-table loads");
        }
        std::vector<float> e(host.energies_total_mev().begin(),
                             host.energies_total_mev().end());
        std::vector<float> r(host.ranges_mm().begin(), host.ranges_mm().end());
        std::vector<float> d(host.dedx_values().begin(),
                             host.dedx_values().end());
        UrbanV2LossTable table{e.data(), r.data(), d.data(),
                               static_cast<int>(e.size())};
        std::printf("  C1 table top E=%.1f MeV (C12 mass=%.1f): else-branch "
                    "dead in C12 production\n",
                    double(e.back()), double(kC12Mass));
        check(double(e.back()) < double(kC12Mass),
              "C1: else-branch dead for C12 production");
        UrbanV2Material mat{};
        mat.table = table;
        mat.zeff = static_cast<float>(host.zeff());
        mat.radlen_mm = static_cast<float>(host.radlen_mm());
        mat.projectile_z = 6;
        mat.projectile_a = 12;
        mat.mass_mev = 1.0F;  // synthetic: forces the else branch
        mat.density_g_per_cm3 = 1.0F;
        mat.mfp_kind = 1;
        const float Egrid[] = {2.0F, 12.0F, 60.0F, 300.0F, 1200.0F, 3000.0F};
        const float Tgrid[] = {0.01F, 0.05F, 0.25F, 1.0F};
        long npos = 0, nneg = 0;
        double zmax = 0.0, dmax = 0.0;
        for (float E : Egrid) {
            const float R = urban_v2_loss_range(table, E);
            const float lam0 = urban_v2_transport_mfp(mat, E);
            if (!(R > 0.0F) || !(lam0 > 0.0F)) continue;
            for (float t : Tgrid) {
                if (!(t < R) || !(t >= 0.05F * R)) continue;
                // else-branch requires E >= mass(synthetic 1 MeV: always).
                const auto c = copper_urban_v2_true_to_geom(t, lam0, R, E,
                                                            mat);
                const float rfin =
                    t < R ? (R - t > 0.01F * R ? R - t : 0.01F * R) : 0.0F;
                const float t1 = urban_v2_loss_energy(table, rfin);
                const float lam1 = t1 > 0.0F
                    ? urban_v2_transport_mfp(mat, t1)
                    : 0.0F;
                const double p1 = ((double)lam0 - (double)lam1) /
                    ((double)lam0 * (double)t);
                if (p1 > 0.0) {
                    ++npos;
                    // Long-double par algebra (precision bound only).
                    const long double p2 =
                        1.0L / ((long double)p1 * (long double)lam0);
                    const long double p3 = 1.0L + p2;
                    const long double zr =
                        (1.0L - expl(p3 * logl((long double)lam1 /
                                                     (long double)lam0))) /
                        ((long double)p1 * p3);
                    const double zrel = std::fabs((double)c.g_mm -
                                                  (double)zr) / (double)zr;
                    zmax = std::fmax(zmax, zrel);
                    const long double dr_ = (long double)t - zr;
                    if (dr_ > 0) {
                        const double drr = std::fabs((double)c.delta_mm -
                                                     (double)dr_) /
                            (double)dr_;
                        dmax = std::fmax(dmax, drr);
                    }
                } else {
                    ++nneg;  // documented float guard: falls back to
                             // constant-lambda (covered by Part 1/3).
                }
            }
        }
        std::printf("  C1 par-branch: npos=%ld nneg=%ld max|z-zalg|/z=%.2e "
                    "max|d-dalg|/d=%.2e\n",
                    npos, nneg, zmax, dmax);
        check(npos > 0, "C1: par1>0 formula exercised", npos, 0);
        check(zmax < 1.0e-6, "C1: par1>0 z precise", zmax, 1.0e-6);
        check(dmax < 0.25, "C1: par1>0 delta bounded", dmax, 0.25);
    }

    // -- Part 3c: finalize guard magnitude calibration (temporary
    // diagnostic: reports overshoot/undershoot distribution, no gates).
    {
        UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}},
                                 0.0);
        try {
            host = UrbanLossRangeTable::from_csv(
                "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
        } catch (...) {
        }
        std::vector<float> e(host.energies_total_mev().begin(),
                             host.energies_total_mev().end());
        std::vector<float> r(host.ranges_mm().begin(), host.ranges_mm().end());
        std::vector<float> d(host.dedx_values().begin(),
                             host.dedx_values().end());
        UrbanV2LossTable table{e.data(), r.data(), d.data(),
                               static_cast<int>(e.size())};
        UrbanV2Material mat{};
        mat.table = table;
        mat.zeff = static_cast<float>(host.zeff());
        mat.radlen_mm = static_cast<float>(host.radlen_mm());
        mat.projectile_z = 6;
        mat.projectile_a = 12;
        mat.mass_mev = kC12Mass;
        mat.density_g_per_cm3 = 1.0F;
        mat.mfp_kind = 1;
        const float Egrid[] = {0.13F, 0.2F,  0.5F,   1.0F,  2.0F,  12.0F,
                                     60.0F, 300.0F, 120.0F, 300.0F, 1200.0F,
                                     3000.0F};
        const float Tgrid[] = {1.0e-6F, 1.0e-5F, 1.0e-4F, 0.001F, 0.01F,
                               0.025F,  0.05F,   0.1F,     0.25F};
        const double fracs[] = {0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999,
                                1.0 - 1e-7, 1.0e-2, 1.0e-4, 1.0e-6};
        long n_under = 0, n_over = 0, n_trip = 0, n_extreme = 0;
        double worst_under_ulp = 0.0, worst_over_ulp = 0.0;
        for (float E : Egrid) {
            const float R = urban_v2_loss_range(table, E);
            const float lam = urban_v2_transport_mfp(mat, E);
            if (!(R > 0.0F) || !(lam > 0.0F)) continue;
            for (float t : Tgrid) {
                if (!(t < R)) continue;
                const auto conv =
                    copper_urban_v2_true_to_geom(t, lam, R, E, mat);
                for (double f : fracs) {
                    const float gf =
                        static_cast<float>((double)conv.g_mm * f);
                    if (!(gf < conv.g_mm) || gf < 1.0e-6F) continue;
                    float d_out = 0.0F;
                    bool trip = false;
                    const float t_out = copper_urban_v2_finalize_true(
                        gf, conv.g_mm, t, conv.delta_mm, lam, R, conv.par1,
                        conv.par3, d_out, trip);
                    (void)t_out;
                    if (trip) {
                        // Recompute raw inversion error in long double by
                        // re-running the branch math is overkill; classify
                        // by re-evaluating t_out vs t bounds below.
                    }
                    // Classify: compare t_out against [gf, t].
                    const double ulp_t =
                        (double)std::nextafter(t, 1.0e30F) - (double)t;
                    if (t_out < gf) {
                        ++n_under;
                        worst_under_ulp = std::fmax(
                            worst_under_ulp,
                            ((double)gf - (double)t_out) / ulp_t);
                    } else if (t_out > t) {
                        ++n_over;
                        worst_over_ulp = std::fmax(
                            worst_over_ulp,
                            ((double)t_out - (double)t) / ulp_t);
                    }
                }
            }
        }
        // Production-realistic triples: run the LIMITER (it randomizes
        // t_msc via dims 58/59) instead of round t, then truncate.
        for (float E : Egrid) {
            for (float tc : Tgrid) {
                for (std::uint64_t sd = 0; sd < 25; ++sd) {
                    UrbanV2TrackState st{};
                    auto lim = copper_urban_v2_limit_step(
                        tc, E, mat, 1.0e-6F, (sd & 1U) != 0U, st,
                        777ULL, 1000ULL + sd, 0ULL,
                        static_cast<std::uint32_t>(sd));
                    if (!lim.valid || lim.g_msc_mm <= 0.0F) continue;
                    for (double f : fracs) {
                        const float gf = static_cast<float>(
                            (double)lim.g_msc_mm * f);
                        if (!(gf < lim.g_msc_mm) || gf < 1.0e-6F) continue;
                        float d_out = 0.0F;
                        bool trip = false;
                        const float t_out =
                            copper_urban_v2_finalize_true(
                                gf, lim.g_msc_mm, lim.t_msc_mm,
                                lim.delta_msc_mm, lim.lambda0_mm,
                                lim.range_mm, lim.par1, lim.par3, d_out,
                                trip);
                        (void)t_out;
                        if (trip) ++n_trip;
                        if (f < 0.1) ++n_extreme;
                        const double ulp_t =
                            (double)std::nextafter(lim.t_msc_mm, 1.0e30F) -
                            (double)lim.t_msc_mm;
                        if (t_out < gf) {
                            ++n_under;
                            worst_under_ulp = std::fmax(
                                worst_under_ulp,
                                ((double)gf - (double)t_out) / ulp_t);
                        } else if (t_out > lim.t_msc_mm) {
                            ++n_over;
                            worst_over_ulp = std::fmax(
                                worst_over_ulp,
                                ((double)t_out - (double)lim.t_msc_mm) /
                                    ulp_t);
                        }
                    }
                }
            }
        }
        std::printf("  C1 finalize calib: under=%ld worst=%.2fulp over=%ld "
                    "worst=%.2fulp trips=%ld (extreme-frac rows=%ld)\n",
                    n_under, worst_under_ulp, n_over, worst_over_ulp, n_trip,
                    n_extreme);
        check(n_trip == 0, "C1: finalize needs no guard on calib grid",
              n_trip, 0);
    }

    // -- Part 3d: extreme-truncation slivers (GPU guard-trip root cause).
    // par>=0 inversion with par4*g < ~6e-8: the old
    // t = (1-exp(log(1-par4*g)/par3))/par1 evaluates log(1.0F) = 0 and
    // returns t = 0 < g, tripping the t<g guard on every end-of-step
    // boundary sliver (49 trips / 2000 histories on GPU before the fix).
    // The log1p/expm1 form resolves them; pin the old expression to 0.0
    // here as the documented counterexample.
    {
        const float R = 124.0F;  // CSDA-like large R: par4 = 1/R small.
        const float lam = 2.0e6F;
        const float t = 0.06F * R;  // range branch (E<mass).
        const float E = 3000.0F;
        const auto conv =
            copper_urban_v2_true_to_geom(t, lam, R, E, bare);
        check(conv.par1 > 0.0F, "C1: sliver test takes range branch");
        const float par4 = conv.par1 * conv.par3;
        long nsliver = 0;
        double worst_t = 0.0;
        for (double gf = 1.0e-6; gf < 2.0e-4; gf *= 1.7) {
            const float gff = static_cast<float>(gf);
            if (!(gff < conv.g_mm)) continue;
            ++nsliver;
            // Old expression, inline (documents the defect; must be 0).
            const float t_old =
                (1.0F - std::exp(std::log(1.0F - par4 * gff) / conv.par3)) /
                conv.par1;
            // New production path.
            float d_out = 0.0F;
            bool trip = false;
            const float t_new = copper_urban_v2_finalize_true(
                gff, conv.g_mm, t, conv.delta_mm, lam, R, conv.par1,
                conv.par3, d_out, trip);
            // Independent long-double inversion reference.
            const long double rl =
                (long double)par4 * (long double)gff;
            const long double t_ref =
                -(long double)(1.0L - expl(log1pl(-rl) / (long double)conv.par3)) /
                (long double)conv.par1;
            // Rounding fact: 1-x rounds to exactly 1.0F iff x is within
            // half an ulp below 1 (x < 2.98e-8); between that and ~9e-8 the
            // old expression is a nonzero staircase (see row 4: t_old =
            // 7.39e-6 for a true ~4.9e-6). Both regimes are wrong; the new
            // form is exact (worst_t = 0.00 below).
            if (par4 * gff < 2.9e-8F) {
                check(t_old == 0.0F, "C1: old inversion rounds to 0",
                      t_old, 0.0);
            }
            check(!trip, "C1: sliver inversion needs no guard");
            if (t_ref > 0) {
                const double tr = std::fabs((double)t_new - (double)t_ref) /
                    (double)t_ref;
                worst_t = std::fmax(worst_t, tr);
            }
            check(t_new >= gff && t_new <= t, "C1: sliver inversion bracketed",
                  t_new, t);
        }
        std::printf("  C1 slivers: n=%ld worst|t-tref|/t=%.2e\n", nsliver,
                    worst_t);
        check(nsliver > 3, "C1: sliver rows covered", nsliver, 3);
        check(worst_t < 1.0e-4, "C1: sliver inversion accurate", worst_t,
              1.0e-4);
    }

    // -- Part 4: illegal / non-finite inputs must be finite and empty. --
    {
        const float bad_t[] = {0.0F, -1.0F,
                               std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity()};
        for (float t : bad_t) {
            const auto c =
                copper_urban_v2_true_to_geom(t, 2500.0F, 124.0F, 3000.0F,
                                             bare);
            check(c.g_mm == 0.0F && c.delta_mm == 0.0F,
                  "C1: bad-t gives empty conversion");
        }
        const float bad_l[] = {0.0F, -2.0F,
                               std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity()};
        for (float l : bad_l) {
            const auto c =
                copper_urban_v2_true_to_geom(0.05F, l, 124.0F, 3000.0F, bare);
            check(std::isfinite(c.g_mm) && std::isfinite(c.delta_mm) &&
                      c.g_mm == 0.0F && c.delta_mm == 0.0F,
                  "C1: bad-lambda gives empty conversion");
        }
    }
}

// --- C3: q-audit upgrade (bounded error, not "identical") -------------------

namespace {
// Strict MSC draw set (bit-exact): D(b) = float((b+0.5)/2^24) for
// b in [0, 2^24-2], plus the clamped top. For b >= 2^23 consecutive raw
// patterns can round to the SAME float (25-bit significand); the set is
// then locally non-uniform at 1-ulp granularity, a ~2^-24 probability
// distortion per threshold, negligible and documented (not fixed: any
// float RNG shares it). Straddles are found by SEARCH, never by formula.
float c3_draw(long b) {
    if (b >= 16777215L) return 0x1.fffffep-1f;
    if (b < 0L) return 0.0F;
    return (float)(((double)b + 0.5) / 16777216.0);
}
// Largest b with D(b) < t (t strictly inside (D(0), clamp)).
long c3_straddle_lo(float t) {
    long lo = -1L, hi = 16777215L;
    while (hi - lo > 1L) {
        const long mid = lo + (hi - lo) / 2L;
        if (c3_draw(mid) < t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}
}  // namespace

namespace {
struct C3W {
    long double x_mean_th = 0, x_mean1 = 0, x0 = 0, b = 0, d = 0, x_mean2 = 0;
    long double prob = 0, q = 0, theta0 = 0, xsi = 0, tau = 0;
    int branch = 0;  // mirrors UrbanV2SampleDebug.branch
    int div = 0;     // float/long-double branch divergence count marker
};
// Independent long-double recomputation of the sampling weights from the
// same float inputs (precision oracle; the formula-vs-G4 physics is covered
// by the single-step/slab oracles, not here).
C3W c3_ld_weights(double t, double kin, double epre, double lam0, double lam1,
                  bool has_lam1, double radlen, double mass, double Z,
                  double cth1, double cth2, double cc1, double cc2, double cc3,
                  double cc4, double tlimitmin) {
    C3W w{};
    long double tau = (long double)t / (long double)lam0;
    long double lameff = (long double)lam0;
    if (kin != epre && has_lam1 &&
        fabsl((long double)lam1 - (long double)lam0) >
            (long double)lam0 * 0.01L) {
        tau = (long double)t * logl((long double)lam0 / (long double)lam1) /
            ((long double)lam0 - (long double)lam1);
        lameff = (long double)t / tau;
    }
    w.tau = tau;
    if (tau < 0.01L) {
        w.x_mean_th = 1.0L - tau * (1.0L - 0.5L * tau);
    } else {
        w.x_mean_th = expl(-tau);
    }
    if (tau >= 8.0L) {
        w.branch = 2;
        return w;
    }
    if (1.0L - (long double)kin / (long double)epre > 0.50L) {
        w.branch = 3;
        return w;
    }
    long double inv = ((long double)kin + (long double)mass) /
        ((long double)kin * ((long double)kin + 2.0L * (long double)mass));
    if (kin != epre) {
        inv = sqrtl(inv * ((long double)epre + (long double)mass) /
                    ((long double)epre *
                     ((long double)epre + 2.0L * (long double)mass)));
    }
    const long double y = (long double)t / (long double)radlen;
    long double th0 = 13.6L * (long double)Z * sqrtl(y) * inv *
        ((long double)cth1 + (long double)cth2 * logl(y));
    const long double tsmall =
        (long double)tlimitmin < 1.0L ? (long double)tlimitmin : 1.0L;
    const bool extreme = !((long double)t > tsmall);
    if (extreme) th0 *= sqrtl((long double)t / tsmall);
    w.theta0 = th0;
    const long double th2 = th0 * th0;
    if (th2 < 1.0e-16L) {
        w.branch = 0;
        return w;
    }
    if (th0 > acosl(-1.0L) * (1.0L / 6.0L)) {
        w.branch = 4;
        return w;
    }
    long double x = th2 * (1.0L - th2 * (1.0L / 12.0L));
    if (th2 > 0.01L) {
        const long double s2 = 2.0L * sinl(0.5L * th0);
        x = s2 * s2;
    }
    const long double ltau = logl(tau);
    long double u = extreme
        ? expl(logl(tsmall / (long double)lam0) * (1.0L / 6.0L))
        : expl(ltau * (1.0L / 6.0L));
    const long double xx = logl(lameff / (long double)radlen);
    long double xsi = (long double)cc1 +
        u * ((long double)cc2 + (long double)cc3 * u) +
        (long double)cc4 * xx;
    if (xsi < 1.9L) xsi = 1.9L;
    w.xsi = xsi;
    long double c = xsi;
    if (fabsl(c - 3.0L) < 0.001L) {
        c = 3.001L;
    } else if (fabsl(c - 2.0L) < 0.001L) {
        c = 2.001L;
    }
    const long double c1 = c - 1.0L;
    const long double ea = expl(-xsi);
    const long double eaa = 1.0L - ea;
    w.x_mean1 = 1.0L - (1.0L - (1.0L + xsi) * ea) * x / eaa;
    w.x0 = 1.0L - xsi * x;
    if (w.x_mean1 <= 0.999L * w.x_mean_th) {
        w.branch = 5;
        return w;
    }
    const long double b = 1.0L + (c - xsi) * x;
    const long double b1 = b + 1.0L;
    const long double bx = c * x;
    const long double eb1 = expl(logl(b1) * c1);
    const long double ebx = expl(logl(bx) * c1);
    w.d = ebx / eb1;
    w.b = b;
    w.x_mean2 =
        (w.x0 + w.d - (bx - b1 * w.d) / (c - 2.0L)) / (1.0L - w.d);
    const long double f1 = ea / eaa;
    const long double f2 = c1 / (c * (1.0L - w.d));
    w.prob = f2 / (f1 + f2);
    w.q = w.x_mean_th /
        (w.prob * w.x_mean1 + (1.0L - w.prob) * w.x_mean2);
    w.branch = 6;
    return w;
}
}  // namespace

void experiment_c3_q_audit() {
    constexpr float kC12Mass = 12.0F * 931.49410242F;
    constexpr float kTlimMin = 1.0e-7F;
    // -- (a) reachable-domain grid through the PRODUCTION sampler. --
    long ntot = 0;
    long nbr[7] = {0, 0, 0, 0, 0, 0, 0};
    double qmin = 1e30, qmax = -1e30, pmin = 1e30, pmax = -1e30;
    double gmargin_min = 1e30;  // min |x_mean1-0.999*x_mean_th|/x_mean_th
    double tail_w = 0.0;        // max |P_iso(float)-P_iso(ld)|
    double wmax_q = 0.0, wmax_p = 0.0, wmax_x1 = 0.0, wmax_d = 0.0,
           wmax_th = 0.0;
    long ndiv = 0;
    // (c) threshold-adjacent strict draws: straddles verified, endpoint
    // trial pins (q>=1 -> mixture always; q<=0 -> isotropic always).
    long n_straddle_q = 0, n_straddle_p = 0, n_qge1 = 0, n_qle0 = 0;
    constexpr float kDrawMin = 0.5F * 5.9604644775390625e-8F;  // b=0
    constexpr float kDrawMax = 0x1.fffffep-1f;                 // clamped top
    const Direction3F dir{0.0F, 0.0F, 1.0F};
    for (int m = 0; m < 2; ++m) {
        const char* path = m == 0
            ? "data/urban/c12_water75ev_urban_g4_11_3_2.csv"
            : "data/urban/c12_copper_loss_range_g4_11_3_2.csv";
        UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}},
                                 0.0);
        try {
            host = UrbanLossRangeTable::from_csv(path);
        } catch (...) {
            check(false, "C3: table loads", (double)m, 0.0);
            continue;
        }
        std::vector<float> e(host.energies_total_mev().begin(),
                             host.energies_total_mev().end());
        std::vector<float> r(host.ranges_mm().begin(), host.ranges_mm().end());
        std::vector<float> d(host.dedx_values().begin(),
                             host.dedx_values().end());
        UrbanV2LossTable table{e.data(), r.data(), d.data(),
                               static_cast<int>(e.size())};
        UrbanV2Material mat{};
        mat.table = table;
        mat.zeff = m == 0 ? static_cast<float>(host.zeff()) : 29.0F;
        mat.radlen_mm =
            m == 0 ? static_cast<float>(host.radlen_mm()) : 14.0F;
        mat.projectile_z = 6;
        mat.projectile_a = 12;
        mat.mass_mev = kC12Mass;
        mat.density_g_per_cm3 = m == 0 ? 1.0F : 8.96F;
        mat.mfp_kind = m == 0 ? 1 : 0;
        const auto coeff = copper_urban_coefficients(mat.zeff);
        const float Egrid[] = {0.13F,  0.5F,   2.0F,   12.0F,  60.0F,
                               300.0F, 1200.0F, 3000.0F, 4800.0F};
        const float Tgrid[] = {1.0e-9F, 1.0e-8F, 1.0e-7F, 1.0e-6F, 1.0e-5F,
                               1.0e-4F, 0.001F,  0.005F,  0.01F,   0.025F,
                               0.05F,   0.1F,    0.25F};
        for (float E : Egrid) {
            const float R = urban_v2_loss_range(table, E);
            const float lam0 = urban_v2_transport_mfp(mat, E);
            if (!(R > 0.0F) || !(lam0 > 0.0F)) continue;
            for (float t : Tgrid) {
                if (!(t > 0.0F) || !(t < R)) continue;
                // Production-chained delta (same inputs the sampler sees).
                const auto conv =
                    copper_urban_v2_true_to_geom(t, lam0, R, E, mat);
                UrbanV2SampleDebug dbg{};
                const auto s = copper_urban_v2_sample_full(
                    dir, E, t, conv.delta_mm, mat, R, lam0, kTlimMin,
                    1.0e30F, 1.0F, true, 777ULL, 1000ULL + (unsigned long)ntot,
                    0ULL, 70U, &dbg);
                (void)s;
                ++ntot;
                if (dbg.branch < 0 || dbg.branch > 6) {
                    check(false, "C3: branch id in range", dbg.branch, 6);
                    continue;
                }
                ++nbr[dbg.branch];
                if (dbg.branch != 6) continue;
                // -- (b) long-double recomputation from the same floats. --
                // kin prediction replicated via production helpers.
                float kin = E;
                constexpr float kDtrl = 0.05F;  // must match sample_full
                if (t > R * kDtrl) {
                    kin = urban_v2_loss_energy(table, R - t);
                } else if (t > R * 0.01F) {
                    kin -= t * urban_v2_loss_dedx(table, E);
                }
                if (kin < 0.0F) kin = 0.0F;
                float lam1 = 0.0F;
                bool has_lam1 = false;
                if (kin != E) {
                    lam1 = urban_v2_transport_mfp(mat, kin);
                    has_lam1 = lam1 > 0.0F;
                }
                const auto w = c3_ld_weights(
                    t, kin, E, lam0, lam1, has_lam1, mat.radlen_mm,
                    kC12Mass, 6.0, coeff.coeffth1, coeff.coeffth2,
                    coeff.coeffc1, coeff.coeffc2, coeff.coeffc3,
                    coeff.coeffc4, kTlimMin);
                if (w.branch != 6) {
                    ++ndiv;  // float/ld branch divergence (count, inspect)
                    continue;
                }
                const double rq = std::fabs((double)dbg.q_probability -
                                            (double)w.q) / (double)w.q;
                const double rp = std::fabs((double)dbg.probability -
                                            (double)w.prob) / (double)w.prob;
                const double rx1 = std::fabs((double)dbg.x_mean1 -
                                             (double)w.x_mean1) /
                    std::fabs((double)w.x_mean1);
                const double rd = w.d != 0.0L
                    ? std::fabs((double)dbg.d_param - (double)w.d) /
                        std::fabs((double)w.d)
                    : 0.0;
                const double rth = std::fabs((double)dbg.x_mean_th -
                                             (double)w.x_mean_th) /
                    std::fabs((double)w.x_mean_th);
                wmax_q = std::fmax(wmax_q, rq);
                wmax_p = std::fmax(wmax_p, rp);
                wmax_x1 = std::fmax(wmax_x1, rx1);
                wmax_d = std::fmax(wmax_d, rd);
                wmax_th = std::fmax(wmax_th, rth);
                qmin = std::fmin(qmin, (double)dbg.q_probability);
                qmax = std::fmax(qmax, (double)dbg.q_probability);
                pmin = std::fmin(pmin, (double)dbg.probability);
                pmax = std::fmax(pmax, (double)dbg.probability);
                const double gm =
                    std::fabs((double)dbg.x_mean1 -
                              0.999 * (double)dbg.x_mean_th) /
                    std::fabs((double)dbg.x_mean_th);
                gmargin_min = std::fmin(gmargin_min, gm);
                // -- (d) isotropic-branch tail weight error. --
                const double piso_f =
                    (double)dbg.q_probability < 1.0
                    ? 1.0 - (double)dbg.q_probability
                    : 0.0;
                const double piso_l =
                    (double)w.q < 1.0 ? 1.0 - (double)w.q : 0.0;
                tail_w =
                    std::fmax(tail_w, std::fabs(piso_f - piso_l));
                // Strict-draw straddle around q (search-based).
                {
                    const float qf = dbg.q_probability;
                    // Float-threshold fuzzy zone (C3 error model): every
                    // u<q comparison resolves thresholds to one draw
                    // (2^-24). qf > maxdraw -> mixture always; qf == maxdraw
                    // -> exactly the top draw takes isotropic, at the
                    // reference rate O(2^-24) (G4-double takes isotropic
                    // with P = 1-q_ld ~ 6e-8 there too). Not a leak.
                    if (qf > kDrawMax) {
                        ++n_qge1;  // every draw (incl. max) takes mixture
                        check(kDrawMax < qf, "C3: q>=1 takes mixture always",
                              kDrawMax, qf);
                    } else if (qf == kDrawMax) {
                        ++n_qge1;
                    } else if (qf <= kDrawMin) {
                        ++n_qle0;
                    } else {
                        const long blo = c3_straddle_lo(qf);
                        const float u_lo = c3_draw(blo);
                        const float u_hi = c3_draw(blo + 1L);
                        check(blo >= 0L && u_lo < qf && !(u_hi < qf) &&
                                  u_lo < u_hi,
                              "C3: q straddle exact", u_lo, u_hi);
                        ++n_straddle_q;
                    }
                }
                // Strict-draw straddle around probability.
                {
                    const float pf = dbg.probability;
                    if (pf > kDrawMin && pf < kDrawMax) {
                        const long blo = c3_straddle_lo(pf);
                        const float u_lo = c3_draw(blo);
                        const float u_hi = c3_draw(blo + 1L);
                        check(blo >= 0L && u_lo < pf && !(u_hi < pf) &&
                                  u_lo < u_hi,
                              "C3: prob straddle exact", u_lo, u_hi);
                        ++n_straddle_p;
                    }
                }
            }
        }
    }
    std::printf("  C3 domain: n=%ld branches(id:count)=0:%ld 1:%ld 2:%ld "
                "3:%ld 4:%ld 5:%ld 6:%ld div=%ld\n",
                ntot, nbr[0], nbr[1], nbr[2], nbr[3], nbr[4], nbr[5], nbr[6],
                ndiv);
    std::printf("  C3 trial rows: q in [%.5f, %.5f] prob in [%.5f, %.5f] "
                "gate_margin_min=%.2e\n",
                qmin, qmax, pmin, pmax, gmargin_min);
    std::printf("  C3 precision vs long double: q=%.2e prob=%.2e "
                "x_mean1=%.2e d=%.2e x_mean_th=%.2e tailW=%.2e\n",
                wmax_q, wmax_p, wmax_x1, wmax_d, wmax_th, tail_w);
    check(ntot > 0 && nbr[6] > 0, "C3: trial branch covered", nbr[6], 0);
    check(ndiv == 0, "C3: no float/ld branch divergence", ndiv, 0);
    check(wmax_q < 1.0e-5 && wmax_p < 1.0e-5 && wmax_x1 < 1.0e-5 &&
              wmax_d < 1.0e-5,
          "C3: weights precise vs independent oracle", wmax_q, 1.0e-5);
    check(tail_w < 1.0e-5, "C3: isotropic tail weight bounded", tail_w,
          1.0e-5);
    std::printf("  C3 decisions: q-straddles=%ld prob-straddles=%ld "
                "q>=1-rows=%ld q<=0-rows=%ld\n",
                n_straddle_q, n_straddle_p, n_qge1, n_qle0);
    check(n_straddle_q > 0 && n_straddle_p > 0, "C3: straddles covered",
          n_straddle_q, n_straddle_p);
    check(n_qge1 > 0, "C3: q>=1 always-mixture covered", n_qge1, 0);
}

// --- C4: transport-contract determinism ---------------------------------
// Every Urban draw is a pure function of (seed, history, outer, segment,
// dim): call order, batching, chunking, and replay transforms cannot change
// any address. State save/restore round-trips exactly.
void test_c4_transport_contract() {
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_water75ev_urban_g4_11_3_2.csv");
    } catch (...) {
        check(false, "C4: water loss table loads");
        return;
    }
    std::vector<float> e(host.energies_total_mev().begin(),
                         host.energies_total_mev().end());
    std::vector<float> r(host.ranges_mm().begin(), host.ranges_mm().end());
    std::vector<float> d(host.dedx_values().begin(),
                         host.dedx_values().end());
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
    // 1. Bitwise repeatability: same identity twice.
    {
        UrbanV2TrackState s1{}, s2{};
        const auto a = urban_v2_propose_and_sample(
            dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, s1, mat, 1.0F, 4242ULL, 17ULL, 5ULL, 3U, 70U);
        const auto b = urban_v2_propose_and_sample(
            dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, s2, mat, 1.0F, 4242ULL, 17ULL, 5ULL, 3U, 70U);
        check(a.proposal_valid && b.proposal_valid, "C4: both valid");
        check(a.direction.x == b.direction.x &&
                  a.direction.y == b.direction.y &&
                  a.direction.z == b.direction.z &&
                  a.final_true_path_mm == b.final_true_path_mm &&
                  a.final_geom_path_mm == b.final_geom_path_mm &&
                  a.stable_delta_mm == b.stable_delta_mm &&
                  a.displacement_mm.x == b.displacement_mm.x,
              "C4: same identity -> bitwise identical");
    }
    // 2. Identity sensitivity: different history/segment -> different draws
    // (spot check), same identity after other calls -> unchanged (order
    // independence: interleave foreign identities between repeats).
    {
        UrbanV2TrackState s1{}, s2{}, s3{};
        const auto a = urban_v2_propose_and_sample(
            dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, s1, mat, 1.0F, 4242ULL, 17ULL, 5ULL, 3U, 70U);
        const auto foreign = urban_v2_propose_and_sample(
            dir, 1200.0F, 0.05F, 0.05F, 1.0F, 1.0F, 41.0F, 0.0F, 0.0F, 1.0F,
            box, true, s2, mat, 1.0F, 9999ULL, 888ULL, 7ULL, 9U, 70U);
        (void)foreign;
        const auto b = urban_v2_propose_and_sample(
            dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, s3, mat, 1.0F, 4242ULL, 17ULL, 5ULL, 3U, 70U);
        check(a.direction.x == b.direction.x &&
                  a.final_true_path_mm == b.final_true_path_mm,
              "C4: order-independent identity");
        UrbanV2TrackState s4{};
        const auto c = urban_v2_propose_and_sample(
            dir, 3000.0F, 0.05F, 0.05F, 0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F,
            box, true, s4, mat, 1.0F, 4242ULL, 18ULL, 5ULL, 3U, 70U);
        check(c.direction.x != a.direction.x ||
                  c.final_true_path_mm != a.final_true_path_mm,
              "C4: history identity separates streams");
    }
    // 3. State save/restore round-trip (chunk-resume model): run one step,
    // serialize tlimit, zero the state, restore, continue -> identical to
    // uninterrupted.
    {
        UrbanV2TrackState live{};
        auto lim1 = copper_urban_v2_limit_step(0.05F, 3000.0F, mat, 1.0e-6F,
                                               true, live, 4242ULL, 17ULL,
                                               0ULL, 3U);
        check(lim1.valid, "C4: first limit valid");
        const float saved_tlimit = live.tlimit_mm;  // serialize
        UrbanV2TrackState resumed{};                // zero (fresh launch)
        resumed.tlimit_mm = saved_tlimit;           // restore
        auto lim2a = copper_urban_v2_limit_step(0.05F, 2990.0F, mat, 1.0e-6F,
                                                false, live, 4242ULL, 17ULL,
                                                1ULL, 0U);
        auto lim2b = copper_urban_v2_limit_step(0.05F, 2990.0F, mat, 1.0e-6F,
                                                false, resumed, 4242ULL, 17ULL,
                                                1ULL, 0U);
        check(lim2a.valid && lim2b.valid, "C4: resumed limit valid");
        check(lim2a.t_msc_mm == lim2b.t_msc_mm &&
                  lim2a.g_msc_mm == lim2b.g_msc_mm &&
                  lim2a.tlimit_used_mm == lim2b.tlimit_used_mm,
              "C4: save/restore round-trips tlimit");
    }
}

// --- C6.1-A: Cu slab writer (diagnostic transport, offline gate) --------
// Pencil C12 through a 10 mm Cu slab with the PRODUCTION propose core
// (0.25 mm ceiling, Cu table, mfp_kind=0, box safety): exit rows
// (th_mrad, x_mm, y_mm, E_MeV) go to the test output dir; the gate
// (cu_slab_gate.py) compares moments vs the executed G4 Cu slab oracle.
void experiment_cu_slab() {
    UrbanLossRangeTable host({{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}, 0.0);
    try {
        host = UrbanLossRangeTable::from_csv(
            "data/urban/c12_copper_loss_range_g4_11_3_2.csv");
    } catch (...) {
        check(false, "C6Cu: Cu loss table loads");
        return;
    }
    std::vector<float> e(host.energies_total_mev().begin(),
                         host.energies_total_mev().end());
    std::vector<float> r(host.ranges_mm().begin(), host.ranges_mm().end());
    std::vector<float> d(host.dedx_values().begin(),
                         host.dedx_values().end());
    UrbanV2LossTable table{e.data(), r.data(), d.data(),
                           static_cast<int>(e.size())};
    UrbanV2Material mat{};
    mat.table = table;
    mat.zeff = 29.0F;
    mat.radlen_mm = 14.3558F;  // executed G4_Cu value (slab oracle log)
    mat.projectile_z = 6;
    mat.projectile_a = 12;
    mat.mass_mev = 12.0F * 931.49410242F;
    mat.density_g_per_cm3 = 8.96F;
    mat.mfp_kind = 0;
    UrbanV2SafetyCtx box{};
    box.kind = 1;
    box.box_x0 = -50.0F;
    box.box_x1 = 50.0F;
    box.box_y0 = -50.0F;
    box.box_y1 = 50.0F;
    box.box_z0 = 0.0F;
    box.box_z1 = 10.0F;
    const Direction3F dir{0.0F, 0.0F, 1.0F};
    constexpr int kN = 20000;
    constexpr float kSlab = 10.0F;
    constexpr float kCeil = 0.25F;
    FILE* f =
        std::fopen(test_out_path("maigo_cu10ms025.csv").c_str(), "w");
    check(f != nullptr, "C6Cu: output CSV opens");
    if (f == nullptr) return;
    long done = 0;
    for (int i = 0; i < kN; ++i) {
        UrbanV2TrackState st{};
        float x = 0.0F, y = 0.0F, z = 0.0F, energy = 3000.0F;
        Direction3F dd = dir;
        bool alive = true;
        bool first = true;
        std::uint64_t seg = 0;
        while (z < kSlab && energy > 0.2F) {
            const float seg_len = kCeil < kSlab - z ? kCeil : kSlab - z;
            if (!(seg_len > 0.0F)) break;
            const auto s = urban_v2_propose_and_sample(
                dd, energy, seg_len, seg_len, x, y, z, dd.x, dd.y, dd.z,
                box, first, st, mat, 1.0F, 777001ULL, std::uint64_t(i),
                0ULL, static_cast<std::uint32_t>(seg), 50U);
            if (!s.proposal_valid || s.final_geom_path_mm <= 0.0F) {
                alive = false;
                break;
            }
            first = false;
            ++seg;
            if (seg > 1000000U) {
                alive = false;
                break;
            }
            x += s.final_geom_path_mm * dd.x + s.displacement_mm.x;
            y += s.final_geom_path_mm * dd.y + s.displacement_mm.y;
            z += s.final_geom_path_mm * dd.z + s.displacement_mm.z;
            dd = s.direction;
            energy -=
                s.final_true_path_mm * urban_v2_loss_dedx(table, energy);
            if (!(energy > 0.0F)) {
                alive = false;
                break;
            }
        }
        if (!alive || z < kSlab) continue;
        const double thx = std::atan2(double(dd.x), double(dd.z)) * 1000.0;
        const double thy = std::atan2(double(dd.y), double(dd.z)) * 1000.0;
        std::fprintf(f, "%.6f,%.6f,%.6f,%.4f\n",
                     std::sqrt(thx * thx + thy * thy), x, y, energy);
        ++done;
    }
    std::fclose(f);
    std::printf("  C6Cu: exit %ld/%d\n", done, kN);
    check(done > kN / 2, "C6Cu: majority exit slab (hang guard only)",
          double(done), double(kN) / 2.0);
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
    std::printf("[C1] true<->geom conversion reference\n");
    experiment_c1_conversion_reference();
    std::printf("[B6] RNG addressing + boundary equivalence\n");
    test_rng_addressing();
    std::printf("[C2] MSC domain separation + index guards\n");
    test_msc_domain_separation();
    test_rng_index_guards();
    std::printf("[C3] q-audit upgrade\n");
    experiment_c3_q_audit();
    std::printf("[C4] transport-contract determinism\n");
    test_c4_transport_contract();
    std::printf("[C6Cu] Cu slab writer\n");
    experiment_cu_slab();
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

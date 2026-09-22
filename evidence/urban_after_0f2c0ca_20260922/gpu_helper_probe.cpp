// GPU-vs-host equivalence probe for Urban production helpers (fix C5).
// Runs true_to_geom / finalize_true / msc draws on the GPU and on the host
// (same header) and compares bitwise. Any device-libm divergence would show
// here, separating numerics questions from physics questions.
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "carbon/rng.hpp"
#include "carbon/multiple_scattering.hpp"

namespace carbon {
// Stub for transport-TU context (never called by the probed helpers;
// the tested functions' transitive closure is stub-free production code).
inline float minibeam_isotropic_safety_at_point(
    float, float, float, float, float, float, float, float, float, float,
    float, float, float) noexcept {
    return 1.0e30F;
}
}  // namespace carbon

namespace carbon {
namespace {
#include "detail/sycl_device_math.inc"
}  // namespace
}  // namespace carbon

namespace {
struct Case {
    float t, lam, R, E;
    float gf, gp, tm, dm, p1, p3;
};
struct Out {
    float g, d, t;
    bool fix, trip;
    unsigned r0, r1;
};
}  // namespace

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    // Table-free branches: R huge forces small-t; E<mass+small R forces the
    // range branch (no table); finalize needs no table.
    // Realistic truncated triples, derived host-side from the production
    // conversion (same values both sides invert).
    carbon::UrbanV2Material m0{};
    m0.mass_mev = 12.0F * 931.49410242F;
    m0.projectile_z = 6;
    m0.projectile_a = 12;
    const auto rc = carbon::copper_urban_v2_true_to_geom(0.05F, 347.0F, 0.833F,
                                                         6.0F, m0);
    const Case cases[] = {
        {0.05F, 2500.0F, 1.0e30F, 100.0F, 0, 0, 0, 0, 0, 0},  // small-t
        {0.05F, 347.0F, 0.833F, 6.0F, 0, 0, 0, 0, 0, 0},      // range
        {0.25F, 1.02e6F, 3.75F, 360.0F, 0, 0, 0, 0, 0, 0},    // range, sub-ulp
        {1.0e-4F, 2.775e6F, 83.33F, 600.0F, 0, 0, 0, 0, 0, 0},  // 1nm+
        {0.05F, 347.0F, 0.833F, 6.0F, 0.025F, 0, 0, 0, 0, 0},   // finalize par<0
        {0.05F, 347.0F, 0.833F, 6.0F, rc.g_mm * 0.5F, rc.g_mm, 0.05F,
         rc.delta_mm, rc.par1, rc.par3},  // truncated par>=0
        {0.05F, 347.0F, 0.833F, 6.0F, 2.0e-6F, rc.g_mm, 0.05F, rc.delta_mm,
         rc.par1, rc.par3},  // sliver-scale truncation
    };
    constexpr int N = sizeof(cases) / sizeof(cases[0]);
    Out dev[N]{};
    {
        sycl::buffer<Out, 1> b(dev, N);
        sycl::buffer<Case, 1> c(const_cast<Case*>(cases), N);
        q.submit([&](sycl::handler& h) {
            auto ao = b.get_access<sycl::access::mode::write>(h);
            auto ac = c.get_access<sycl::access::mode::read>(h);
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                const Case cc = ac[i];
                carbon::UrbanV2Material mat{};
                mat.mass_mev = 12.0F * 931.49410242F;
                mat.projectile_z = 6;
                mat.projectile_a = 12;
                const auto conv =
                    carbon::copper_urban_v2_true_to_geom(
                        cc.t, cc.lam, cc.R, cc.E, mat);
                ao[i].g = conv.g_mm;
                ao[i].d = conv.delta_mm;
                ao[i].fix = conv.rounding_fixup;
                float dd = 0.0F;
                bool tr = false;
                ao[i].t = carbon::copper_urban_v2_finalize_true(
                    cc.gf > 0.0F ? cc.gf : conv.g_mm,
                    cc.gp > 0.0F ? cc.gp : conv.g_mm,
                    cc.tm > 0.0F ? cc.tm : cc.t, conv.delta_mm, cc.lam,
                    cc.R < 1.0e29F ? cc.R : 124.0F, cc.p1, cc.p3, dd, tr);
                ao[i].d = dd;
                ao[i].trip = tr;
                ao[i].r0 = carbon::rng::msc_random_u32(11ULL, 22ULL,
                                                       (std::uint64_t)i, 70U);
                ao[i].r1 = carbon::rng::msc_random_u32(11ULL, 22ULL,
                                                       (std::uint64_t)i, 71U);
            });
        }).wait_and_throw();
    }
    int mism = 0;
    double genv_max = 0.0, denv_max = 0.0;
    // Envelope gate (mirrors the C1 4-ulp envelope): |g diff| <= 4 ulp(g),
    // |d diff| <= 4 ulp(t). t/fix/trip/RNG bitwise. Rationale: g carries
    // the device-libm spread (~1 ulp measured); d = t-z amplifies it by
    // t/delta, so only the absolute (ulp(t)-scale) bound is meaningful.
    auto ulpof = [](float x) -> double {
        float n = std::nextafter(x, 1.0e30F);
        return (double)n - (double)x;
    };
    for (int i = 0; i < N; ++i) {
        const Case cc = cases[i];
        carbon::UrbanV2Material mat{};
        mat.mass_mev = 12.0F * 931.49410242F;
        mat.projectile_z = 6;
        mat.projectile_a = 12;
        const auto conv = carbon::copper_urban_v2_true_to_geom(
            cc.t, cc.lam, cc.R, cc.E, mat);
        float dd = 0.0F;
        bool tr = false;
        const float t = carbon::copper_urban_v2_finalize_true(
            cc.gf > 0.0F ? cc.gf : conv.g_mm,
            cc.gp > 0.0F ? cc.gp : conv.g_mm, cc.tm > 0.0F ? cc.tm : cc.t,
            conv.delta_mm, cc.lam, cc.R < 1.0e29F ? cc.R : 124.0F, cc.p1,
            cc.p3, dd, tr);
        const unsigned r0 =
            carbon::rng::msc_random_u32(11ULL, 22ULL, (std::uint64_t)i, 70U);
        const unsigned r1 =
            carbon::rng::msc_random_u32(11ULL, 22ULL, (std::uint64_t)i, 71U);
        const double genv =
            std::fabs((double)dev[i].g - (double)conv.g_mm) /
            (4.0 * ulpof(conv.g_mm) + 1e-300);
        const double denv =
            std::fabs((double)dev[i].d - (double)dd) /
            (4.0 * ulpof(cc.t > 0.0F ? cc.t : conv.g_mm) + 1e-300);
        if (genv > genv_max) genv_max = genv;
        if (denv > denv_max) denv_max = denv;
        const bool ok = genv <= 1.0 && denv <= 1.0 && dev[i].t == t &&
            dev[i].fix == conv.rounding_fixup && dev[i].trip == tr &&
            dev[i].r0 == r0 && dev[i].r1 == r1;
        if (!ok) {
            ++mism;
            std::printf("MISMATCH case %d: dev g=%a d=%a t=%a fix=%d trip=%d "
                        "r0=%08x r1=%08x | host g=%a d=%a t=%a fix=%d trip=%d "
                        "r0=%08x r1=%08x\n",
                        i, (double)dev[i].g, (double)dev[i].d, (double)dev[i].t,
                        (int)dev[i].fix, (int)dev[i].trip, dev[i].r0, dev[i].r1,
                        (double)conv.g_mm, (double)dd, (double)t,
                        (int)conv.rounding_fixup, (int)tr, r0, r1);
        }
    }
    std::printf("max g envelope %.2f, max d envelope %.2f (%d cases)\n",
                genv_max, denv_max, N);
    std::printf(mism == 0 ? "GPU-HOST SPREAD WITHIN BOUNDS\n"
                          : "GPU-HOST MISMATCHES: %d\n",
                mism);
    return mism == 0 ? 0 : 1;
}

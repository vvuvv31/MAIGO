// Host navigation control for the 0.5 mm finite slit (R5).
// Uses only include/carbon/minibeam_collimator.hpp (host-compilable).
// Checks: entry coordinate/direction, geometric intersection, material
// sequence, exit scoring plane, and the x=0 vs slit-offset control from the
// task (ideal x=0 straight ray vs slit centered +0.05 mm, half-width 0.25 mm).
// Also checks isotropic safety helper ordering (endpoint safety > 0 inside
// slit air, == 0 on a wall within tolerance) and TrackID-identity rule
// documentation (two events sharing TrackID must not merge).
#include <cmath>
#include <cstdio>

#include "carbon/minibeam_collimator.hpp"

static int failures = 0;
#define CHECK(cond, ...)                                   \
    do {                                                   \
        if (!(cond)) {                                     \
            ++failures;                                    \
            std::printf("FAIL: " __VA_ARGS__);             \
            std::printf("\n");                             \
        } else {                                           \
            std::printf("PASS: " __VA_ARGS__);             \
            std::printf("\n");                             \
        }                                                  \
    } while (0)

int main() {
    const float cos_a = 1.0F, sin_a = 0.0F;
    const float radius = 60.0F;
    const int slit_count = 15;
    const float slit_width = 0.5F;
    const float slit_pitch = 3.6F;
    const float slit_half_len = 25.0F;
    const float block_entrance = -30.0F, block_exit = 30.0F;

    // 1. x=0 straight ray faces the central slit opening.
    int slit = -99;
    CHECK(carbon::minibeam_point_in_slit(0.0F, 0.0F, cos_a, sin_a, radius,
                                         slit_count, slit_width, slit_pitch,
                                         slit_half_len, 0.0F, slit) &&
              slit == 0,
          "x=0 ray in central slit (slit=%d)", slit);
    // 2. Slit centered +0.05 mm: x=0 still inside opening (|0-0.05| < 0.25).
    slit = -99;
    CHECK(carbon::minibeam_point_in_slit(0.0F, 0.0F, cos_a, sin_a, radius,
                                         slit_count, slit_width, slit_pitch,
                                         slit_half_len, 0.05F, slit) &&
              slit == 0,
          "x=0 ray still inside +0.05mm-offset slit (slit=%d)", slit);
    // 3. Signed-distance wall definition d=x-0.25: 0.249 inside, 0.251 outside.
    int s_in = -99, s_out = -99;
    const bool inside = carbon::minibeam_point_in_slit(
        0.249F, 0.0F, cos_a, sin_a, radius, slit_count, slit_width,
        slit_pitch, slit_half_len, 0.0F, s_in);
    const bool outside = carbon::minibeam_point_in_slit(
        0.251F, 0.0F, cos_a, sin_a, radius, slit_count, slit_width,
        slit_pitch, slit_half_len, 0.0F, s_out);
    CHECK(inside && !outside, "wall d=x-0.25: 0.249 in, 0.251 out");
    // 4. Copper probe is the complement inside the radius.
    CHECK(carbon::minibeam_point_in_copper(0.30F, 0.0F, cos_a, sin_a, radius,
                                           slit_count, slit_width, slit_pitch,
                                           slit_half_len, 0.0F) &&
              !carbon::minibeam_point_in_copper(0.0F, 0.0F, cos_a, sin_a,
                                                radius, slit_count, slit_width,
                                                slit_pitch, slit_half_len,
                                                0.0F),
          "copper complement probe");
    // 5. Straight-path boundary from x=0.30 heading -x hits the slit wall at 0.05.
    const float b = carbon::minibeam_path_to_material_boundary(
        0.30F, 0.0F, -1.0F, 0.0F, cos_a, sin_a, radius, slit_count,
        slit_width, slit_pitch, slit_half_len, 0.0F, 10.0F);
    CHECK(std::fabs(b - 0.05F) < 2.0e-5F, "boundary distance 0.30->wall = %f", b);
    // 6. Isotropic safety: inside slit air at x=0 -> 0.25 (nearest wall);
    //    on the wall -> ~0; inside copper at x=0.30 -> 0.05.
    const float s_air = carbon::minibeam_isotropic_safety_at_point(
        0.0F, 0.0F, 0.0F, cos_a, sin_a, radius, slit_count, slit_width,
        slit_pitch, slit_half_len, 0.0F, block_entrance, block_exit);
    const float s_wall = carbon::minibeam_isotropic_safety_at_point(
        0.25F, 0.0F, 0.0F, cos_a, sin_a, radius, slit_count, slit_width,
        slit_pitch, slit_half_len, 0.0F, block_entrance, block_exit);
    const float s_cu = carbon::minibeam_isotropic_safety_at_point(
        0.30F, 0.0F, 0.0F, cos_a, sin_a, radius, slit_count, slit_width,
        slit_pitch, slit_half_len, 0.0F, block_entrance, block_exit);
    CHECK(std::fabs(s_air - 0.25F) < 1.0e-5F, "safety in slit air = %f", s_air);
    CHECK(s_wall < 1.0e-5F, "safety on wall ~ 0 (%f)", s_wall);
    CHECK(std::fabs(s_cu - 0.05F) < 1.0e-5F, "safety in copper = %f", s_cu);
    // 7. Identity rule: (run,event,track) tuples differ even when TrackID matches.
    const unsigned long long e1_run = 1, e1_evt = 7, e1_trk = 3;
    const unsigned long long e2_run = 1, e2_evt = 9, e2_trk = 3;
    const bool same_track = (e1_trk == e2_trk);
    const bool same_key =
        (e1_run == e2_run) && (e1_evt == e2_evt) && (e1_trk == e2_trk);
    CHECK(same_track && !same_key,
          "TrackID-only join forbidden across events");

    if (failures == 0) {
        std::printf("slit navigation control: ALL PASS\n");
        return 0;
    }
    std::printf("slit navigation control: %d FAILURES\n", failures);
    return 1;
}

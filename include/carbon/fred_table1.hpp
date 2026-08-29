#pragma once

#include "carbon/detail/fred_fragmentation_data.hpp"

#include <array>
#include <cstdint>

namespace carbon {

struct Table1InvertResult {
    std::array<float, 18> sample_prob{};
    std::array<double, 18> inclusive_fraction{};
    std::array<double, 18> table_fraction{};
    float max_abs_fraction_error{};
    unsigned closed_events{};
    unsigned events{};
    int iterations{};
};

struct JointTable1InvertResult {
    std::array<float, 18> projectile_prob{};
    std::array<float, 18> target_prob{};
    std::array<double, 18> inclusive_fraction{};
    std::array<double, 18> table_fraction{};
    float max_abs_fraction_error{};
    unsigned accepted_events{};
    unsigned events{};
    int iterations{};
};

void simulate_nucleon_conserving_inclusive(const float* sample_prob,
                                           int a0,
                                           int z0,
                                           unsigned events,
                                           std::uint32_t seed,
                                           double counts_out[18],
                                           unsigned* closed_events,
                                           int max_frags,
                                           bool stop_on_heavy);

inline void simulate_nucleon_conserving_inclusive(const float* sample_prob,
                                                  int a0,
                                                  int z0,
                                                  unsigned events,
                                                  std::uint32_t seed,
                                                  double counts_out[18],
                                                  unsigned* closed_events) {
    const int max_frags = (a0 == 12 && z0 == 6) ? 8 : 12;
    const bool stop_on_heavy = (a0 == 16 && z0 == 8);
    simulate_nucleon_conserving_inclusive(sample_prob, a0, z0, events, seed, counts_out,
                                          closed_events, max_frags, stop_on_heavy);
}

void simulate_projectile_inclusive(const float* sample_prob,
                                   unsigned events,
                                   std::uint32_t seed,
                                   double counts_out[18],
                                   unsigned* closed_events);

Table1InvertResult invert_table1_independent_probs(const float* table_percent,
                                                   int a0,
                                                   int z0,
                                                   unsigned events,
                                                   unsigned iterations,
                                                   std::uint32_t seed);


void simulate_joint_table1_inclusive(const float* projectile_prob,
                                     const float* target_prob,
                                     int target_a, int target_z,
                                     unsigned events, std::uint32_t seed,
                                     double counts_out[18], unsigned* accepted_events);

JointTable1InvertResult invert_table1_joint_probs(const float* table_percent,
                                                  int target_a, int target_z,
                                                  unsigned events, unsigned iterations,
                                                  std::uint32_t seed);

inline Table1InvertResult invert_table1_independent_probs(const float* table_percent,
                                                          unsigned events,
                                                          unsigned iterations,
                                                          std::uint32_t seed) {
    return invert_table1_independent_probs(table_percent, 12, 6, events, iterations, seed);
}

}  // namespace carbon

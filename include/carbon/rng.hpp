#pragma once

#include <array>
#include <cstdint>

namespace carbon::rng {

using PhiloxBlock = std::array<std::uint32_t, 4>;

// Branch-tag roles for deterministic child RNG streams. Tags must not depend on
// GPU atomic queue indices — only on parent stream identity and product order.
inline constexpr std::uint32_t branch_role_primary_charged = 0x10000000U;
inline constexpr std::uint32_t branch_role_primary_neutral = 0x20000000U;
inline constexpr std::uint32_t branch_role_cascade_charged = 0x30000000U;
inline constexpr std::uint32_t branch_role_cascade_neutral = 0x40000000U;
inline constexpr std::uint32_t branch_role_neutral_charged = 0x50000000U;
inline constexpr std::uint32_t branch_role_neutral_continuation = 0x60000000U;
inline constexpr std::uint32_t branch_role_neutral_electron = 0x70000000U;

inline constexpr std::uint32_t branch_tag(const std::uint32_t role,
                                          const std::uint32_t product_index) noexcept {
    return role | (product_index & 0x0FFFFFFFU);
}

// SplitMix-style mix so (parent_stream, branch_tag) maps injectively enough for MC.
// Device-friendly: no library hash, pure arithmetic.
inline std::uint64_t child_stream(const std::uint64_t parent_stream,
                                  const std::uint32_t tag) noexcept {
    std::uint64_t z =
        parent_stream + 0x9E3779B97F4A7C15ULL +
        (static_cast<std::uint64_t>(tag) + 1ULL) * 0xD1B54A32D192ED03ULL;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

// Event identity and product ordinal are separate levels: siblings must never
// share subsequent transport draws. Atomic queue positions are not identities.
inline std::uint64_t event_product_stream(std::uint64_t parent,
                                          std::uint32_t collision_index,
                                          std::uint32_t role,
                                          std::uint32_t product_index) noexcept {
    return child_stream(child_stream(parent, collision_index),
                        branch_tag(role, product_index));
}

inline PhiloxBlock philox4x32_10(PhiloxBlock counter,
                                std::uint32_t key0,
                                std::uint32_t key1) noexcept {
    constexpr std::uint32_t multiplier0 = 0xD2511F53U;
    constexpr std::uint32_t multiplier1 = 0xCD9E8D57U;
    constexpr std::uint32_t key_increment0 = 0x9E3779B9U;
    constexpr std::uint32_t key_increment1 = 0xBB67AE85U;

    for (int round = 0; round < 10; ++round) {
        const auto product0 = static_cast<std::uint64_t>(multiplier0) * counter[0];
        const auto product1 = static_cast<std::uint64_t>(multiplier1) * counter[2];
        const auto low0 = static_cast<std::uint32_t>(product0);
        const auto high0 = static_cast<std::uint32_t>(product0 >> 32U);
        const auto low1 = static_cast<std::uint32_t>(product1);
        const auto high1 = static_cast<std::uint32_t>(product1 >> 32U);
        counter = {high1 ^ counter[1] ^ key0, low1,
                   high0 ^ counter[3] ^ key1, low0};
        key0 += key_increment0;
        key1 += key_increment1;
    }
    return counter;
}

inline std::uint32_t random_u32(std::uint64_t seed,
                                 std::uint64_t history_id,
                                 std::uint64_t interaction_index,
                                 std::uint32_t random_dimension) noexcept {
    const auto block = random_dimension / 4U;
    PhiloxBlock counter{
        static_cast<std::uint32_t>(history_id),
        static_cast<std::uint32_t>(history_id >> 32U),
        static_cast<std::uint32_t>(interaction_index),
        static_cast<std::uint32_t>(interaction_index >> 32U) ^ block,
    };
    const auto generated = philox4x32_10(
        counter, static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32U));
    return generated[random_dimension % 4U];
}

inline float uniform01(std::uint64_t seed,
                       std::uint64_t history_id,
                       std::uint64_t interaction_index,
                       std::uint32_t random_dimension) noexcept {
    // Use the leading 24 bits so every returned float is strictly inside (0, 1).
    constexpr float inverse_two_to_24 = 5.9604644775390625e-8f;
    const auto bits = random_u32(seed, history_id, interaction_index, random_dimension) >> 8U;
    const float value = (static_cast<float>(bits) + 0.5f) * inverse_two_to_24;
    // The largest 24-bit midpoint rounds to 1 in FP32. Keep the open
    // interval promised above, so -log(u) cannot create a zero flight.
    // All other draws retain their existing value and counter identity.
    return value < 1.0f ? value : 0x1.fffffep-1f;
}

// MSC-dedicated domain draw (fix C2, versioned, research path).
// Problem: Urban draws used (seed, history, steps*1024+segment, 70/71/...),
// while all-ion elastic uses (seed, history, steps, 70/71) on the SAME
// (seed, history) key. At outer_step = 0, segment = 0 the raw Philox
// addresses coincide exactly (same counter words, same dim lane) — the same
// "random" bits would steer two different physics decisions in full physics
// (EM-only never runs elastic, which is why this never showed there).
// Fix: MSC draws set bit 30 of Philox counter word 3. Proof of separation:
//   - legacy counter[3] = (ii>>32) ^ (dim/4) with ii < 2^32 in every legacy
//     consumer (outer step counts, fixed 0/indices) and dim any u32, so
//     counter[3] <= 0x3FFFFFFF: bit 30 is ALWAYS clear;
//   - MSC counter[3] = (ii>>32) ^ (dim/4) ^ 0x40000000 with ii < 2^32
//     (enforced: outer < 2^22, segment < 1024): bit 30 is ALWAYS set.
// The two counter SETS are disjoint; Philox with a fixed key is a bijection
// over counters, so no MSC draw can ever equal a legacy draw. This is a
// structural guarantee about the address sets, not a "vanishing collision
// probability" claim, and it changes no legacy stream bit.
// Callers must guarantee interaction_index < 2^32 (guarded at the Urban
// call sites: outer_step < 2^22, segment_index < 1024); larger values would
// break the proof and must fail before any draw.
inline constexpr std::uint32_t msc_domain_tag = 0x40000000U;

inline std::uint32_t msc_random_u32(std::uint64_t seed,
                                    std::uint64_t history_id,
                                    std::uint64_t interaction_index,
                                    std::uint32_t random_dimension) noexcept {
    const auto block = random_dimension / 4U;
    PhiloxBlock counter{
        static_cast<std::uint32_t>(history_id),
        static_cast<std::uint32_t>(history_id >> 32U),
        static_cast<std::uint32_t>(interaction_index),
        static_cast<std::uint32_t>(interaction_index >> 32U) ^ block ^
            msc_domain_tag,
    };
    const auto generated = philox4x32_10(
        counter, static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32U));
    return generated[random_dimension % 4U];
}

// Strict-unit MSC draw: same endpoint clamp as urban_unit_strict, on the
// MSC domain. All Urban consumers (sampler dims, limiter 58/59) must use
// this; urban_unit_strict is kept for the regression tests only.
inline float msc_unit_strict(std::uint64_t seed,
                             std::uint64_t history_id,
                             std::uint64_t interaction_index,
                             std::uint32_t random_dimension) noexcept {
    constexpr float inverse_two_to_24 = 5.9604644775390625e-8f;
    constexpr float one_minus_ulp = 0x1.fffffep-1f;
    const auto bits =
        msc_random_u32(seed, history_id, interaction_index, random_dimension) >>
        8U;
    const float v = (static_cast<float>(bits) + 0.5f) * inverse_two_to_24;
    return v < 1.0f ? v : one_minus_ulp;
}
// Urban-MSC-only strict unit draw (versioned, research path).
// uniform01's mapping rounds bits=2^24-1 to exactly 1.0f in FP32, which
// corrupts the Urban Bernoulli trial `u0 < q_probability` (a 2^-24-rate
// spurious isotropic branch even when q >= 1, i.e. when the reference takes
// the mixture branch with probability 1). This variant clamps only that
// endpoint to the largest float below 1 (0x1.fffffep-1); every other draw is
// bit-identical to uniform01, and all non-Urban consumers keep uniform01, so
// no production baseline stream changes.
inline float urban_unit_strict(std::uint64_t seed,
                               std::uint64_t history_id,
                               std::uint64_t interaction_index,
                               std::uint32_t random_dimension) noexcept {
    constexpr float inverse_two_to_24 = 5.9604644775390625e-8f;
    constexpr float one_minus_ulp = 0x1.fffffep-1f;
    const auto bits = random_u32(seed, history_id, interaction_index, random_dimension) >> 8U;
    const float v = (static_cast<float>(bits) + 0.5f) * inverse_two_to_24;
    return v < 1.0f ? v : one_minus_ulp;
}

}  // namespace carbon::rng

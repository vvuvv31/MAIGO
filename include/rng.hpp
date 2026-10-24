#pragma once

#include <array>
#include <cstdint>

namespace carbon::rng {

using PhiloxBlock = std::array<std::uint32_t, 4>;

// Branch-tag roles for deterministic child RNG streams. Tags must not depend on
// GPU atomic queue indices — only on parent stream identity and product order.
inline constexpr std::uint32_t branch_role_primary_charged = 0x10000000U;
inline constexpr std::uint32_t branch_role_beamline_charged = 0x20000000U;
inline constexpr std::uint32_t branch_role_cascade_charged = 0x30000000U;

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

// Descendant workers use a shared Philox key, so their parent identity must
// include the primary's full spot seed as well as its spot-local history.
// Both maps are bijective for one fixed input; mixing a pair into 64 bits has
// the same finite hash-collision limitation as the existing child streams.
// Do not change the primary (seed, history) draws or use atomic queue indices.
inline std::uint64_t primary_descendant_stream(const std::uint64_t seed,
                                               const std::uint64_t history) noexcept {
    return child_stream(history ^ child_stream(seed, 0x53454544U), 0x524F4F54U);
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

// Copper loss draws have a per-step draw counter. Its upper 32 bits
// must not alias source, medium-loss or nuclear dimensions in random_u32.
// A separate Philox key keeps this sequence independent of those decisions;
// the existing air-transport key and all other streams stay unchanged.
inline std::uint32_t copper_loss_u32(std::uint64_t seed,
                                     std::uint64_t history_id,
                                     std::uint64_t step,
                                     std::uint64_t draw) noexcept {
    constexpr std::uint64_t domain_key = 0xC05F9D7A6B3E2149ULL;
    return random_u32(seed ^ domain_key, history_id, (step << 32U) | draw, 124U);
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

// Strict-unit draw on the MSC domain. All Urban consumers (sampler dims
// and limiter 58/59) use this endpoint clamp.
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

// Step-keyed fluctuation stream. One Philox block per transport step yields
// a 64-bit key; draw j of that step is SplitMix64 output j+1 from state key.
// Every value is fixed by (key seed, history, step, j): no cached state, no
// lane-dependent refresh. Counter word 3 carries bit 28; the legacy copper and
// air loss counters use word 3 = step ^ 31 with step < 2^17, so the sets are
// disjoint under each existing Philox key.
inline constexpr std::uint32_t step_key_domain_tag = 0x10000000U;

inline std::uint64_t fluctuation_step_key(std::uint64_t key_seed,
                                          std::uint64_t history_id,
                                          std::uint64_t step) noexcept {
    const PhiloxBlock counter{
        static_cast<std::uint32_t>(history_id),
        static_cast<std::uint32_t>(history_id >> 32U),
        static_cast<std::uint32_t>(step),
        (static_cast<std::uint32_t>(step >> 32U) & 0x0FFFFFFFU) ^ step_key_domain_tag,
    };
    const auto block = philox4x32_10(counter, static_cast<std::uint32_t>(key_seed),
                                     static_cast<std::uint32_t>(key_seed >> 32U));
    return static_cast<std::uint64_t>(block[0]) | (static_cast<std::uint64_t>(block[1]) << 32U);
}

// Leading 24 bits of SplitMix64 output (index+1) from state key, as in uniform01.
inline float splitmix_unit24(std::uint64_t key, std::uint64_t index) noexcept {
    std::uint64_t z = key + (index + 1ULL) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    z ^= z >> 31U;
    return static_cast<float>(static_cast<std::uint32_t>(z >> 40U)) * 0x1p-24f;
}

}  // namespace carbon::rng

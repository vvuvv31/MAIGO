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
    return (static_cast<float>(bits) + 0.5f) * inverse_two_to_24;
}

}  // namespace carbon::rng

#pragma once

#include <array>
#include <cstdint>

namespace carbon::rng {

using PhiloxBlock = std::array<std::uint32_t, 4>;

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

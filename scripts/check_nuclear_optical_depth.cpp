// Standalone check: c++ -std=c++20 -O2 -Iinclude scripts/check_nuclear_optical_depth.cpp -o /tmp/check_nuclear_optical_depth
#include "carbon/cross_section.hpp"
#include <cassert>
#include <iostream>
#include <random>

int main() {
    using carbon::consume_schneider_optical_depth_segment;
    // A surviving clock must traverse zero-rate regions and changed materials
    // without resampling. Copying it models a scheduler suspension/resumption.
    float remaining = 0.7F;
    bool active = true;
    float step = 0.5F;
    assert(!consume_schneider_optical_depth_segment(remaining, active, step, 0.4F, 0.01F));
    assert(std::abs(remaining - 0.5F) < 1.e-6F);
    step = 100.F;
    assert(!consume_schneider_optical_depth_segment(remaining, active, step, 0.F, 0.01F));
    auto saved_remaining = remaining;
    auto saved_active = active;
    step = 2.F;
    assert(consume_schneider_optical_depth_segment(saved_remaining, saved_active, step, 0.5F, 0.01F));
    assert(std::abs(step - 1.F) < 1.e-6F && !saved_active);

    // Compare sampled survival and competing-process frequencies with their
    // analytic probabilities, for coarse and finely subdivided paths.
    constexpr unsigned samples = 400000;
    for (unsigned subdivisions : {1U, 20U}) {
        std::mt19937 engine(9271);
        std::uniform_real_distribution<float> uniform(0.F, 1.F);
        unsigned hits = 0, elastic = 0;
        for (unsigned i = 0; i < samples; ++i) {
            float tau = 0.F;
            bool armed = false;
            const float u = uniform(engine);
            for (unsigned j = 0; j < subdivisions; ++j) {
                float length = 2.F / subdivisions;
                if (consume_schneider_optical_depth_segment(tau, armed, length, 0.5F, u)) {
                    ++hits;
                    if (uniform(engine) < 0.3F / 0.5F) ++elastic;
                    break;
                }
            }
        }
        const double p = 1. - std::exp(-1.);
        assert(std::abs(double(hits) / samples - p) < 6. * std::sqrt(p * (1. - p) / samples));
        assert(std::abs(double(elastic) / hits - 0.6) < 6. * std::sqrt(0.24 / hits));
    }
    // Generation-limited tracks consume the elastic-only rate normally.
    remaining = 0.3F; active = true; step = 2.F;
    assert(consume_schneider_optical_depth_segment(remaining, active, step, 0.3F, 1.F));
    assert(std::abs(step - 1.F) < 1.e-6F);
    std::cout << "Passed nuclear survival, process competition, boundary/continuation and elastic-only checks\n";
}

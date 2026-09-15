#include "carbon/nuclear_collision.hpp"
#include "carbon/charged_species.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    try {
        using carbon::sample_exponential_collision;
        const auto zero = sample_exponential_collision(0.f, 1.f, .1f);
        check(!zero.occurred && zero.distance_mm == 1.f, "zero-rate segment");
        const auto median = sample_exponential_collision(2.f, 1.f, .5f);
        check(median.occurred && std::abs(median.distance_mm - std::log(2.f)/2.f) < 1.e-7f,
              "exponential median");
        const auto escaped = sample_exponential_collision(2.f, 1.f, .99f);
        check(!escaped.occurred && escaped.distance_mm == 1.f, "finite horizon censoring");
        const auto short_step = sample_exponential_collision(1.e6f, 1.e-6f, 0.f);
        check(short_step.occurred && short_step.distance_mm <= 1.e-6f, "short-step bound");
        // Uniform quantile grid: compare the sampled distribution to the analytic CDF.
        constexpr int samples = 100000;
        for (float cutoff : {.1f, .3f, .8f}) {
            int count = 0;
            for (int i = 0; i < samples; ++i) {
                const auto draw = sample_exponential_collision(2.f, 1.f, (i+.5f)/samples);
                count += draw.occurred && draw.distance_mm <= cutoff;
            }
            const double expected = samples * (1. - std::exp(-2. * cutoff));
            check(std::abs(count-expected) < 2., "exponential distribution CDF");
        }
        for (int i=0;i<18;++i) {
            auto ion=carbon::kChargedIons[i];
            check(carbon::get_charged_species_idx(ion.z,ion.a)==i,"package species order");
        }
        check(carbon::get_charged_species_idx(0,1)==-1,"neutron is not charged");
        check(carbon::get_charged_species_idx(4,6)==17,"Be6 has its own table");
        check(carbon::get_charged_species_idx(7,14)==-1,"unknown species must not alias");
        std::cout << "collision distribution and charged species passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

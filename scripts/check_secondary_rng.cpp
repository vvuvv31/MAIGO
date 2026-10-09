// Standalone RNG address regression; no CPU transport implementation.
// g++ -std=c++20 -O2 -I include scripts/check_secondary_rng.cpp -o /tmp/check_secondary_rng
#include "carbon/rng.hpp"
#include <iostream>
#include <stdexcept>
#include <unordered_set>

int main() {
    using namespace carbon::rng;
    std::unordered_set<std::uint64_t> identities;
    unsigned checks=0;
    // Actual campaign repeats spot-local histories across 121 spots and batches.
    // Include high seed bits: truncating the seed to 32 bits is not sufficient.
    for (const std::uint64_t seed : {202610085000000ULL,202610085000001ULL,
                                    202610085001000ULL,202610185000000ULL,
                                    202610085000000ULL+(1ULL<<32)}) {
        for (std::uint64_t history=0;history<1000;++history) {
            const auto parent=primary_descendant_stream(seed,history);
            if (parent!=primary_descendant_stream(seed,history))
                throw std::runtime_error("parent identity is not reproducible");
            const auto elastic=child_stream(parent,branch_tag(branch_role_primary_charged,42));
            const auto medium=event_product_stream(parent,42,branch_role_primary_charged,0);
            const auto sibling=event_product_stream(parent,42,branch_role_primary_charged,1);
            const auto copper=event_product_stream(parent,42,branch_role_beamline_charged,0);
            const auto cascade=event_product_stream(medium,42,branch_role_cascade_charged,0);
            for (auto stream : {elastic,medium,sibling,copper,cascade}) {
                if (!identities.insert(stream).second)
                    throw std::runtime_error("descendant identity reused across seed/history/branch");
                ++checks;
            }
        }
    }
    // Reproduce the former alias in the actual shared-key worker address.
    const auto old_a=event_product_stream(17,42,branch_role_primary_charged,0);
    const auto old_b=event_product_stream(17,42,branch_role_primary_charged,0);
    if (random_u32(2026,old_a,0,121)!=random_u32(2026,old_b,0,121))
        throw std::runtime_error("old alias fixture is invalid");
    const auto a=event_product_stream(primary_descendant_stream(202610085000000ULL,17),42,branch_role_primary_charged,0);
    const auto b=event_product_stream(primary_descendant_stream(202610085000001ULL,17),42,branch_role_primary_charged,0);
    bool different=false;
    for (unsigned dimension=0;dimension<128;++dimension)
        different |= random_u32(2026,a,0,dimension)!=random_u32(2026,b,0,dimension);
    if (!different)throw std::runtime_error("seeded workers still repeat the Philox block");
    std::cout<<"Passed "<<checks<<" descendant identities and old shared-key alias regression\n";
}

#pragma once
#include <cstddef>

namespace carbon {
// Small immutable metadata, separate from the much larger response views.
// Sections are sorted; entries within a section retain original table order,
// including equal-density ties. Table IDs always refer to the original bank.
struct MaterialElectronLookupSection {
    int section{};
    std::size_t begin{},end{};
};
struct MaterialElectronLookupEntry {
    double density{};
    std::size_t table{};
};
struct MaterialElectronLookupView {
    const MaterialElectronLookupSection* sections{};
    const MaterialElectronLookupEntry* entries{};
    std::size_t section_count{};
    std::size_t table_count{};

    const MaterialElectronLookupSection* find(int section) const {
        std::size_t lo=0,hi=section_count;
        while(lo<hi) {
            const auto mid=lo+(hi-lo)/2;
            if(sections[mid].section<section)lo=mid+1;
            else hi=mid;
        }
        return lo<section_count && sections[lo].section==section?sections+lo:nullptr;
    }
};
} // namespace carbon

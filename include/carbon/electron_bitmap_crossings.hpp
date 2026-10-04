#pragma once
#include <bit>
#include <cstdint>
#include <cstddef>
#include <limits>
namespace carbon {
// Sparse bins use row IDs; dense bins use exact membership bits and rank
// checkpoints. Descriptors can be shared only for identical ordered bins.
struct ElectronBitmapBin {
    std::uint64_t offset{}; // sparse uint32 row offset or dense uint64 word offset
    std::uint32_t rank_offset{};
    std::uint32_t blocks{}; // zero means sparse; dense blocks contain 512 rows
};
static_assert(sizeof(ElectronBitmapBin)==16);
struct ElectronBitmapCrossingsView {
    const ElectronBitmapBin* bins{};
    const std::uint32_t* sparse{};
    const std::uint64_t* words{};
    const std::uint32_t* ranks{};
    std::uint64_t select(std::size_t bin,std::uint64_t index) const {
        const auto d=bins[bin];
        if(!d.blocks)return sparse[d.offset+index];
        // upper_bound(rank,index)-1: equal ranks from empty blocks must be skipped.
        std::uint32_t lo=0,hi=d.blocks;
        while(lo+1<hi) {auto mid=lo+(hi-lo)/2;if(ranks[d.rank_offset+mid]<=index)lo=mid;else hi=mid;}
        auto remaining=static_cast<std::uint32_t>(index-ranks[d.rank_offset+lo]);
        for(unsigned w=0;w<8;++w) {
            auto bits=words[d.offset+std::uint64_t(lo)*8+w];const auto count=std::popcount(bits);
            if(remaining>=static_cast<unsigned>(count)){remaining-=count;continue;}
            // Select within one word in six fixed subdivisions, no rank-sized loop.
            unsigned bit=0;
            for(unsigned half=32;half;half/=2) {
                const auto low=bits&((std::uint64_t{1}<<half)-1);const auto below=std::popcount(low);
                if(remaining>=static_cast<unsigned>(below)){remaining-=below;bits>>=half;bit+=half;}else bits=low;
            }
            return std::uint64_t(lo)*512+w*64+bit;
        }
        return std::numeric_limits<std::uint64_t>::max();
    }
};
}

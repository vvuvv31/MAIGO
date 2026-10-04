#pragma once
#include "carbon/electron_bitmap_crossings.hpp"
#include <vector>
#include <map>
#include <stdexcept>
#include <algorithm>
namespace carbon {
struct ElectronBitmapBuilder {
    std::vector<ElectronBitmapBin> bins;
    std::vector<std::uint32_t> sparse,ranks;
    std::vector<std::uint64_t> words;
    ElectronBitmapCrossingsView view()const{return {bins.data(),sparse.data(),words.data(),ranks.data()};}
    static ElectronBitmapBuilder build(std::uint64_t rows,const std::vector<std::uint64_t>& offsets,
        const std::vector<std::uint64_t>& starts,const std::vector<std::uint32_t>& candidates) {
        if(!rows || rows>=UINT32_MAX || offsets.size()!=starts.size()+1 || offsets.front()!=0)
            throw std::runtime_error("Bitmap crossing dimensions invalid");
        ElectronBitmapBuilder out;std::map<std::pair<std::uint64_t,std::uint64_t>,ElectronBitmapBin> shared;
        const auto blocks=(rows+511)/512,dense_bytes=blocks*64+(blocks+1)*4;
        for(std::size_t b=0;b<starts.size();++b) {
            if(offsets[b+1]<offsets[b])throw std::runtime_error("Nonmonotone crossing offsets");
            auto count=offsets[b+1]-offsets[b],start=starts[b];
            if(start>candidates.size() || count>candidates.size()-start || count>rows)
                throw std::runtime_error("Bitmap candidate range invalid");
            const auto key=std::make_pair(start,count);
            if(auto it=shared.find(key);it!=shared.end()){out.bins.push_back(it->second);continue;}
            for(std::uint64_t i=0;i<count;++i)if(candidates[start+i]>=rows || (i && candidates[start+i]<=candidates[start+i-1]))
                throw std::runtime_error("Bitmap requires strictly ordered unique rows");
            ElectronBitmapBin d;
            if(count*4<=dense_bytes) {
                d.offset=out.sparse.size();out.sparse.insert(out.sparse.end(),candidates.begin()+start,candidates.begin()+start+count);
            } else {
                if(out.ranks.size()+blocks+1>UINT32_MAX)throw std::runtime_error("Bitmap rank offset overflow");
                d.offset=out.words.size();d.rank_offset=out.ranks.size();d.blocks=blocks;
                out.words.resize(out.words.size()+blocks*8,0);
                for(std::uint64_t i=0;i<count;++i){auto row=candidates[start+i];out.words[d.offset+row/64]|=std::uint64_t{1}<<(row%64);}
                std::uint32_t rank=0;
                for(std::uint64_t block=0;block<blocks;++block){out.ranks.push_back(rank);for(unsigned w=0;w<8;++w)rank+=std::popcount(out.words[d.offset+block*8+w]);}
                out.ranks.push_back(rank);if(rank!=count)throw std::runtime_error("Bitmap rank mismatch");
            }
            shared.emplace(key,d);out.bins.push_back(d);
        }
        return out;
    }
};
inline std::vector<std::uint32_t> electron_graph32(const std::vector<std::uint64_t>& source,bool sentinel=false) {
    std::vector<std::uint32_t> result;result.reserve(source.size());
    for(auto v:source) {
        if(sentinel && v==UINT64_MAX)result.push_back(UINT32_MAX);
        else {if(v>=UINT32_MAX)throw std::runtime_error("Electron graph exceeds uint32 domain");result.push_back(static_cast<std::uint32_t>(v));}
    }
    return result;
}
}

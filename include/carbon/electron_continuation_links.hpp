#pragma once
#include "carbon/electron_continuation_replay.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace carbon {
// Host-only construction from the already SHA-verified raw source. This avoids
// a second independently versioned genealogy file. Source row numbers remain
// unchanged, so sample/node references and replay share exactly one identity.
struct ElectronContinuationLinks {
    std::vector<std::uint64_t> next,child_offsets,child_rows;
    ElectronContinuationView view(const unsigned char* raw,std::size_t bytes) const {
        return {raw,bytes,next.data(),child_offsets.data(),child_rows.data(),child_rows.size()};
    }
    static ElectronContinuationLinks build(const unsigned char* raw,std::size_t bytes,
                                          std::size_t byte_budget) {
        if(!raw || !bytes || bytes%kElectronStateV4RecordBytes)
            throw std::invalid_argument("Invalid continuation source layout");
        const auto n=bytes/kElectronStateV4RecordBytes;
        if(byte_budget<8 || n>(byte_budget-8)/16)
            throw std::runtime_error("Continuation links exceed device byte budget");
        using Key=std::array<std::int32_t,4>;
        struct Entry {Key key;std::uint64_t row;};
        struct Child {Key parent,key;std::uint64_t row;};
        std::vector<Entry> ordered;ordered.reserve(n);
        std::vector<Child> children;
        for(std::uint64_t row=0;row<n;++row) {
            const auto state=read_electron_continuation_step(raw,bytes,row);
            if(!state.valid || (state.pdg!=11 && state.pdg!=22 && state.pdg!=1000060120))
                throw std::invalid_argument("Invalid/unsupported continuation source state");
            ordered.push_back({{state.run,state.event,state.track,state.step},row});
            if(state.step==1 && state.parent!=0)
                children.push_back({{state.run,state.event,state.parent,state.parent_step},
                    {state.run,state.event,state.track,state.step},row});
        }
        if(children.size()>(byte_budget-8-16*n)/8)
            throw std::runtime_error("Continuation children exceed device byte budget");
        std::sort(ordered.begin(),ordered.end(),[](const Entry& a,const Entry& b){return a.key<b.key;});
        // Canonical sibling order matches exact track-head ordering, independent
        // of TOPAS thread interleaving in the raw stream.
        std::sort(children.begin(),children.end(),[](const Child& a,const Child& b){return a.key<b.key;});
        ElectronContinuationLinks out;
        out.next.assign(n,kElectronContinuationEnd);out.child_offsets.assign(n+1,0);
        for(std::size_t i=0;i<n;++i) {
            const auto& item=ordered[i];
            const bool same=i && item.key[0]==ordered[i-1].key[0] &&
                item.key[1]==ordered[i-1].key[1] && item.key[2]==ordered[i-1].key[2];
            if(!same) {
                if(item.key[3]!=1)throw std::invalid_argument("Missing first continuation track step");
            } else {
                if(std::int64_t(item.key[3])!=std::int64_t(ordered[i-1].key[3])+1)
                    throw std::invalid_argument("Missing/duplicate continuation track step");
                out.next[ordered[i-1].row]=item.row;
            }
        }
        std::vector<std::uint64_t> parents;parents.reserve(children.size());
        for(const auto& child:children) {
            const auto parent=std::lower_bound(ordered.begin(),ordered.end(),child.parent,
                [](const Entry& a,const Key& b){return a.key<b;});
            if(parent==ordered.end() || parent->key!=child.parent ||
               !generated_by_electron_step(read_electron_continuation_step(raw,bytes,parent->row),
                                          read_electron_continuation_step(raw,bytes,child.row)))
                throw std::invalid_argument("Missing/invalid exact generating parent step");
            parents.push_back(parent->row);++out.child_offsets[parent->row+1];
        }
        for(std::size_t i=1;i<=n;++i)out.child_offsets[i]+=out.child_offsets[i-1];
        auto cursor=out.child_offsets;
        out.child_rows.resize(children.size());
        for(std::size_t i=0;i<children.size();++i)out.child_rows[cursor[parents[i]]++]=children[i].row;
        const auto view=out.view(raw,bytes);
        for(std::uint64_t row=0;row<n;++row) {
            const auto state=read_electron_continuation_step(raw,bytes,row);
            if((state.pdg==11 || state.pdg==22) &&
               prepare_electron_replay_step(view,row).status!=ElectronReplayStatus::ready)
                throw std::invalid_argument("Continuation source fails step energy/genealogy closure at row "+std::to_string(row));
        }
        return out;
    }
};
} // namespace carbon

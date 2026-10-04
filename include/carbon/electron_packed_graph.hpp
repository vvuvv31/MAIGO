#pragma once
#include "carbon/electron_continuation_links.hpp"
#include <vector>
#include <stdexcept>

namespace carbon {
// Experimental host transform after original SHA/primary-birth validation.
// Retain every electron/photon record, preserving fields and sibling order.
// The caller MUST remap birth and crossing references before publishing it.
// This is not a standalone loader or a promoted runtime format.
struct PackedElectronGraph {
    std::vector<ElectronContinuationStep> decoded;
    std::vector<std::uint64_t> next, offsets, children, original_rows, packed_rows;
    explicit PackedElectronGraph(const ElectronContinuationView& original) {
        const auto n=original.bytes/kElectronStateV4RecordBytes;
        packed_rows.assign(n,kElectronContinuationEnd);
        for(std::uint64_t row=0;row<n;++row) {
            const auto state=original.read(row);
            if(!state.valid)throw std::runtime_error("Invalid source state before packing");
            if(state.pdg!=11 && state.pdg!=22) {
                if(state.parent!=0)throw std::runtime_error("Unsupported non-EM descendant");
                continue;
            }
            const auto replay=prepare_electron_replay_step(original,row);
            if(replay.status!=ElectronReplayStatus::ready)
                throw std::runtime_error("Unvalidated EM row before packing");
            packed_rows[row]=decoded.size(); original_rows.push_back(row);
            decoded.push_back(state);
        }
        offsets.push_back(0);
        for(const auto row:original_rows) {
            const auto old_next=original.next_rows[row];
            next.push_back(old_next==kElectronContinuationEnd?old_next:map(old_next));
            for(auto i=original.child_offsets[row];i<original.child_offsets[row+1];++i)
                children.push_back(map(original.child_rows[i]));
            offsets.push_back(children.size());
        }
    }
    std::uint64_t map(std::uint64_t row) const {
        if(row>=packed_rows.size() || packed_rows[row]==kElectronContinuationEnd)
            throw std::runtime_error("Packed reference targets missing/non-EM row");
        return packed_rows[row];
    }
    ElectronContinuationView view() const {
        return {nullptr,decoded.size()*kElectronStateV4RecordBytes,next.data(),offsets.data(),
                children.data(),children.size(),decoded.data()};
    }
    std::size_t runtime_bytes() const {
        return decoded.size()*sizeof(ElectronContinuationStep)+
               (next.size()+offsets.size()+children.size())*sizeof(std::uint64_t);
    }
};
} // namespace carbon

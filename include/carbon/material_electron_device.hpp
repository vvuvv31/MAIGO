#pragma once
#include "carbon/material_electron_response.hpp"
#include "carbon/electron_continuation_links.hpp"
#include "carbon/electron_continuation_crossings.hpp"
#include "carbon/electron_birth_index.hpp"
#include "carbon/electron_short_range.hpp"
#include <sycl/sycl.hpp>
#include <limits>
#include <stdexcept>
#include <iostream>

namespace carbon {
enum class MaterialElectronMemory { device, host_mapped };
// Host-mapped storage is an explicit opt-in, never an implicit fallback. The
// GPU executes identical kernels on identical bytes through mapped host USM.
class MaterialElectronDeviceBank {
public:
    explicit MaterialElectronDeviceBank(sycl::queue queue,
        MaterialElectronMemory memory=MaterialElectronMemory::device,
        std::size_t hot_device_budget=0,bool device_indices=false):queue_(std::move(queue)),memory_(memory),hot_budget_(hot_device_budget),device_indices_(device_indices) {
        if(device_indices_ && memory_==MaterialElectronMemory::host_mapped && !hot_budget_)
            throw std::invalid_argument("Device indices require an explicit device budget");
        if(memory_==MaterialElectronMemory::host_mapped &&
           !queue_.get_device().has(sycl::aspect::usm_host_allocations))
            throw std::runtime_error("Device does not support explicit host-mapped response storage");
    }
    MaterialElectronDeviceBank(const MaterialElectronDeviceBank&)=delete;
    MaterialElectronDeviceBank& operator=(const MaterialElectronDeviceBank&)=delete;
    ~MaterialElectronDeviceBank() {
        // Constructor/load copies are synchronous; transport callers must
        // complete their kernel before releasing this bank.
        for(auto* p:allocations_)sycl::free(p,queue_);
    }
    void load(const MaterialElectronResponseIndex& index,
              const std::vector<std::pair<int,double>>& geometry,std::size_t byte_budget,
              bool packet_only=false,bool short_range_cache=false) {
        if(!allocations_.empty() || views_)throw std::logic_error("Response bank already loaded");
        const auto selected=index.required_tables(geometry);
        if(selected.empty())throw std::invalid_argument("Empty response geometry demand");
        budget_=byte_budget;
        std::vector<MaterialElectronResponseView> host;
        host.reserve(selected.size());
        for(auto id:selected) {
            const auto table=index.load_table(id);
            MaterialElectronResponseView view;
            view.section=table.material_section;
            view.density_g_cm3=index.files[id].density_g_cm3;
            view.reference_density_g_cm3=table.reference_density_g_cm3;
            view.channel_count=table.channels.size();view.node_count=table.nodes.size();
            view.channels=copy(table.channels,true);
            // Packet kernels consume raw steps/births, never the old
            // deposition-weighted endpoint/path bank. Still verify that bank
            // with load_table; only omit unused resident copies.
            if(!packet_only) {
                view.samples=copy(table.samples);
                view.heads=copy(table.heads);
                view.nodes=copy(table.nodes);
                view.prefix_radius=copy(table.prefix_radius);
            }
            if(!index.files[id].state_reference_file.empty()) {
                const auto payload=index.load_state_references(id,budget_-bytes_);
                auto refs=ElectronStateReferenceView::parse(payload.data(),payload.size());
                if(!refs.valid || refs.samples!=table.samples.size() || refs.nodes!=table.nodes.size())
                    throw std::invalid_argument("State references do not match resident response table");
                if(!packet_only) {refs.data=copy(payload);view.state_references=refs;}
                const auto sources=index.load_state_sources(id,budget_-bytes_);
                if(sources.size()!=refs.sources)throw std::invalid_argument("State-source count mismatch");
                std::vector<MaterialElectronStateSource> source_views;
                ElectronBirthIndex births(table.channels);
                for(std::size_t s=0;s<sources.size();++s) {
                    const auto expected=ElectronStateReferenceView::integer(payload.data()+32+8*s,8);
                    if(sources[s].size()/kElectronStateV4RecordBytes!=expected)
                        throw std::invalid_argument("State-source row count differs from references");
                    const auto* raw_device=copy(sources[s]);
                    const auto links=ElectronContinuationLinks::build(sources[s].data(),sources[s].size(),budget_-bytes_);
                    births.add(links.view(sources[s].data(),sources[s].size()),static_cast<std::uint32_t>(s));
                    MaterialElectronStateSource state{raw_device,sources[s].size(),copy(links.next,device_indices_),
                        copy(links.child_offsets,device_indices_),copy(links.child_rows,device_indices_),links.child_rows.size()};
                    if(short_range_cache) {
                        const auto rows=links.next.size();
                        if(rows>(budget_-bytes_)/sizeof(double))
                            throw std::runtime_error("Short-range cache exceeds response budget");
                        std::vector<double> lengths(rows);
                        const auto source=links.view(sources[s].data(),sources[s].size());
                        for(std::size_t r=0;r<rows;++r)
                            lengths[r]=electron_short_range_length(source,r);
                        state.short_range_lengths=copy(lengths);
                    }
                    const auto crossings=ElectronEnergyCrossings::build(links.view(sources[s].data(),sources[s].size()),budget_-bytes_);
                    state.crossings={copy(crossings.edges,true),copy(crossings.offsets,true),copy(crossings.rows,device_indices_),
                        crossings.edges.size()-1,crossings.rows.size()};
                    source_views.push_back(state);
                }
                view.state_sources=copy(source_views,true);view.state_source_count=source_views.size();
                births.finish(budget_-bytes_);
                view.birth_channels=copy(births.channels,true);view.birth_samples=copy(births.samples,device_indices_);
                view.birth_sample_count=births.samples.size();
            }
            host.push_back(view);
            if(memory_==MaterialElectronMemory::host_mapped)
                std::cout<<"[material-response] mapped tables="<<host.size()<<'/'<<selected.size()
                         <<" resident_MiB="<<bytes_/(1024*1024)<<std::endl;
        }
        // Publish the device view only after the entire required bank loads.
        views_=copy(host,true);count_=host.size();
    }
    const MaterialElectronResponseView* views() const {return views_;}
    std::size_t size() const {return count_;}
    std::size_t allocated_bytes() const {return bytes_;}
    std::size_t hot_device_bytes() const {return hot_bytes_;}
private:
    template<class T> T* copy(const std::vector<T>& source,bool hot=false) {
        if(source.empty())return nullptr;
        if(source.size()>std::numeric_limits<std::size_t>::max()/sizeof(T))throw std::overflow_error("Response allocation size");
        const auto bytes=source.size()*sizeof(T);
        if(bytes>budget_ || bytes_>budget_-bytes)throw std::runtime_error("Material response exceeds device byte budget; no fallback");
        const bool promoted=memory_==MaterialElectronMemory::host_mapped && hot && hot_budget_>0;
        if(promoted && (bytes>hot_budget_ || hot_bytes_>hot_budget_-bytes))
            throw std::runtime_error("Hot metadata device budget exceeded; no silent fallback");
        const bool device=memory_==MaterialElectronMemory::device || promoted;
        auto* dest=device?sycl::malloc_device<T>(source.size(),queue_):
            sycl::malloc_host<T>(source.size(),queue_);
        if(!dest)throw std::bad_alloc();
        try {allocations_.push_back(dest);}catch(...) {sycl::free(dest,queue_);throw;}
        bytes_+=bytes;
        if(promoted)hot_bytes_+=bytes;
        if(!device)std::copy(source.begin(),source.end(),dest);
        else queue_.copy(source.data(),dest,source.size()).wait_and_throw();
        return dest;
    }
    sycl::queue queue_;
    MaterialElectronMemory memory_;
    std::vector<void*> allocations_;
    MaterialElectronResponseView* views_{};
    std::size_t count_{},bytes_{},budget_{};
    std::size_t hot_budget_{},hot_bytes_{};
    bool device_indices_{};
};
} // namespace carbon

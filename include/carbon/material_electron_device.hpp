#pragma once
#include "carbon/material_electron_response.hpp"
#include "carbon/material_electron_lookup.hpp"
#include "carbon/electron_continuation_links.hpp"
#include "carbon/electron_continuation_crossings.hpp"
#include "carbon/electron_birth_index.hpp"
#include "carbon/electron_short_range.hpp"
#include "carbon/material_electron_prepared.hpp"
#include "carbon/electron_packed_graph.hpp"
#include <sycl/sycl.hpp>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <numeric>

namespace carbon {
enum class MaterialElectronMemory { device, host_mapped };
// Host-mapped storage is an explicit opt-in, never an implicit fallback. The
// GPU executes identical kernels on identical bytes through mapped host USM.
class MaterialElectronDeviceBank {
public:
    explicit MaterialElectronDeviceBank(sycl::queue queue,
        MaterialElectronMemory memory=MaterialElectronMemory::device,
        std::size_t hot_device_budget=0,bool device_indices=false,bool tiered=false,bool small_first=false,bool decoded=false):queue_(std::move(queue)),memory_(memory),hot_budget_(hot_device_budget),device_indices_(device_indices),tiered_(tiered),small_first_(small_first),decoded_(decoded) {
        if(tiered_ && (memory_!=MaterialElectronMemory::host_mapped || hot_budget_<metadata_reserve_))
            throw std::invalid_argument("Tiered response requires host_mapped and at least 64 MiB hot budget");
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
              bool packet_only=false,bool short_range_cache=false,bool material_lookup=false,
              const std::filesystem::path& packed_directory={},const std::filesystem::path& compact_directory={}) {
        if(!allocations_.empty() || views_)throw std::logic_error("Response bank already loaded");
        const auto selected=index.required_tables(geometry);
        if(selected.empty())throw std::invalid_argument("Empty response geometry demand");
        budget_=byte_budget;
        const auto compact_configured_root=compact_directory.string();
        const char* compact_root=compact_configured_root.empty()?std::getenv("CARBON_ELECTRON_CARBON_COMPACT_DIR"):compact_configured_root.c_str();
        const bool compact=compact_root && *compact_root;
        if(compact && (!packed_directory.empty() || index.projectile_z!=6 || index.projectile_a!=12))
            throw std::invalid_argument("Carbon compact directory requires C12 and cannot be combined with a proton artifact");
        const char* proof_setting=std::getenv("CARBON_ELECTRON_PREVALIDATED");
        const bool prevalidated=compact || !packed_directory.empty() || (proof_setting && std::string_view(proof_setting)=="1");
        if(prevalidated && (!decoded_ || !packet_only))
            throw std::invalid_argument("Prevalidated replay requires decoded packet-only bank");
        std::size_t proof_rows=0,proof_ready=0,packed_rows_saved=0,packed_links_saved=0;
        const char* packed_setting=std::getenv("CARBON_ELECTRON_PACK_EM_ONLY");
        const bool packed=compact || !packed_directory.empty() || (packed_setting && std::string_view(packed_setting)=="1");
        if(packed && (!decoded_ || !packet_only || short_range_cache || (!compact && (index.projectile_z!=1 || index.projectile_a!=1))))
            throw std::invalid_argument("Packed EM graph is a decoded proton packet-only research mode");

        if(decoded_ && !packet_only)throw std::invalid_argument("Decoded response requires packet-only views");
        // An explicit YAML artifact is authoritative; environment switches remain
        // available for old configs and isolated comparisons with an empty path.
        const auto configured_root=packed_directory.string();
        const char* prepared_root=compact?compact_root:configured_root.empty()?std::getenv("CARBON_MATERIAL_PREPARED_DIR"):configured_root.c_str();
        if(prepared_root && !*prepared_root)prepared_root=nullptr;
        if(prepared_root && packed!=prevalidated)throw std::invalid_argument("Prepared packed artifact requires both packing and prevalidation");
        if(prepared_root && (!decoded_ || !packet_only || short_range_cache))
            throw std::invalid_argument("Prepared bank requires decoded packet-only transport without short-range cache");
#if !defined(CARBON_ELECTRON_CROSSING_ROWS32)
        if(prepared_root)throw std::invalid_argument("Prepared bank requires lossless 32-bit crossing rows");
#endif
        std::vector<MaterialElectronResponseView> host;
        host.resize(selected.size());
        std::vector<std::size_t> load_order(selected.size());
        std::iota(load_order.begin(),load_order.end(),std::size_t{0});
        // Only allocation order changes. Published table IDs remain identical.
        // Reference-file size is a cheap row-count proxy, never physics input.
        if(small_first_) {
            std::vector<std::uintmax_t> sizes(selected.size());
            for(std::size_t i=0;i<selected.size();++i) {
                const auto& file=index.files[selected[i]];
                sizes[i]=std::filesystem::file_size(file.state_reference_file.empty()?file.path:file.state_reference_file);
            }
            std::stable_sort(load_order.begin(),load_order.end(),[&](auto a,auto b){return sizes[a]<sizes[b];});
        }
        std::size_t loaded=0;
        for(auto slot:load_order) {
            const auto id=selected[slot];
            const auto hot_before=hot_bytes_;
            const auto table_start=std::chrono::steady_clock::now();
            if(prepared_root) {
                host[slot]=load_prepared(index,id,prepared_root,packed,compact,proof_rows,proof_ready);
                ++loaded;
                std::cout<<"[material-response-prepared] tables="<<loaded<<'/'<<selected.size()
                         <<" packed="<<packed<<" resident_MiB="<<bytes_/(1024*1024)
                         <<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-table_start).count()<<std::endl;
                continue;
            }
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
                std::vector<std::vector<std::uint64_t>> source_row_maps(sources.size());
                ElectronBirthIndex births(table.channels,table.projectile_z,table.projectile_a);
                const int projectile_pdg=table.projectile_z==1 && table.projectile_a==1?2212:
                    1000000000+table.projectile_z*10000+table.projectile_a*10;
                for(std::size_t s=0;s<sources.size();++s) {
                    const auto expected=ElectronStateReferenceView::integer(payload.data()+32+8*s,8);
                    if(sources[s].size()/kElectronStateV4RecordBytes!=expected)
                        throw std::invalid_argument("State-source row count differs from references");
                    const auto links=ElectronContinuationLinks::build(sources[s].data(),sources[s].size(),budget_-bytes_,projectile_pdg);
                    const auto original=links.view(sources[s].data(),sources[s].size());
                    births.add(original,static_cast<std::uint32_t>(s));
                    MaterialElectronStateSource state;
                    if(packed) {
                        PackedElectronGraph graph(original);
                        state.bytes=graph.decoded.size()*kElectronStateV4RecordBytes;
                        state.decoded=copy(graph.decoded,!(tiered_ && device_indices_),true);
                        decoded_bytes_saved_+=sources[s].size()-graph.decoded.size()*sizeof(ElectronContinuationStep);
                        state.next_rows=copy(graph.next,device_indices_,true);
                        state.child_offsets=copy(graph.offsets,device_indices_,true);
                        state.child_rows=copy(graph.children,device_indices_,true);state.child_count=graph.children.size();
                        packed_rows_saved+=links.next.size()-graph.decoded.size();
                        packed_links_saved+=(links.next.size()+links.child_offsets.size()+links.child_rows.size()-
                                             graph.next.size()-graph.offsets.size()-graph.children.size())*sizeof(std::uint64_t);
                        if(prevalidated) {
                            std::vector<double> certified(graph.decoded.size());
                            for(std::size_t row=0;row<certified.size();++row) {
                                const auto replay=prepare_electron_replay_step(original,graph.original_rows[row]);
                                if(replay.status!=ElectronReplayStatus::ready)throw std::runtime_error("Packed proof is not ready");
                                certified[row]=replay.child_energy_MeV;
                            }
                            proof_rows+=certified.size();proof_ready+=certified.size();
                            state.certified_child_energy=copy(certified,true,true);
                        }
                        source_row_maps[s]=std::move(graph.packed_rows);
                    } else {
                        if(decoded_) {
                            static_assert(sizeof(ElectronContinuationStep)<=kElectronStateV4RecordBytes);
                            std::vector<ElectronContinuationStep> decoded(sources[s].size()/kElectronStateV4RecordBytes);
                            for(std::size_t row=0;row<decoded.size();++row)
                                decoded[row]=read_electron_continuation_step(sources[s].data(),sources[s].size(),row);
                            state.decoded=copy(decoded,!(tiered_ && device_indices_),true);
                            decoded_bytes_saved_+=sources[s].size()-decoded.size()*sizeof(ElectronContinuationStep);
                        } else state.data=copy(sources[s],!(tiered_ && device_indices_),true);
                        state.bytes=sources[s].size();state.next_rows=copy(links.next,device_indices_,true);
                        state.child_offsets=copy(links.child_offsets,device_indices_,true);
                        state.child_rows=copy(links.child_rows,device_indices_,true);state.child_count=links.child_rows.size();
                        if(prevalidated) {
                            std::vector<double> certified(links.next.size(),-1.);
                            for(std::size_t row=0;row<certified.size();++row) {
                                const auto replay=prepare_electron_replay_step(original,row);
                                if(replay.status==ElectronReplayStatus::ready) {certified[row]=replay.child_energy_MeV;++proof_ready;}
                            }
                            proof_rows+=certified.size();state.certified_child_energy=copy(certified,true,true);
                        }
                    }
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
                    auto crossings=ElectronEnergyCrossings::build(original,budget_-bytes_);
                    if(packed)for(auto& row:crossings.rows) {
                        if(row>=source_row_maps[s].size() || source_row_maps[s][row]==kElectronContinuationEnd)
                            throw std::runtime_error("Crossing references removed row");
                        row=source_row_maps[s][row];
                    }
                    state.crossings.edges=copy(crossings.edges,true);
                    state.crossings.offsets=copy(crossings.offsets,true);
                    state.crossings.bins=crossings.edges.size()-1;
                    state.crossings.count=crossings.rows.size();
#if defined(CARBON_ELECTRON_CROSSING_ROWS32)
                    const auto source_rows=state.bytes/kElectronStateV4RecordBytes;
                    if(source_rows<=std::uint64_t{std::numeric_limits<std::uint32_t>::max()}+1) {
                        std::vector<std::uint32_t> compact;
                        compact.reserve(crossings.rows.size());
                        for(auto row:crossings.rows) {
                            if(row>=source_rows || row>std::numeric_limits<std::uint32_t>::max())
                                throw std::runtime_error("Electron crossing row cannot be represented losslessly");
                            compact.push_back(static_cast<std::uint32_t>(row));
                        }
                        state.crossings.rows32=copy(compact,device_indices_,true);
                        crossing_bytes_saved_+=crossings.rows.size()*sizeof(std::uint32_t);
                        ++compact_crossing_sources_;
                    } else
#endif
                    {
                        state.crossings.rows=copy(crossings.rows,device_indices_,true);
                    }
                    source_views.push_back(state);
                }
                view.state_sources=copy(source_views,true);view.state_source_count=source_views.size();
                births.finish(budget_-bytes_);
                if(packed)for(auto& birth:births.samples) {
                    if(birth.source>=source_row_maps.size() || birth.row>=source_row_maps[birth.source].size() ||
                       source_row_maps[birth.source][birth.row]==kElectronContinuationEnd)
                        throw std::runtime_error("Birth references removed row");
                    birth.row=source_row_maps[birth.source][birth.row];
                }
                view.birth_channels=copy(births.channels,true);view.birth_samples=copy(births.samples,true,true);
                view.birth_sample_count=births.samples.size();
            }
            host[slot]=view;
            ++loaded;
            std::cout<<"[material-response-placement] table="<<slot<<" section="<<view.section
                     <<" density="<<view.density_g_cm3<<" hot_bytes="<<hot_bytes_-hot_before<<"\n";
            if(memory_==MaterialElectronMemory::host_mapped)
                std::cout<<"[material-response] mapped tables="<<loaded<<'/'<<selected.size()
                         <<" resident_MiB="<<bytes_/(1024*1024)<<std::endl;
        }
        // Publish the device view only after the entire required bank loads.
        if(material_lookup) {
            std::vector<std::size_t> order(host.size());
            std::iota(order.begin(),order.end(),std::size_t{0});
            std::stable_sort(order.begin(),order.end(),[&](auto a,auto b) {
                return host[a].section<host[b].section;
            });
            std::vector<MaterialElectronLookupSection> sections;
            std::vector<MaterialElectronLookupEntry> entries;
            entries.reserve(host.size());
            for(const auto table:order) {
                if(sections.empty() || sections.back().section!=host[table].section)
                    sections.push_back({host[table].section,entries.size(),entries.size()});
                entries.push_back({host[table].density_g_cm3,table});
                sections.back().end=entries.size();
            }
            lookup_={copy(sections,true),copy(entries,true),sections.size(),host.size()};
        }
        views_=copy(host,true);count_=host.size();
        std::cout<<"[electron-packed-graph] enabled="<<packed<<" rows_removed="<<packed_rows_saved
                 <<" decoded_bytes_saved="<<packed_rows_saved*sizeof(ElectronContinuationStep)
                 <<" link_bytes_saved="<<packed_links_saved<<std::endl;
        std::cout<<"[electron-prevalidated] enabled="<<prevalidated<<" rows="<<proof_rows
                 <<" ready="<<proof_ready<<" bytes="<<proof_rows*sizeof(double)<<std::endl;

        std::cout<<"[material-response-layout] policy="<<(small_first_?"tiered_small_first":(tiered_?(device_indices_?"tiered_indices":"tiered"):"strict"))
                 <<" total_bytes="<<bytes_<<" hot_device_bytes="<<hot_bytes_
                 <<" metadata_device_bytes="<<metadata_bytes_
                 <<" bulk_device_bytes="<<bulk_bytes_
                 <<" budget_mapped_bulk_bytes="<<budget_mapped_bytes_
                 <<" crossing_bytes_saved="<<crossing_bytes_saved_
                 <<" decoded="<<decoded_<<" decoded_record_bytes="<<sizeof(ElectronContinuationStep)
                 <<" decoded_bytes_saved="<<decoded_bytes_saved_
                 <<" compact_crossing_sources="<<compact_crossing_sources_<<std::endl;
    }
    const MaterialElectronResponseView* views() const {return views_;}
    MaterialElectronLookupView lookup() const {return lookup_;}
    std::size_t size() const {return count_;}
    std::size_t allocated_bytes() const {return bytes_;}
    std::size_t hot_device_bytes() const {return hot_bytes_;}
private:
    template<class T> const T* read_prepared(MaterialPreparedReader& reader,const std::string& name,
                                            std::size_t& count,bool hot=false,bool bulk=false) {
        T* dest=nullptr;bool device=false;std::vector<T> staging;
        reader.read<T>(name,[&](std::size_t n) {
            dest=allocate<T>(n,hot,bulk,device);
            if(device) {staging.resize(n);return staging.data();}
            return dest;
        },count);
        if(device && count)queue_.copy(staging.data(),dest,count).wait_and_throw();
        return dest;
    }
    MaterialElectronResponseView load_prepared(const MaterialElectronResponseIndex& index,std::size_t id,
                                               const std::filesystem::path& root,bool packed,bool compact,std::size_t& proof_rows,std::size_t& proof_ready) {
        MaterialPreparedReader reader(root,material_prepared_key(index,id,packed,compact),packed,compact);
        const auto& file=index.files[id];
        if(reader.section!=file.section || reader.density!=file.density_g_cm3 ||
           reader.reference_density!=file.reference_density_g_cm3)
            throw std::runtime_error("Prepared material identity mismatch");
        MaterialElectronResponseView view;
        view.section=reader.section;view.density_g_cm3=reader.density;
        view.reference_density_g_cm3=reader.reference_density;view.node_count=reader.nodes;
        view.channels=read_prepared<WaterElectronChannel>(reader,"channels",view.channel_count,true);
        std::vector<MaterialElectronStateSource> sources;
        for(std::size_t s=0;s<reader.sources;++s) {
            MaterialElectronStateSource state;std::size_t rows=0,n=0;ElectronCompactStateView cv;
            if(compact) {
                cv.records=read_prepared<ElectronEndpointRecord>(reader,"endpoint_records",rows,!(tiered_ && device_indices_),true);cv.rows=rows;
                cv.endpoints=read_prepared<ElectronEndpoint>(reader,"endpoints",cv.endpoint_count,!(tiered_ && device_indices_),true);
                if(rows+cv.endpoint_count>UINT32_MAX)throw std::runtime_error("Prepared endpoint count invalid");
                cv.tracks=read_prepared<ElectronCompactTrack>(reader,"compact_tracks",cv.track_count,device_indices_,true);
                cv.media=read_prepared<ElectronCompactMedium>(reader,"compact_media",cv.medium_count,true);
                if(!cv.track_count || cv.track_count>UINT32_MAX || !cv.medium_count || cv.medium_count>UINT32_MAX)
                    throw std::runtime_error("Prepared compact dictionary count invalid");
            } else state.decoded=read_prepared<ElectronContinuationStep>(reader,"decoded",rows,!(tiered_ && device_indices_),true);
            if(!rows || rows>UINT32_MAX)throw std::runtime_error("Prepared rows outside compact domain");
            state.bytes=rows*kElectronStateV4RecordBytes;
            if(!compact)decoded_bytes_saved_+=rows*(kElectronStateV4RecordBytes-sizeof(ElectronContinuationStep));
            if(compact) {
                if(rows>=UINT32_MAX)throw std::runtime_error("Prepared graph exceeds uint32 domain");
                cv.next32=read_prepared<std::uint32_t>(reader,"next32",n,device_indices_,true);
                if(n!=rows)throw std::runtime_error("Prepared next count mismatch");
                cv.child_offsets32=read_prepared<std::uint32_t>(reader,"child_offsets32",n,device_indices_,true);
                if(n!=rows+1)throw std::runtime_error("Prepared offsets count mismatch");
                cv.children32=read_prepared<std::uint32_t>(reader,"children32",n,device_indices_,true);
                if(n>=UINT32_MAX)throw std::runtime_error("Prepared children exceed uint32 domain");
                state.child_count=n;state.compact=copy(std::vector<ElectronCompactStateView>{cv},true);
            } else {
            state.next_rows=read_prepared<std::uint64_t>(reader,"next",n,device_indices_,true);
            if(n!=rows)throw std::runtime_error("Prepared next count mismatch");
            state.child_offsets=read_prepared<std::uint64_t>(reader,"child_offsets",n,device_indices_,true);
            if(n!=rows+1)throw std::runtime_error("Prepared offsets count mismatch");
            state.child_rows=read_prepared<std::uint64_t>(reader,"children",n,device_indices_,true);state.child_count=n;
            }
            if(packed) {
                state.certified_child_energy=read_prepared<double>(reader,"certified_child_energy",n,true,true);
                if(n!=rows)throw std::runtime_error("Prepared proof count mismatch");
                proof_rows+=rows;proof_ready+=rows;
            }
            state.crossings.edges=read_prepared<double>(reader,"edges",n,true);
            if(n!=258)throw std::runtime_error("Prepared energy edges mismatch");
            state.crossings.bins=n-1;
            state.crossings.offsets=read_prepared<std::uint64_t>(reader,"offsets",n,true);
            if(n!=state.crossings.bins+1)throw std::runtime_error("Prepared energy offsets mismatch");
            if(compact) {
                std::uint64_t logical=0;
                reader.read<std::uint64_t>("logical_crossing_count",[&](std::size_t count){
                    if(count!=1)throw std::runtime_error("Prepared logical count descriptor invalid");
                    return &logical;
                },n);
                ElectronBitmapCrossingsView bitmap;
                bitmap.bins=read_prepared<ElectronBitmapBin>(reader,"bitmap_bins",n,true);
                if(n!=state.crossings.bins)throw std::runtime_error("Prepared bitmap bin count mismatch");
                bitmap.sparse=read_prepared<std::uint32_t>(reader,"bitmap_sparse",n,device_indices_,true);auto stored=n*4;
                bitmap.words=read_prepared<std::uint64_t>(reader,"bitmap_words",n,device_indices_,true);stored+=n*8;
                bitmap.ranks=read_prepared<std::uint32_t>(reader,"bitmap_ranks",n,true,true);stored+=n*4;
                state.crossings.bitmap=copy(std::vector<ElectronBitmapCrossingsView>{bitmap},true);
                state.crossings.count=logical;
                if(logical*8>stored)crossing_bytes_saved_+=logical*8-stored;
            } else {
                state.crossings.rows32=read_prepared<std::uint32_t>(reader,"crossings32",n,device_indices_,true);
                state.crossings.count=n;crossing_bytes_saved_+=n*4;
            }
            ++compact_crossing_sources_;
            sources.push_back(state);
        }
        view.state_sources=copy(sources,true);view.state_source_count=sources.size();
        std::size_t n=0;
        view.birth_channels=read_prepared<ElectronBirthChannel>(reader,"birth_channels",n,true);
        if(n!=view.channel_count)throw std::runtime_error("Prepared birth channel count mismatch");
        view.birth_samples=read_prepared<ElectronBirthSample>(reader,"birth_samples",n,true,true);view.birth_sample_count=n;
        reader.finish();return view;
    }
    template<class T> T* allocate(std::size_t count,bool hot,bool bulk,bool& device) {
        if(!count) {device=false;return nullptr;}
        if(count>std::numeric_limits<std::size_t>::max()/sizeof(T))throw std::overflow_error("Response allocation size");
        const auto bytes=count*sizeof(T);
        if(bytes>budget_ || bytes_>budget_-bytes)throw std::runtime_error("Material response exceeds device byte budget; no fallback");
        bool promoted=memory_==MaterialElectronMemory::host_mapped && hot && hot_budget_>0;
        // Explicit tiered policy: reserve metadata capacity before considering
        // any bulk allocation. A non-fitting bulk array stays intact in mapped
        // memory; allocation failures still throw and never trigger a fallback.
        if(promoted && tiered_) {
            if(bulk) {
                const auto capacity=hot_budget_-metadata_reserve_;
                if(bytes>capacity || bulk_bytes_>capacity-bytes) {
                    promoted=false;budget_mapped_bytes_+=bytes;
                }
            } else if(bytes>metadata_reserve_ || metadata_bytes_>metadata_reserve_-bytes) {
                throw std::runtime_error("Tiered response metadata exceeds reserved 64 MiB");
            }
        }
        if(promoted && (bytes>hot_budget_ || hot_bytes_>hot_budget_-bytes))
            throw std::runtime_error("Hot metadata device budget exceeded; no silent fallback");
        device=memory_==MaterialElectronMemory::device || promoted;
        auto* dest=device?sycl::malloc_device<T>(count,queue_):
            sycl::malloc_host<T>(count,queue_);
        if(!dest)throw std::bad_alloc();
        try {allocations_.push_back(dest);}catch(...) {sycl::free(dest,queue_);throw;}
        bytes_+=bytes;
        if(promoted) {hot_bytes_+=bytes;(bulk?bulk_bytes_:metadata_bytes_)+=bytes;}
        return dest;
    }
    template<class T> T* copy(const std::vector<T>& source,bool hot=false,bool bulk=false) {
        bool device=false;T* dest=allocate<T>(source.size(),hot,bulk,device);
        if(source.empty())return dest;
        if(!device)std::copy(source.begin(),source.end(),dest);
        else queue_.copy(source.data(),dest,source.size()).wait_and_throw();
        return dest;
    }
    sycl::queue queue_;
    MaterialElectronMemory memory_;
    std::vector<void*> allocations_;
    MaterialElectronResponseView* views_{};
    MaterialElectronLookupView lookup_{};
    std::size_t count_{},bytes_{},budget_{};
    std::size_t hot_budget_{},hot_bytes_{};
    bool device_indices_{},tiered_{},small_first_{},decoded_{};
    std::size_t decoded_bytes_saved_{};
    static constexpr std::size_t metadata_reserve_=64ULL*1024*1024;
    std::size_t metadata_bytes_{},bulk_bytes_{},budget_mapped_bytes_{};
    std::size_t crossing_bytes_saved_{},compact_crossing_sources_{};
};
} // namespace carbon

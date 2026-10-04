#include "carbon/material_electron_prepared.hpp"
#include "carbon/electron_continuation_links.hpp"
#include "carbon/electron_continuation_crossings.hpp"
#include "carbon/electron_birth_index.hpp"
#include <iostream>
#include "carbon/electron_packed_graph.hpp"
#include "carbon/electron_compact_builder.hpp"
#include "carbon/electron_compact_crossings.hpp"
#include "carbon/electron_bitmap_builder.hpp"
#include "carbon/electron_endpoint_builder.hpp"

int main(int argc,char** argv) {
    try {
        if(argc!=8 && argc!=9)throw std::invalid_argument("Usage: prepare INDEX INDEX_SHA CACHE_ROOT Z A BEGIN END [packed]");
        const auto index=carbon::MaterialElectronResponseIndex::load(argv[1],argv[2],std::stoi(argv[4]),std::stoi(argv[5]));
        const bool compact=argc==9 && std::string(argv[8])=="carbon_compact";
        const bool packed=compact || (argc==9 && std::string(argv[8])=="packed");
        if(argc==9 && !packed)throw std::invalid_argument("Unknown artifact mode");
        if(compact && (index.projectile_z!=6 || index.projectile_a!=12))throw std::invalid_argument("Carbon compact artifact requires C12 source");
        if(packed && !compact && (index.projectile_z!=1 || index.projectile_a!=1))throw std::invalid_argument("Packed artifact requires proton source");
        const std::filesystem::path root=argv[3];std::filesystem::create_directories(root);
        const auto begin=std::stoul(argv[6]),end=std::min<std::size_t>(std::stoul(argv[7]),index.files.size());
        constexpr std::size_t budget=96ULL*1024*1024*1024;
        for(std::size_t id=begin;id<end;++id) {
            const auto key=carbon::material_prepared_key(index,id,packed,compact);const auto dest=root/key;
            if(std::filesystem::exists(dest)) {
                carbon::MaterialPreparedReader check(root,key,packed,compact);
                std::cout<<"existing "<<id<<' '<<key<<std::endl;continue;
            }
            const auto start=std::chrono::steady_clock::now();
            const auto table=index.load_table(id);
            const auto refs=index.load_state_references(id,budget);
            const auto view=carbon::ElectronStateReferenceView::parse(refs.data(),refs.size());
            if(!view.valid || view.samples!=table.samples.size() || view.nodes!=table.nodes.size())
                throw std::runtime_error("Prepared source/reference mismatch");
            const auto sources=index.load_state_sources(id,budget);
            if(sources.size()!=view.sources)throw std::runtime_error("Prepared source count mismatch");
            // Unique staging prevents readers observing partially written banks.
            const auto staging=root/(key+".partial."+std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
            carbon::MaterialPreparedWriter writer(staging,key,table,index.files[id].density_g_cm3,sources.size(),packed,compact);
            writer.write("channels",table.channels);
            carbon::ElectronBirthIndex births(table.channels,table.projectile_z,table.projectile_a);
            const int pdg=table.projectile_z==1 && table.projectile_a==1?2212:
                1000000000+table.projectile_z*10000+table.projectile_a*10;
            std::vector<std::vector<std::uint64_t>> row_maps(sources.size());
            for(std::size_t s=0;s<sources.size();++s) {
                const auto& raw=sources[s];const auto rows=raw.size()/carbon::kElectronStateV4RecordBytes;
                if(rows!=carbon::ElectronStateReferenceView::integer(refs.data()+32+8*s,8) || rows>UINT32_MAX)
                    throw std::runtime_error("Prepared row count mismatch or requires 64-bit crossing rows");
                const auto links=carbon::ElectronContinuationLinks::build(raw.data(),raw.size(),budget,pdg);
                const auto original=links.view(raw.data(),raw.size());
                births.add(original,static_cast<std::uint32_t>(s));
                if(packed) {
                    carbon::PackedElectronGraph graph(original);
                    if(compact) {
                        carbon::ElectronCompactBuilder fields;
                        fields.records.reserve(graph.decoded.size());
                        for(const auto& record:graph.decoded) {
                            fields.add(record);
                            if(!carbon::electron_state_fields_equal(record,fields.view().read(fields.records.size()-1)))
                                throw std::runtime_error("Compact state bitwise roundtrip failed");
                        }
                        fields.release_indices();
                        const auto endpoints=carbon::ElectronEndpointBuilder::build(fields.records,carbon::electron_graph32(graph.next,true));
                        carbon::ElectronCompactStateView compact_view{endpoints.records.data(),fields.tracks.data(),fields.media.data(),endpoints.records.size(),fields.tracks.size(),fields.media.size(),endpoints.endpoints.data(),endpoints.endpoints.size()};
                        for(std::size_t row=0;row<graph.decoded.size();++row)
                            if(!carbon::electron_state_fields_equal(graph.decoded[row],compact_view.read(row)))throw std::runtime_error("Endpoint roundtrip failed");
                        writer.write("endpoint_records",endpoints.records);writer.write("endpoints",endpoints.endpoints);writer.write("compact_tracks",fields.tracks);writer.write("compact_media",fields.media);
                        std::cout<<"compact_state "<<id<<" source="<<s<<" rows="<<graph.decoded.size()<<" bytes="<<fields.bytes()<<std::endl;
                    } else writer.write("decoded",graph.decoded);
                    if(compact) {
                        writer.write("next32",carbon::electron_graph32(graph.next,true));
                        writer.write("child_offsets32",carbon::electron_graph32(graph.offsets));
                        writer.write("children32",carbon::electron_graph32(graph.children));
                    } else {writer.write("next",graph.next);writer.write("child_offsets",graph.offsets);writer.write("children",graph.children);}
                    std::vector<double> certified(graph.decoded.size());
                    for(std::size_t row=0;row<certified.size();++row) {
                        const auto replay=carbon::prepare_electron_replay_step(original,graph.original_rows[row]);
                        if(replay.status!=carbon::ElectronReplayStatus::ready)throw std::runtime_error("Packed proof not ready");
                        certified[row]=replay.child_energy_MeV;
                    }
                    writer.write("certified_child_energy",certified);
                    row_maps[s]=std::move(graph.packed_rows);
                } else {
                    std::vector<carbon::ElectronContinuationStep> decoded(rows);
                    for(std::size_t r=0;r<rows;++r)decoded[r]=original.read(r);
                    writer.write("decoded",decoded);
                    writer.write("next",links.next);writer.write("child_offsets",links.child_offsets);writer.write("children",links.child_rows);
                }
                if(compact) {
                    auto crossings=carbon::ElectronCompactCrossings::build(original,budget);
                    for(auto& row:crossings.rows) {
                        if(row>=row_maps[s].size() || row_maps[s][row]==carbon::kElectronContinuationEnd || row_maps[s][row]>UINT32_MAX)
                            throw std::runtime_error("Compact crossing remap invalid");
                        row=static_cast<std::uint32_t>(row_maps[s][row]);
                    }
                    writer.write("edges",crossings.edges);writer.write("offsets",crossings.offsets);
                    const auto kept=std::count_if(row_maps[s].begin(),row_maps[s].end(),[](auto row){return row!=carbon::kElectronContinuationEnd;});
                    const auto bitmap=carbon::ElectronBitmapBuilder::build(kept,crossings.offsets,crossings.starts,crossings.rows);
                    writer.write("logical_crossing_count",std::vector<std::uint64_t>{crossings.offsets.back()});
                    writer.write("bitmap_bins",bitmap.bins);writer.write("bitmap_sparse",bitmap.sparse);
                    writer.write("bitmap_words",bitmap.words);writer.write("bitmap_ranks",bitmap.ranks);
                    std::cout<<"compact_crossings "<<id<<" logical="<<crossings.offsets.back()<<" stored="<<crossings.rows.size()<<" groups="<<crossings.unique_bins<<std::endl;
                } else {
                const auto crossings=carbon::ElectronEnergyCrossings::build(links.view(raw.data(),raw.size()),budget);
                writer.write("edges",crossings.edges);writer.write("offsets",crossings.offsets);
                std::vector<std::uint32_t> compact;compact.reserve(crossings.rows.size());
                for(auto row:crossings.rows) {
                    if(row>=rows)throw std::runtime_error("Prepared crossing row out of bounds");
                    if(packed) {
                        if(row_maps[s][row]==carbon::kElectronContinuationEnd)throw std::runtime_error("Packed crossing removed");
                        row=row_maps[s][row];
                    }
                    compact.push_back(static_cast<std::uint32_t>(row));
                }
                writer.write("crossings32",compact);
                }
            }
            births.finish(budget);
            if(packed)for(auto& birth:births.samples) {
                if(birth.source>=row_maps.size() || birth.row>=row_maps[birth.source].size() ||
                   row_maps[birth.source][birth.row]==carbon::kElectronContinuationEnd)throw std::runtime_error("Packed birth removed");
                birth.row=row_maps[birth.source][birth.row];
            }
            writer.write("birth_channels",births.channels);writer.write("birth_samples",births.samples);
            writer.finish();std::filesystem::rename(staging,dest);
            std::cout<<"prepared "<<id<<' '<<key<<" seconds="
                <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<std::endl;
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

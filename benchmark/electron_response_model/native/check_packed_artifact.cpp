#include "carbon/material_electron_prepared.hpp"
#include "carbon/electron_packed_graph.hpp"
#include "carbon/material_electron_response.hpp"
#include "carbon/electron_birth_index.hpp"
#include "carbon/electron_continuation_crossings.hpp"
#include "carbon/electron_continuation_links.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
void compare(const carbon::ElectronReplayStep&a,const carbon::ElectronReplayStep&b) {
 if(std::memcmp(&a.status,&b.status,sizeof(a.status)))throw std::runtime_error("status");
 if(std::memcmp(&a.next,&b.next,sizeof(a.next)))throw std::runtime_error("next");
 if(std::memcmp(&a.children_begin,&b.children_begin,sizeof(a.children_begin)))throw std::runtime_error("children_begin");
 if(std::memcmp(&a.children_end,&b.children_end,sizeof(a.children_end)))throw std::runtime_error("children_end");
 if(std::memcmp(&a.child_energy_MeV,&b.child_energy_MeV,sizeof(a.child_energy_MeV)))throw std::runtime_error("child_energy_MeV");
 if(std::memcmp(&a.energy_residual_MeV,&b.energy_residual_MeV,sizeof(a.energy_residual_MeV)))throw std::runtime_error("energy_residual_MeV");
 if(std::memcmp(&a.state.valid,&b.state.valid,sizeof(a.state.valid)))throw std::runtime_error("state.valid");
 if(std::memcmp(&a.state.run,&b.state.run,sizeof(a.state.run)))throw std::runtime_error("state.run");
 if(std::memcmp(&a.state.event,&b.state.event,sizeof(a.state.event)))throw std::runtime_error("state.event");
 if(std::memcmp(&a.state.track,&b.state.track,sizeof(a.state.track)))throw std::runtime_error("state.track");
 if(std::memcmp(&a.state.parent,&b.state.parent,sizeof(a.state.parent)))throw std::runtime_error("state.parent");
 if(std::memcmp(&a.state.pdg,&b.state.pdg,sizeof(a.state.pdg)))throw std::runtime_error("state.pdg");
 if(std::memcmp(&a.state.step,&b.state.step,sizeof(a.state.step)))throw std::runtime_error("state.step");
 if(std::memcmp(&a.state.parent_valid,&b.state.parent_valid,sizeof(a.state.parent_valid)))throw std::runtime_error("state.parent_valid");
 if(std::memcmp(&a.state.parent_step,&b.state.parent_step,sizeof(a.state.parent_step)))throw std::runtime_error("state.parent_step");
 if(std::memcmp(&a.state.pre_position,&b.state.pre_position,sizeof(a.state.pre_position)))throw std::runtime_error("state.pre_position");
 if(std::memcmp(&a.state.post_position,&b.state.post_position,sizeof(a.state.post_position)))throw std::runtime_error("state.post_position");
 if(std::memcmp(&a.state.pre_direction,&b.state.pre_direction,sizeof(a.state.pre_direction)))throw std::runtime_error("state.pre_direction");
 if(std::memcmp(&a.state.post_direction,&b.state.post_direction,sizeof(a.state.post_direction)))throw std::runtime_error("state.post_direction");
 if(std::memcmp(&a.state.parent_direction,&b.state.parent_direction,sizeof(a.state.parent_direction)))throw std::runtime_error("state.parent_direction");
 if(std::memcmp(&a.state.pre_energy_MeV,&b.state.pre_energy_MeV,sizeof(a.state.pre_energy_MeV)))throw std::runtime_error("state.pre_energy_MeV");
 if(std::memcmp(&a.state.post_energy_MeV,&b.state.post_energy_MeV,sizeof(a.state.post_energy_MeV)))throw std::runtime_error("state.post_energy_MeV");
 if(std::memcmp(&a.state.deposited_MeV,&b.state.deposited_MeV,sizeof(a.state.deposited_MeV)))throw std::runtime_error("state.deposited_MeV");
 if(std::memcmp(&a.state.length_mm,&b.state.length_mm,sizeof(a.state.length_mm)))throw std::runtime_error("state.length_mm");
 if(std::memcmp(&a.state.birth_energy_MeV,&b.state.birth_energy_MeV,sizeof(a.state.birth_energy_MeV)))throw std::runtime_error("state.birth_energy_MeV");
 if(std::memcmp(&a.state.pre_density_g_cm3,&b.state.pre_density_g_cm3,sizeof(a.state.pre_density_g_cm3)))throw std::runtime_error("state.pre_density_g_cm3");
 if(std::memcmp(&a.state.post_density_g_cm3,&b.state.post_density_g_cm3,sizeof(a.state.post_density_g_cm3)))throw std::runtime_error("state.post_density_g_cm3");
 if(std::memcmp(&a.state.pre_material,&b.state.pre_material,sizeof(a.state.pre_material)))throw std::runtime_error("state.pre_material");
 if(std::memcmp(&a.state.post_material,&b.state.post_material,sizeof(a.state.post_material)))throw std::runtime_error("state.post_material");
 if(std::memcmp(&a.state.post_step_status,&b.state.post_step_status,sizeof(a.state.post_step_status)))throw std::runtime_error("state.post_step_status");
 if(std::memcmp(&a.state.track_status,&b.state.track_status,sizeof(a.state.track_status)))throw std::runtime_error("state.track_status");
}

template<class T> std::vector<T> block(carbon::MaterialPreparedReader& r,const std::string& name) {
 std::vector<T> v;std::size_t n=0;r.read<T>(name,[&](std::size_t count){v.resize(count);return v.data();},n);return v;
}
template<class T> void eq(const std::vector<T>&a,const std::vector<T>&b) {
 if(a!=b)throw std::runtime_error("Prepared vector differs");
}
int main(int argc,char**argv) {
 if(argc!=3 && argc!=4)throw std::runtime_error("Usage: check INDEX CACHE [TABLE_ID]");
 const auto index=carbon::MaterialElectronResponseIndex::load(argv[1],carbon::compute_file_sha256_hex(argv[1]),1,1);
 std::uint64_t checked=0,birth_count=0,crossing_count=0,draw_count=0;
 constexpr std::size_t budget=64ULL<<30;
 for(std::size_t id=argc==4?std::stoul(argv[3]):0;id<(argc==4?std::stoul(argv[3])+1:index.files.size());++id) {
  carbon::MaterialPreparedReader reader(argv[2],carbon::material_prepared_key(index,id,true),true);
  const auto table=index.load_table(id);
  if(reader.section!=table.material_section || reader.density!=index.files[id].density_g_cm3 || reader.reference_density!=table.reference_density_g_cm3 || reader.nodes!=table.nodes.size())throw std::runtime_error("Material mismatch");
  const auto channels=block<carbon::WaterElectronChannel>(reader,"channels");
  if(channels.size()!=table.channels.size())throw std::runtime_error("Channel count");
  for(std::size_t i=0;i<channels.size();++i){const auto&a=channels[i];const auto&b=table.channels[i];if(a.low!=b.low || a.high!=b.high || a.fraction!=b.fraction || a.unresolved!=b.unresolved || a.offset!=b.offset || a.count!=b.count)throw std::runtime_error("Channel field");}
  const auto sources=index.load_state_sources(id,budget);
  if(reader.sources!=sources.size())throw std::runtime_error("Source count");
  carbon::ElectronBirthIndex births(table.channels,1,1);
  std::vector<carbon::PackedElectronGraph> graphs;
  for(std::size_t source=0;source<sources.size();++source) {
   const auto& raw=sources[source];auto links=carbon::ElectronContinuationLinks::build(raw.data(),raw.size(),budget,2212);
   const auto a=links.view(raw.data(),raw.size());births.add(a,source);
   graphs.emplace_back(a);const auto& graph=graphs.back();
   const auto decoded=block<carbon::ElectronContinuationStep>(reader,"decoded");
   const auto next=block<std::uint64_t>(reader,"next"),offsets=block<std::uint64_t>(reader,"child_offsets"),children=block<std::uint64_t>(reader,"children");
   const auto proofs=block<double>(reader,"certified_child_energy");
   if(decoded.size()!=graph.decoded.size() || proofs.size()!=decoded.size())throw std::runtime_error("Packed row count");
   eq(next,graph.next);eq(offsets,graph.offsets);eq(children,graph.children);
   const carbon::ElectronContinuationView b{nullptr,decoded.size()*312,next.data(),offsets.data(),children.data(),children.size(),decoded.data(),proofs.data()};
   for(std::size_t row=0;row<graph.decoded.size();++row) {
    auto before=carbon::prepare_electron_replay_step(a,graph.original_rows[row]),after=carbon::prepare_electron_replay_step(b,row);
    if(before.children_end-before.children_begin!=after.children_end-after.children_begin)throw std::runtime_error("child count");
    for(std::uint64_t i=0;i<before.children_end-before.children_begin;++i)
     if(a.child_rows[before.children_begin+i]!=graph.original_rows[b.child_rows[after.children_begin+i]])throw std::runtime_error("child identity/order");
    if(after.next!=carbon::kElectronContinuationEnd)after.next=graph.original_rows[after.next];
    after.children_begin=before.children_begin;after.children_end=before.children_end;compare(before,after);
   }
   checked+=graph.decoded.size();
   for(std::size_t i=0;i<proofs.size();++i) {const auto before=carbon::prepare_electron_replay_step(a,graph.original_rows[i]);if(std::memcmp(&proofs[i],&before.child_energy_MeV,8))throw std::runtime_error("Proof field differs");}
   auto crossings=carbon::ElectronEnergyCrossings::build(a,budget);const auto original_crossings=crossings;
   for(auto& row:crossings.rows)row=graph.map(row);
   if(crossings.edges!=original_crossings.edges || crossings.offsets!=original_crossings.offsets)throw std::runtime_error("crossing buckets changed");
   for(std::size_t i=0;i<crossings.rows.size();++i)
    if(graph.original_rows[crossings.rows[i]]!=original_crossings.rows[i])throw std::runtime_error("crossing identity/order");
   eq(block<double>(reader,"edges"),crossings.edges);eq(block<std::uint64_t>(reader,"offsets"),crossings.offsets);
   const auto rows32=block<std::uint32_t>(reader,"crossings32");
   if(rows32.size()!=crossings.rows.size())throw std::runtime_error("Crossing count");
   for(std::size_t i=0;i<rows32.size();++i)if(rows32[i]!=crossings.rows[i])throw std::runtime_error("Crossing row");
   crossing_count+=crossings.rows.size();
  }
  births.finish(budget);auto packed_samples=births.samples;
  for(auto& birth:packed_samples)birth.row=graphs.at(birth.source).map(birth.row);
  for(std::size_t i=0;i<packed_samples.size();++i) {
   const auto& a=births.samples[i];const auto& b=packed_samples[i];
   if(a.cdf!=b.cdf || a.source!=b.source || a.row!=graphs[b.source].original_rows[b.row])throw std::runtime_error("birth CDF/identity/order");
  }
  const auto bc=block<carbon::ElectronBirthChannel>(reader,"birth_channels");
  const auto bs=block<carbon::ElectronBirthSample>(reader,"birth_samples");
  if(bc.size()!=births.channels.size() || bs.size()!=packed_samples.size())throw std::runtime_error("Birth counts");
  for(std::size_t i=0;i<bc.size();++i){const auto&a=bc[i];const auto&b=births.channels[i];if(a.low!=b.low || a.high!=b.high || a.fraction!=b.fraction || a.offset!=b.offset || a.count!=b.count)throw std::runtime_error("Birth channel");}
  for(std::size_t i=0;i<bs.size();++i)if(bs[i].cdf!=packed_samples[i].cdf || bs[i].row!=packed_samples[i].row || bs[i].source!=packed_samples[i].source)throw std::runtime_error("Birth sample");
  reader.finish();birth_count+=packed_samples.size();
  for(const auto& ch:births.channels)for(unsigned i=0;i<100;++i) {
   const double energy=(ch.low+ch.high)*.5,u=i/100.;
   const auto a=carbon::sample_electron_birth(energy,u,births.channels.data(),births.channels.size(),births.samples.data(),births.samples.size());
   const auto b=carbon::sample_electron_birth(energy,u,births.channels.data(),births.channels.size(),packed_samples.data(),packed_samples.size());
   if(a.valid!=b.valid || a.fraction!=b.fraction || a.reference.valid!=b.reference.valid)throw std::runtime_error("birth sampling status");
   if(a.reference.valid && (a.reference.source!=b.reference.source || a.reference.row!=graphs[b.reference.source].original_rows[b.reference.row]))throw std::runtime_error("birth sampling identity");
   ++draw_count;
  }
 }
 std::cout<<"{\"tables\":"<<index.files.size()<<",\"EM_rows\":"<<checked<<",\"birth_samples\":"<<birth_count<<",\"crossing_entries\":"<<crossing_count<<",\"birth_draws\":"<<draw_count<<",\"passed\":true}\n";
}

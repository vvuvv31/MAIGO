#include "carbon/material_electron_prepared.hpp"
#include "carbon/electron_compact_builder.hpp"
#include "carbon/electron_continuation_links.hpp"
#include "carbon/electron_continuation_crossings.hpp"
#include "carbon/electron_birth_index.hpp"
#include <iostream>
#include "carbon/electron_bitmap_builder.hpp"
#include <map>
using namespace carbon;
template<class T>std::vector<T> read(MaterialPreparedReader& r,const char* key){std::vector<T> v;std::size_t n;r.read<T>(key,[&](std::size_t n){v.resize(n);return v.data();},n);return v;}
void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
bool bits(double a,double b){return std::bit_cast<std::uint64_t>(a)==std::bit_cast<std::uint64_t>(b);}
int main(int argc,char** argv){try{
 require(argc==4,"Usage: checker INDEX CACHE ID");const auto index=MaterialElectronResponseIndex::load(argv[1],compute_file_sha256_hex(argv[1]),6,12);const auto id=std::stoul(argv[3]);
 MaterialPreparedReader r(argv[2],material_prepared_key(index,id,true,true),true,true);const auto table=index.load_table(id);
 require(r.section==table.material_section && r.density==index.files[id].density_g_cm3 && r.reference_density==table.reference_density_g_cm3 && r.nodes==table.nodes.size(),"material mismatch");
 const auto channels=read<WaterElectronChannel>(r,"channels");require(channels.size()==table.channels.size(),"channel size");
 for(std::size_t i=0;i<channels.size();++i){auto a=channels[i],b=table.channels[i];require(bits(a.low,b.low)&&bits(a.high,b.high)&&bits(a.fraction,b.fraction)&&bits(a.unresolved,b.unresolved)&&a.offset==b.offset&&a.count==b.count,"channels");}
 constexpr std::size_t budget=96ULL<<30;const auto raw_sources=index.load_state_sources(id,budget);
 require(raw_sources.size()==r.sources,"source count");ElectronBirthIndex births(table.channels,6,12);
 std::vector<std::vector<std::uint64_t>> maps(raw_sources.size());std::size_t checked=0,crossing_count=0,stored_count=0;
 for(std::size_t s=0;s<raw_sources.size();++s){const auto& raw=raw_sources[s];const auto nr=raw.size()/312;
  const auto links=ElectronContinuationLinks::build(raw.data(),raw.size(),budget,1000060120);const auto original=links.view(raw.data(),raw.size());births.add(original,s);
  auto records=read<ElectronEndpointRecord>(r,"endpoint_records");auto endpoints=read<ElectronEndpoint>(r,"endpoints");auto tracks=read<ElectronCompactTrack>(r,"compact_tracks");auto media=read<ElectronCompactMedium>(r,"compact_media");
  auto next=read<std::uint32_t>(r,"next32"),offsets=read<std::uint32_t>(r,"child_offsets32"),children=read<std::uint32_t>(r,"children32");auto proof=read<double>(r,"certified_child_energy");
  require(next.size()==records.size()&&offsets.size()==records.size()+1&&proof.size()==records.size(),"compact counts");
  ElectronCompactStateView cv{records.data(),tracks.data(),media.data(),records.size(),tracks.size(),media.size(),endpoints.data(),endpoints.size()};
  cv.next32=next.data();cv.child_offsets32=offsets.data();cv.children32=children.data();
  ElectronContinuationView v{nullptr,records.size()*312,nullptr,nullptr,nullptr,children.size(),nullptr,proof.data(),&cv};
  auto& map=maps[s];map.assign(nr,kElectronContinuationEnd);std::size_t packed=0;
  for(std::size_t i=0;i<nr;++i){const auto state=original.read(i);if(state.pdg!=11&&state.pdg!=22){require(state.pdg==1000060120&&state.parent==0,"non-EM removal");continue;}
   require(packed<records.size()&&electron_state_fields_equal(state,v.read(packed)),"state bitwise mismatch");map[i]=packed++;
  }
  require(packed==records.size(),"retained record count");std::size_t childpos=0;
  for(std::size_t i=0;i<nr;++i){if(map[i]==kElectronContinuationEnd)continue;const auto j=map[i];auto oldnext=links.next[i];
   require(v.next_at(j)==(oldnext==kElectronContinuationEnd?oldnext:map.at(oldnext)),"next remap");require(offsets[j]==childpos,"child begin");
   for(auto k=links.child_offsets[i];k<links.child_offsets[i+1];++k){require(childpos<children.size()&&children[childpos]==map.at(links.child_rows[k]),"child order");++childpos;}
   require(offsets[j+1]==childpos,"child end");const auto a=prepare_electron_replay_step(original,i),b=prepare_electron_replay_step(v,j);
   require(a.status==ElectronReplayStatus::ready&&b.status==a.status&&bits(a.child_energy_MeV,proof[j])&&bits(a.energy_residual_MeV,b.energy_residual_MeV),"replay proof");
  }
  require(childpos==children.size(),"child trailing entries");
  auto edges=read<double>(r,"edges");auto logical=read<std::uint64_t>(r,"offsets"),total=read<std::uint64_t>(r,"logical_crossing_count");
  auto bins=read<ElectronBitmapBin>(r,"bitmap_bins");auto sparse=read<std::uint32_t>(r,"bitmap_sparse");auto words=read<std::uint64_t>(r,"bitmap_words");auto ranks=read<std::uint32_t>(r,"bitmap_ranks");
  const auto old=ElectronEnergyCrossings::build(original,budget);require(edges==old.edges&&logical==old.offsets&&bins.size()+1==logical.size()&&total.size()==1&&total[0]==old.rows.size(),"crossing metadata");
  ElectronBitmapCrossingsView bv{bins.data(),sparse.data(),words.data(),ranks.data()};
  for(std::size_t b=0;b<bins.size();++b) {
   const auto d=bins[b];const auto n=logical[b+1]-logical[b];std::uint64_t rank=0;
   if(!d.blocks) {
    require(d.offset<=sparse.size()&&n<=sparse.size()-d.offset,"sparse bounds");
    for(;rank<n;++rank)require(bv.select(b,rank)==map.at(old.rows[logical[b]+rank]),"sparse ordered mismatch");
   } else {
    require(d.blocks==(packed+511)/512&&d.offset<=words.size()&&std::uint64_t(d.blocks)*8<=words.size()-d.offset,"word bounds");
    require(d.rank_offset<=ranks.size()&&d.blocks+1<=ranks.size()-d.rank_offset,"rank bounds");
    for(std::uint64_t w=0;w<std::uint64_t(d.blocks)*8;++w) {
     if(w%8==0)require(ranks[d.rank_offset+w/8]==rank,"rank prefix mismatch");
     auto bits=words[d.offset+w];while(bits){auto bit=std::countr_zero(bits);require(rank<n&&w*64+bit==map.at(old.rows[logical[b]+rank]),"dense ordered mismatch");bits&=bits-1;++rank;}
    }
    require(rank==n&&ranks[d.rank_offset+d.blocks]==n,"final rank mismatch");
    for(std::uint64_t k=0;k<n;k+=257)require(bv.select(b,k)==map.at(old.rows[logical[b]+k]),"rank/select mismatch");
   }
  }
  checked+=packed;crossing_count+=old.rows.size();stored_count+=sparse.size()*4+words.size()*8+ranks.size()*4+bins.size()*sizeof(ElectronBitmapBin);
 }
 births.finish(budget);const auto bc=read<ElectronBirthChannel>(r,"birth_channels");const auto bs=read<ElectronBirthSample>(r,"birth_samples");require(bc.size()==births.channels.size()&&bs.size()==births.samples.size(),"birth size");
 for(std::size_t i=0;i<bc.size();++i){auto a=bc[i],b=births.channels[i];require(bits(a.low,b.low)&&bits(a.high,b.high)&&bits(a.fraction,b.fraction)&&a.offset==b.offset&&a.count==b.count,"birth channels");}
 for(std::size_t i=0;i<bs.size();++i){auto a=bs[i],b=births.samples[i];require(bits(a.cdf,b.cdf)&&a.source==b.source&&a.row==maps.at(b.source).at(b.row),"birth order/cdf/remap");}
 r.finish();std::cout<<"{\"table\":"<<id<<",\"rows\":"<<checked<<",\"logical_crossings\":"<<crossing_count<<",\"bitmap_bytes\":"<<stored_count<<",\"births\":"<<bs.size()<<",\"all_fields_indices_proofs_equal\":true}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

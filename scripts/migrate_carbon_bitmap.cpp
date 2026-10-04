// Explicit audited v4 -> v5 storage conversion. All common blocks are SHA
// verified; graphs and every ordered candidate are reconstructed before publish.
#include "carbon/material_electron_prepared.hpp"
#include "carbon/electron_bitmap_builder.hpp"
#include <iostream>
#include <fstream>
#include <map>
using namespace carbon;
struct Migration {
    std::filesystem::path source,dest;std::ifstream manifest;std::ofstream output;std::size_t in_block{},out_block{},total{};
    struct Block {std::string name,sha;std::size_t n,w;std::filesystem::path path;};
    Block descriptor(const char* name,std::size_t width) {
        Block b;manifest>>b.name>>b.n>>b.w>>b.sha;b.path=source/(std::to_string(in_block++)+".bin");
        if(!manifest || b.name!=name || b.w!=width || b.sha.size()!=64 || b.n>SIZE_MAX/b.w || std::filesystem::file_size(b.path)!=b.n*b.w)
            throw std::runtime_error("Invalid v4 block descriptor");
        return b;
    }
    void verify(const Block& b) {if(!file_sha256_matches(b.path,b.sha))throw std::runtime_error("v4 block SHA mismatch");}
    template<class T>std::vector<T> read(const char* name) {
        auto b=descriptor(name,sizeof(T));std::vector<T> v(b.n);std::ifstream f(b.path,std::ios::binary);
        if(!f.read(reinterpret_cast<char*>(v.data()),b.n*b.w))throw std::runtime_error("Truncated v4 block");
        MaterialPreparedDigest digest;digest.update(v.data(),b.n*b.w);if(digest.finish()!=b.sha)throw std::runtime_error("v4 SHA mismatch");return v;
    }
    std::size_t common(const char* name,std::size_t width) {
        auto b=descriptor(name,width);verify(b);auto path=dest/(std::to_string(out_block++)+".bin");
        // New directories only; immutable old blocks retain their own manifests.
        std::filesystem::create_hard_link(b.path,path);
        output<<b.name<<' '<<b.n<<' '<<b.w<<' '<<b.sha<<'\n';total+=b.n*b.w;return b.n;
    }
    template<class T>void write(const char* name,const std::vector<T>& v) {
        auto path=dest/(std::to_string(out_block++)+".bin");std::ofstream f(path,std::ios::binary);
        f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));f.close();if(!f)throw std::runtime_error("v5 write failed");
        MaterialPreparedDigest digest;digest.update(v.data(),v.size()*sizeof(T));
        output<<name<<' '<<v.size()<<' '<<sizeof(T)<<' '<<digest.finish()<<'\n';total+=v.size()*sizeof(T);
    }
};
void check(bool value,const char* text){if(!value)throw std::runtime_error(text);}
int main(int argc,char** argv) {try{
 check(argc==5,"Usage: migrate V4_DIR NEW_STAGING OLD_KEY NEW_KEY");Migration m;m.source=argv[1];m.dest=argv[2];
 std::ifstream pin(m.source/"manifest.sha256");std::string sha,extra;pin>>sha;check(bool(pin)&&sha.size()==64&&!(pin>>extra)&&file_sha256_matches(m.source/"manifest",sha),"Invalid v4 manifest SHA");
 m.manifest.open(m.source/"manifest");std::string magic,key;int section;double density,ref;std::size_t nodes,sources;
 m.manifest>>magic>>key>>section>>density>>ref>>nodes>>sources;
 check(bool(m.manifest)&&magic=="MAIGO_CARBON_EM_4"&&key==argv[3]&&sources>0&&sources<=1024,"Invalid v4 manifest identity");
 check(std::filesystem::create_directory(m.dest),"Staging exists");m.output.open(m.dest/"manifest");
 m.output<<"MAIGO_CARBON_EM_5 "<<argv[4]<<' '<<section<<' '<<std::setprecision(17)<<density<<' '<<ref<<' '<<nodes<<' '<<sources<<'\n';
 m.common("channels",sizeof(WaterElectronChannel));std::size_t checked=0,selects=0,rows_total=0;
 for(std::size_t s=0;s<sources;++s) {
    auto rows=m.common("compact_records",sizeof(ElectronCompactRecord));rows_total+=rows;check(rows>0&&rows<UINT32_MAX,"Graph rows exceed32-bit domain");
    m.common("compact_tracks",sizeof(ElectronCompactTrack));m.common("compact_media",sizeof(ElectronCompactMedium));
    {auto v=m.read<std::uint64_t>("next");check(v.size()==rows,"Next length");auto narrow=electron_graph32(v,true);
     for(std::size_t i=0;i<v.size();++i){check(v[i]==UINT64_MAX||v[i]<rows,"Next range");check(v[i]==(narrow[i]==UINT32_MAX?UINT64_MAX:std::uint64_t(narrow[i])),"Next roundtrip");}m.write("next32",narrow);}
    std::uint64_t child_count;
    {auto v=m.read<std::uint64_t>("child_offsets");check(v.size()==rows+1&&v.front()==0&&std::is_sorted(v.begin(),v.end()),"Child offsets");child_count=v.back();auto narrow=electron_graph32(v);for(std::size_t i=0;i<v.size();++i)check(v[i]==narrow[i],"Offset roundtrip");m.write("child_offsets32",narrow);}
    {auto v=m.read<std::uint64_t>("children");check(v.size()==child_count,"Child count");auto narrow=electron_graph32(v);for(std::size_t i=0;i<v.size();++i)check(v[i]==narrow[i]&&v[i]<rows,"Child roundtrip");m.write("children32",narrow);}
    check(m.common("certified_child_energy",8)==rows,"Proof size");m.common("edges",8);
    auto offsets=m.read<std::uint64_t>("offsets");m.write("offsets",offsets);auto starts=m.read<std::uint64_t>("shared_starts");auto logical=m.read<std::uint64_t>("logical_crossing_count");check(logical.size()==1&&logical[0]==offsets.back(),"Logical count");m.write("logical_crossing_count",logical);
    auto old=m.read<std::uint32_t>("crossings32");auto bitmap=ElectronBitmapBuilder::build(rows,offsets,starts,old);auto view=bitmap.view();
    std::map<std::pair<std::uint64_t,std::uint64_t>,bool> seen;
    for(std::size_t b=0;b<starts.size();++b) {
        auto n=offsets[b+1]-offsets[b],start=starts[b];if(!seen.emplace(std::make_pair(start,n),true).second)continue;
        const auto d=bitmap.bins[b];std::size_t rank=0;
        if(!d.blocks) {for(;rank<n;++rank)check(view.select(b,rank)==old[start+rank],"Sparse select mismatch");selects+=n;}
        else {
            for(std::size_t w=0;w<std::size_t(d.blocks)*8;++w) {
                if(w%8==0)check(bitmap.ranks[d.rank_offset+w/8]==rank,"Rank checkpoint mismatch");
                auto bits=bitmap.words[d.offset+w];while(bits){auto bit=std::countr_zero(bits);check(rank<n&&w*64+bit==old[start+rank],"Bitmap ordered reconstruction mismatch");bits&=bits-1;++rank;}
            }
            check(rank==n&&bitmap.ranks[d.rank_offset+d.blocks]==n,"Bitmap final rank");
            for(std::size_t block=0;block<d.blocks;++block) {
                auto lo=bitmap.ranks[d.rank_offset+block],hi=bitmap.ranks[d.rank_offset+block+1];
                if(lo<hi){check(view.select(b,lo)==old[start+lo]&&view.select(b,hi-1)==old[start+hi-1],"Select checkpoint boundary");selects+=2;}
            }
            for(std::uint64_t k=0;k<n;k+=257){check(view.select(b,k)==old[start+k],"Select strided rank");++selects;}
        }
        checked+=n;
    }
    m.write("bitmap_bins",bitmap.bins);m.write("bitmap_sparse",bitmap.sparse);m.write("bitmap_words",bitmap.words);m.write("bitmap_ranks",bitmap.ranks);
 }
 m.common("birth_channels",sizeof(ElectronBirthChannel));m.common("birth_samples",sizeof(ElectronBirthSample));check(!(m.manifest>>extra),"Trailing v4 descriptors");
 m.output.close();check(bool(m.output),"Manifest write failed");std::ofstream outpin(m.dest/"manifest.sha256");outpin<<compute_file_sha256_hex(m.dest/"manifest")<<'\n';outpin.close();check(bool(outpin),"Pin write failed");
 std::cout<<"{\"rows\":"<<rows_total<<",\"candidates_reconstructed\":"<<checked<<",\"select_checks\":"<<selects<<",\"array_bytes\":"<<m.total<<"}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

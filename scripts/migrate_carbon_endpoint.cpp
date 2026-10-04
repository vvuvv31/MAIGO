// Explicit audited v5 -> v5 storage conversion. All common blocks are SHA
// verified; graphs and every ordered candidate are reconstructed before publish.
#include "carbon/material_electron_prepared.hpp"
#include "carbon/electron_endpoint_builder.hpp"
#include <iostream>
#include <fstream>
#include <map>
#include "carbon/electron_bitmap_crossings.hpp"
using namespace carbon;
struct Migration {
    std::filesystem::path source,dest;std::ifstream manifest;std::ofstream output;std::size_t in_block{},out_block{},total{};
    struct Block {std::string name,sha;std::size_t n,w;std::filesystem::path path;};
    Block descriptor(const char* name,std::size_t width) {
        Block b;manifest>>b.name>>b.n>>b.w>>b.sha;b.path=source/(std::to_string(in_block++)+".bin");
        if(!manifest || b.name!=name || b.w!=width || b.sha.size()!=64 || b.n>SIZE_MAX/b.w || std::filesystem::file_size(b.path)!=b.n*b.w)
            throw std::runtime_error("Invalid v5 block descriptor");
        return b;
    }
    void verify(const Block& b) {if(!file_sha256_matches(b.path,b.sha))throw std::runtime_error("v5 block SHA mismatch");}
    template<class T>std::vector<T> read(const char* name) {
        auto b=descriptor(name,sizeof(T));std::vector<T> v(b.n);std::ifstream f(b.path,std::ios::binary);
        if(!f.read(reinterpret_cast<char*>(v.data()),b.n*b.w))throw std::runtime_error("Truncated v5 block");
        MaterialPreparedDigest digest;digest.update(v.data(),b.n*b.w);if(digest.finish()!=b.sha)throw std::runtime_error("v5 SHA mismatch");return v;
    }
    std::size_t common(const char* name,std::size_t width) {
        auto b=descriptor(name,width);verify(b);auto path=dest/(std::to_string(out_block++)+".bin");
        // New directories only; immutable old blocks retain their own manifests.
        std::filesystem::create_hard_link(b.path,path);
        output<<b.name<<' '<<b.n<<' '<<b.w<<' '<<b.sha<<'\n';total+=b.n*b.w;return b.n;
    }
    template<class T>void write(const char* name,const std::vector<T>& v) {
        auto path=dest/(std::to_string(out_block++)+".bin");std::ofstream f(path,std::ios::binary);
        f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));f.close();if(!f)throw std::runtime_error("v6 write failed");
        MaterialPreparedDigest digest;digest.update(v.data(),v.size()*sizeof(T));
        output<<name<<' '<<v.size()<<' '<<sizeof(T)<<' '<<digest.finish()<<'\n';total+=v.size()*sizeof(T);
    }
};
void check(bool value,const char* text){if(!value)throw std::runtime_error(text);}
int main(int argc,char** argv) {try{
 check(argc==5,"Usage: migrate V5_DIR NEW_STAGING OLD_KEY NEW_KEY");Migration m;m.source=argv[1];m.dest=argv[2];
 std::ifstream pin(m.source/"manifest.sha256");std::string sha,extra;pin>>sha;check(bool(pin)&&sha.size()==64&&!(pin>>extra)&&file_sha256_matches(m.source/"manifest",sha),"Invalid v5 manifest SHA");
 m.manifest.open(m.source/"manifest");std::string magic,key;int section;double density,ref;std::size_t nodes,sources;
 m.manifest>>magic>>key>>section>>density>>ref>>nodes>>sources;
 check(bool(m.manifest)&&magic=="MAIGO_CARBON_EM_5"&&key==argv[3]&&sources>0&&sources<=1024,"Invalid v5 manifest identity");
 check(std::filesystem::create_directory(m.dest),"Staging exists");m.output.open(m.dest/"manifest");
 m.output<<"MAIGO_CARBON_EM_7 "<<argv[4]<<' '<<section<<' '<<std::setprecision(17)<<density<<' '<<ref<<' '<<nodes<<' '<<sources<<'\n';
 m.common("channels",sizeof(WaterElectronChannel));std::size_t rows_total=0,shared=0,endpoint_count=0;
 for(std::size_t s=0;s<sources;++s) {
    auto old=m.read<ElectronCompactRecord>("compact_records");auto tracks=m.read<ElectronCompactTrack>("compact_tracks");auto media=m.read<ElectronCompactMedium>("compact_media");
    auto next=m.read<std::uint32_t>("next32");const auto rows=old.size();rows_total+=rows;
    auto packed=ElectronEndpointBuilder::build(old,next);shared+=packed.shared;endpoint_count+=packed.endpoints.size();
    ElectronLegacyCompactStateView original{old.data(),tracks.data(),media.data(),rows,tracks.size(),media.size()};
    ElectronCompactStateView compact{packed.records.data(),tracks.data(),media.data(),rows,tracks.size(),media.size(),packed.endpoints.data(),packed.endpoints.size()};
    for(std::size_t row=0;row<rows;++row) {
        check(original.read(row).valid,"Invalid source fields");
        check(electron_state_fields_equal(original.read(row),compact.read(row)),"Endpoint full-field bitwise roundtrip");
    }
    m.write("endpoint_records",packed.records);m.write("endpoints",packed.endpoints);
    m.write("compact_tracks",tracks);m.write("compact_media",media);m.write("next32",next);
    check(m.common("child_offsets32",4)==rows+1,"Child offsets size");m.common("children32",4);
    check(m.common("certified_child_energy",8)==rows,"Proof size");m.common("edges",8);m.common("offsets",8);m.common("logical_crossing_count",8);
    m.common("bitmap_bins",sizeof(ElectronBitmapBin));m.common("bitmap_sparse",4);m.common("bitmap_words",8);m.common("bitmap_ranks",4);
 }
 m.common("birth_channels",sizeof(ElectronBirthChannel));m.common("birth_samples",sizeof(ElectronBirthSample));check(!(m.manifest>>extra),"Trailing v5 descriptors");
 m.output.close();check(bool(m.output),"Manifest write failed");std::ofstream outpin(m.dest/"manifest.sha256");outpin<<compute_file_sha256_hex(m.dest/"manifest")<<'\n';outpin.close();check(bool(outpin),"Pin write failed");
 std::cout<<"{\"rows\":"<<rows_total<<",\"shared\":"<<shared<<",\"endpoints\":"<<endpoint_count<<",\"array_bytes\":"<<m.total<<",\"all_fields_bitwise_equal\":true}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

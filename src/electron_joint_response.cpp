#include "carbon/electron_joint_response.hpp"
#include "carbon/min_json.hpp"
#include "carbon/sha256.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace carbon {
ElectronJointResponseTable ElectronJointResponseTable::from_csv(const std::filesystem::path& path,
    const std::string& data_pin,const std::string& meta_pin) {
    auto meta=path;meta.replace_extension(".metadata.json");
    if(data_pin.size()!=64 || meta_pin.size()!=64 || compute_file_sha256_hex(path)!=data_pin ||
       compute_file_sha256_hex(meta)!=meta_pin)throw std::invalid_argument("Electron joint response SHA mismatch/missing pin");
    std::ifstream mf(meta);std::stringstream text;text<<mf.rdbuf();
    const auto m=minjson::Parser(text.str()).parse();
    auto number=[](const minjson::Value& v) {if(v.type!=minjson::Value::Type::Number || !std::isfinite(v.number))throw std::invalid_argument("Joint metadata number");return v.number;};
    const double schema=number(m.at("schema_version"));
    if((schema!=1 && schema!=2) || m.at("status").str!="UNVALIDATED_INTERFACE_DIAGNOSTIC" ||
       number(m.at("energy_bin_width_MeVu"))!=5 || number(m.at("energy_max_MeVu"))!=185 ||
       m.at("data_sha256").str!=data_pin || m.at("data_filename").str!=path.filename().string() ||
       number(m.at("data_size_bytes"))!=static_cast<double>(std::filesystem::file_size(path)) ||
       m.at("charged_escape_allowed").type!=minjson::Value::Type::Boolean || m.at("charged_escape_allowed").boolean ||
       number(m.at("fixed_parent_inset_mm").at("air"))!=100 || number(m.at("fixed_parent_inset_mm").at("tissue"))!=5 ||
       m.at("sources").type!=minjson::Value::Type::Array || m.at("sources").arr.empty())
        throw std::invalid_argument("Unsupported electron joint metadata");
    if(m.at("projectile").arr.size()!=2 || number(m.at("projectile").at(0))!=6 || number(m.at("projectile").at(1))!=12)
        throw std::invalid_argument("Joint response requires C12");
    ElectronJointResponseTable out;out.minimum.fill(std::numeric_limits<double>::infinity());
    std::array<bool,74> declared{};std::array<double,74> fractions{};
    for(const auto& v:m.at("channels").arr) {
        const double sec=number(v.at("section_id")),bin=number(v.at("energy_bin"));
        if((sec!=0 && sec!=8) || bin<0 || bin>=37 || bin!=std::floor(bin))throw std::invalid_argument("Joint channel key");
        const int material=sec==8,index=material*37+static_cast<int>(bin);
        if(declared[index])throw std::invalid_argument("Duplicate joint channel");declared[index]=true;
        const double low=number(v.at("energy_min_MeVu")),high=number(v.at("energy_max_MeVu"));
        if(low<bin*5 || high>(bin+1)*5 || low>high)throw std::invalid_argument("Joint energy interval");
        out.minimum[material]=std::min(out.minimum[material],low);out.maximum[material]=std::max(out.maximum[material],high);
        fractions[index]=number(v.at("nonlocal_fraction"));
        if(fractions[index]<0 || fractions[index]>1)throw std::invalid_argument("Joint fraction");
        const double terminal=number(v.at("terminal_local_excess_MeV"));
        const double residual=number(v.at("raw_kinetic_closure_residual_MeV"));
        const double loss=number(v.at("parent_loss_MeV"));
        if(loss<=0 || std::abs(residual-terminal)>1e-7*loss ||
           (terminal!=0 && (bin!=0 || fractions[index]!=0 || !v.at("redistribution_inactive").boolean)))
            throw std::invalid_argument("Unexplained joint energy residual");
    }
    std::ifstream input(path);std::string line;
    if(!std::getline(input,line) || line!="section_id,energy_bin,nonlocal_fraction,cdf,longitudinal_mass_g_cm2,radial_mass_g_cm2")
        throw std::invalid_argument("Joint CSV header");
    int previous=-1;
    while(std::getline(input,line)) {
        std::stringstream row(line);std::string field;std::array<double,6> values{};int n=0;
        while(std::getline(row,field,',')) {if(n==6)throw std::invalid_argument("Joint CSV width");std::size_t used=0;
            values[n]=std::stod(field,&used);if(used!=field.size() || !std::isfinite(values[n++]))throw std::invalid_argument("Joint CSV value");}
        if(n!=6 || (values[0]!=0 && values[0]!=8) || values[1]<0 || values[1]>=37 || values[1]!=std::floor(values[1]))
            throw std::invalid_argument("Joint CSV channel");
        const int index=(values[0]==8)*37+static_cast<int>(values[1]);auto& ch=out.channels[index];
        if(index<previous || !declared[index] || std::abs(values[2]-fractions[index])>1e-12 ||
           values[3]<=0 || values[3]>1 || values[5]<0)throw std::invalid_argument("Joint CSV probability");
        if(ch.count==0) {ch.offset=static_cast<std::uint32_t>(out.samples.size());ch.fraction=values[2];}
        else if(values[3]<out.samples.back().cdf)throw std::invalid_argument("Joint CDF order");
        out.samples.push_back({values[3],values[4],values[5]});++ch.count;previous=index;
        if(out.samples.size()>2000000)throw std::invalid_argument("Joint payload exceeds diagnostic limit");
    }
    for(int i=0;i<74;++i)if(declared[i]) {
        const auto ch=out.channels[i];
        if(!ch.count || out.samples[ch.offset+ch.count-1].cdf!=1)throw std::invalid_argument("Incomplete joint channel");
    }
    if(!std::isfinite(out.minimum[0]) || !std::isfinite(out.minimum[1]))throw std::invalid_argument("Missing joint material");
    if(schema==1 && m.contains("ordered_path_file"))throw std::invalid_argument("Ordered paths require explicit schema 2");
    if(schema==2) {
        const auto filename=m.at("ordered_path_file").str;
        if(filename.empty() || std::filesystem::path(filename).filename().string()!=filename)
            throw std::invalid_argument("Ordered path must be a companion filename");
        const auto binary=path.parent_path()/filename;
        out.path_sha256=m.at("ordered_path_sha256").str;
        if(out.path_sha256.size()!=64 || compute_file_sha256_hex(binary)!=out.path_sha256 ||
           number(m.at("ordered_path_size_bytes"))!=static_cast<double>(std::filesystem::file_size(binary)))
            throw std::invalid_argument("Ordered path SHA/size mismatch");
        std::ifstream f(binary,std::ios::binary);char magic[8];std::uint32_t version=0,ns=0,nv=0;
        f.read(magic,8);f.read(reinterpret_cast<char*>(&version),4);f.read(reinterpret_cast<char*>(&ns),4);f.read(reinterpret_cast<char*>(&nv),4);
        if(!f || std::memcmp(magic,"ELPATH01",8)!=0 || version!=1 || ns!=out.samples.size() || !nv || nv>20000000 ||
           std::filesystem::file_size(binary)!=20ULL+8ULL*ns+24ULL*nv)
            throw std::invalid_argument("Ordered path header/count mismatch");
        static_assert(sizeof(ElectronPathRange)==8 && sizeof(std::array<double,3>)==24);
        out.path_ranges.resize(ns);out.path_vectors.resize(nv);
        f.read(reinterpret_cast<char*>(out.path_ranges.data()),8ULL*ns);
        f.read(reinterpret_cast<char*>(out.path_vectors.data()),24ULL*nv);
        if(!f)throw std::invalid_argument("Truncated ordered paths");
        std::size_t next=0;
        for(std::size_t i=0;i<ns;++i) {
            const auto r=out.path_ranges[i];std::array<double,3> net{};
            if(r.offset!=next || !r.count || r.count>4096 || std::uint64_t(r.offset)+r.count>nv)
                throw std::invalid_argument("Ordered path range");
            for(std::size_t j=r.offset;j<r.offset+r.count;++j)for(int a=0;a<3;++a) {
                const double v=out.path_vectors[j][a];
                if(!std::isfinite(v))throw std::invalid_argument("Nonfinite ordered vector");net[a]+=v;
            }
            const auto s=out.samples[i];
            if(std::abs(net[0]-s.radial_mass_g_cm2)>1e-9 || std::abs(net[1])>1e-9 || std::abs(net[2]-s.longitudinal_mass_g_cm2)>1e-9)
                throw std::invalid_argument("Ordered endpoint/CSV mismatch");
            next+=r.count;
        }
        if(next!=nv)throw std::invalid_argument("Unreferenced ordered vectors");
    }
    return out;
}
} // namespace carbon

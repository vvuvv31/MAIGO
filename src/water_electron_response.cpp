#include "carbon/water_electron_response.hpp"
#include "carbon/material_electron_response.hpp"
#include "carbon/electron_state_refs.hpp"
#include <algorithm>
#include "carbon/min_json.hpp"
#include "carbon/sha256.hpp"
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace carbon {
WaterElectronResponseTable WaterElectronResponseTable::load(const std::filesystem::path& path,
    const std::string& data_pin,const std::string& metadata_pin,bool material_response) {
    auto metadata=path;metadata.replace_extension(".metadata.json");
    auto require=[](bool ok,const char* why) {if(!ok)throw std::invalid_argument(why);};
    require(data_pin.size()==64 && metadata_pin.size()==64 && file_sha256_matches(path,data_pin) &&
        file_sha256_matches(metadata,metadata_pin),"Water electron data/metadata SHA mismatch");
    std::ifstream mf(metadata);std::stringstream text;text<<mf.rdbuf();
    const auto m=minjson::Parser(text.str()).parse();
    auto num=[&](const minjson::Value& v) {require(v.type==minjson::Value::Type::Number && std::isfinite(v.number),"Water response metadata number");return v.number;};
    const auto schema=num(m.at("schema_version"));
    const auto nominal_max=num(m.at("nominal_energy_max_MeVu"));
    const bool material_schema=material_response && schema==4 && nominal_max==450 &&
        m.at("status").str=="UNVALIDATED_MATERIAL_EM_DIAGNOSTIC";
    const bool water_schema=!material_response && ((schema==2 && nominal_max==300) || (schema==3 && nominal_max==450)) &&
        m.at("status").str=="UNVALIDATED_WATER_EM_DIAGNOSTIC" && m.at("material").str=="Water_75eV" && num(m.at("density_g_cm3"))==1;
    require((material_schema || water_schema) && num(m.at("density_g_cm3"))>0 && num(m.at("production_cut_mm"))==.05 &&
        num(m.at("actual_energy_max_MeVu"))>=nominal_max &&
        num(m.at("actual_energy_max_MeVu"))<=nominal_max+.0001 && m.at("projectile").arr.size()==2 &&
        num(m.at("projectile").at(0))==6 && num(m.at("projectile").at(1))==12 &&
        m.at("data_filename").str==path.filename().string() && m.at("data_sha256").str==data_pin &&
        num(m.at("data_size_bytes"))==std::filesystem::file_size(path),"Wrong water response scope/identity");
    static_assert(sizeof(WaterElectronChannel)==40 && sizeof(WaterElectronSample)==56 && sizeof(WaterElectronPathNode)==32);
    std::ifstream f(path,std::ios::binary);char magic[8];std::uint32_t version=0,nc=0,ns=0;
    f.read(magic,8);f.read(reinterpret_cast<char*>(&version),4);f.read(reinterpret_cast<char*>(&nc),4);f.read(reinterpret_cast<char*>(&ns),4);
    require(f && std::memcmp(magic,material_schema ? "MELSEG01" : "WELSEG01",8)==0 && version==((schema==2 || material_schema) ? 1U : 2U) && nc==nominal_max/5 && ns>0 && ns<=10000000 &&
        std::filesystem::file_size(path)==20ULL+40ULL*nc+56ULL*ns && m.at("channels").arr.size()==nc,"Water response binary header");
    WaterElectronResponseTable out;out.channels.resize(nc);out.samples.resize(ns);
    out.reference_density_g_cm3=num(m.at("density_g_cm3"));out.material_name=m.at("material").str;
    if(material_schema) {
        const auto& id=m.at("material_identity");
        require(id.at("material_name").str==out.material_name && num(id.at("runtime_density_g_cm3"))==out.reference_density_g_cm3,
                "Material response identity mismatch");
        if(id.at("kind").str=="schneider") {
            const auto& truth=id.at("schneider_identity");
            const double section=num(truth.at("material_section"));
            const double hu=num(id.at("hu"));
            require(section>=0 && section<25 && section==std::floor(section) && hu==std::floor(hu) &&
                    hu==num(truth.at("hu")) && hu>=-1000 && hu<=2995,"Material response Schneider key");
            const auto expected=std::string("PatientTissueFromHU")+(hu<0 ? "Negative" : "")+std::to_string(static_cast<int>(std::abs(hu)));
            require(expected==out.material_name,"Material response Schneider name");
            out.material_section=static_cast<int>(section);
        } else require(id.at("kind").str=="water_density" && out.material_name=="ResponseWater75",
                       "Unknown material response kind");
    }
    f.read(reinterpret_cast<char*>(out.channels.data()),40ULL*nc);
    f.read(reinterpret_cast<char*>(out.samples.data()),56ULL*ns);require(bool(f),"Truncated water samples");
    std::uint64_t next=0;
    for(unsigned b=0;b<nc;++b) {
        const auto c=out.channels[b];const auto& v=m.at("channels").at(b);
        const double loss=num(v.at("parent_loss_MeV")),local=num(v.at("raw_local_MeV")),birth=num(v.at("birth_MeV"));
        const double dep=num(v.at("deposited_MeV")),esc=num(v.at("photon_escape_MeV"));
        require(loss>0 && local>=0 && birth>=0 && dep>=0 && esc>=0 &&
            std::abs(local+birth-loss-num(v.at("terminal_local_excess_MeV")))<1e-10*loss &&
            std::abs(local+birth-loss)<(b==0 ? 1e-3 : 1e-7)*loss && std::abs(birth-dep-esc)<1e-7*loss,
            "Water electron kinetic partition");
        require(num(v.at("energy_bin"))==b && c.low==5*b && c.high==(b==nc-1 ? num(m.at("actual_energy_max_MeVu")) : 5*(b+1)) &&
            c.low==num(v.at("energy_min_MeVu")) && c.high==num(v.at("energy_max_MeVu")) &&
            c.fraction==num(v.at("deposited_fraction")) && std::abs(c.fraction-dep/loss)<1e-12 &&
            c.unresolved==esc/loss && c.fraction>=0 && c.unresolved>=0 && c.fraction+c.unresolved<=1 &&
            c.offset==next && c.offset==num(v.at("offset")) && c.count==num(v.at("count")) && next+c.count<=ns,
            "Water electron channel mismatch");
        double previous=0;
        for(std::size_t j=c.offset;j<c.offset+c.count;++j) {
            const auto s=out.samples[j];require(s.cdf>previous && s.cdf<=1,"Water electron CDF");previous=s.cdf;
            for(int a=0;a<3;++a)require(std::isfinite(s.pre[a]) && std::isfinite(s.post[a]),"Nonfinite water sample");
        }
        require(c.count ? previous==1 && c.fraction>0 : c.fraction==0,"Missing water channel payload");next+=c.count;
    }
    require(next==ns,"Unreferenced water samples");
    const auto filename=m.at("path_filename").str;const auto companion=path.parent_path()/filename;
    require(!filename.empty() && std::filesystem::path(filename).filename().string()==filename &&
        file_sha256_matches(companion,m.at("path_sha256").str) &&
        std::filesystem::file_size(companion)==num(m.at("path_size_bytes")),"Water path graph SHA/size");
    std::ifstream p(companion,std::ios::binary);std::uint32_t count=0,nodes=0;
    p.read(magic,8);p.read(reinterpret_cast<char*>(&version),4);p.read(reinterpret_cast<char*>(&count),4);p.read(reinterpret_cast<char*>(&nodes),4);
    require(p && std::memcmp(magic,"WELPTH01",8)==0 && version==1 && count==ns && nodes>0 && nodes<=20000000 &&
        nodes==num(m.at("path_nodes")) && std::filesystem::file_size(companion)==20ULL+4ULL*ns+32ULL*nodes,"Water path graph header");
    out.heads.resize(ns);out.nodes.resize(nodes);
    p.read(reinterpret_cast<char*>(out.heads.data()),4ULL*ns);p.read(reinterpret_cast<char*>(out.nodes.data()),32ULL*nodes);
    require(bool(p),"Truncated water path graph");
    out.prefix_radius.resize(nodes);
    for(std::size_t i=0;i<nodes;++i) {
        const auto n=out.nodes[i];require(n.reserved==0 && (n.previous==kWaterPathNone || n.previous<i),"Cyclic water path graph");
        for(double x:n.point)require(std::isfinite(x) && (n.previous!=kWaterPathNone || x==0),"Invalid water path origin");
        double r2=0;for(double x:n.point)r2+=x*x;
        const double previous=n.previous==kWaterPathNone ? 0 : out.prefix_radius[n.previous];
        out.prefix_radius[i]=std::max(std::sqrt(r2),previous);
    }
    for(std::size_t i=0;i<ns;++i) {
        require(out.heads[i]<nodes,"Invalid water sample path link");
        const auto s=out.samples[i];
        if(s.pre!=s.post)for(int a=0;a<3;++a)
            require(std::abs(out.nodes[out.heads[i]].point[a]-s.pre[a])<=1e-6,"Water sample/path frame mismatch");
    }
    return out;
}
MaterialElectronResponseIndex MaterialElectronResponseIndex::load(const std::filesystem::path& path,
    const std::string& pin) {
    if(pin.size()!=64 || !file_sha256_matches(path,pin))throw std::invalid_argument("Material response index SHA mismatch");
    std::ifstream input(path);std::stringstream text;text<<input.rdbuf();
    const auto m=minjson::Parser(text.str()).parse();
    if(m.at("schema_version").number!=1 || m.at("status").str!="UNVALIDATED_MATERIAL_RESPONSE_BANK" ||
       m.at("tables").type!=minjson::Value::Type::Array || m.at("tables").arr.empty())
        throw std::invalid_argument("Unsupported material response index");
    MaterialElectronResponseIndex out;
    for(const auto& t:m.at("tables").arr) {
        if(t.at("status").str!="COMPILED")throw std::invalid_argument("Incomplete material response index");
        const auto& id=t.at("material_identity");
        MaterialElectronTableFile file;
        const auto& density=id.at("runtime_density_g_cm3");
        if(density.type!=minjson::Value::Type::Number || !std::isfinite(density.number) || density.number<=0)
            throw std::invalid_argument("Invalid response reference density");
        file.density_g_cm3=density.number;
        file.reference_density_g_cm3=density.number;
        if(id.at("kind").str=="schneider") {
            const auto& sec=id.at("schneider_identity").at("material_section");
            if(sec.type!=minjson::Value::Type::Number || !std::isfinite(sec.number) || sec.number<0 || sec.number>=25 || sec.number!=std::floor(sec.number))
                throw std::invalid_argument("Invalid response section");
            file.section=static_cast<int>(sec.number);
            const auto& formula=id.at("schneider_identity").at("density_g_cm3");
            if(formula.type!=minjson::Value::Type::Number || !std::isfinite(formula.number) || formula.number<=0)
                throw std::invalid_argument("Invalid Schneider response formula density");
            file.density_g_cm3=static_cast<double>(static_cast<float>(formula.number));
            if(std::abs(file.reference_density_g_cm3-formula.number)>5.1e-6*formula.number)
                throw std::invalid_argument("Response density exceeds TOPAS serialization precision");
        } else if(id.at("kind").str!="water_density")throw std::invalid_argument("Unknown response material kind");
        file.path=t.at("path").str;
        if(file.path.empty())throw std::invalid_argument("Empty material response path");
        if(file.path.is_relative())file.path=path.parent_path()/file.path;
        file.data_sha256=t.at("data_sha256").str;file.metadata_sha256=t.at("metadata_sha256").str;
        if(file.data_sha256.size()!=64 || file.metadata_sha256.size()!=64)throw std::invalid_argument("Missing response table pins");
        if(t.contains("state_reference_file") || t.contains("state_reference_sha256") || t.contains("state_reference_metadata_sha256")) {
            file.state_reference_file=t.at("state_reference_file").str;
            file.state_reference_sha256=t.at("state_reference_sha256").str;
            file.state_reference_metadata_sha256=t.at("state_reference_metadata_sha256").str;
            if(file.state_reference_file.empty() || file.state_reference_sha256.size()!=64 || file.state_reference_metadata_sha256.size()!=64)
                throw std::invalid_argument("Incomplete material state-reference pins");
            if(file.state_reference_file.is_relative())file.state_reference_file=path.parent_path()/file.state_reference_file;
        }
        out.files.push_back(std::move(file));
    }
    std::sort(out.files.begin(),out.files.end(),[](const auto& a,const auto& b) {
        return a.section<b.section || (a.section==b.section && a.density_g_cm3<b.density_g_cm3);
    });
    for(std::size_t i=1;i<out.files.size();++i)
        if(out.files[i].section==out.files[i-1].section && out.files[i].density_g_cm3==out.files[i-1].density_g_cm3)
            throw std::invalid_argument("Duplicate material density response key");
    return out;
}
std::vector<std::size_t> MaterialElectronResponseIndex::required_tables(
    const std::vector<std::pair<int,double>>& demand) const {
    std::vector<bool> needed(files.size(),false);
    for(const auto& point:demand) {
        const auto section=point.first;const double density=point.second;
        if(section< -1 || section>=25 || !std::isfinite(density) || density<=0)
            throw std::invalid_argument("Invalid material response geometry demand");
        auto first=std::lower_bound(files.begin(),files.end(),section,
            [](const auto& f,int s){return f.section<s;});
        auto last=std::upper_bound(first,files.end(),section,
            [](int s,const auto& f){return s<f.section;});
        if(first==last)throw std::invalid_argument("Geometry material absent from electron response bank");
        auto high=std::lower_bound(first,last,density,
            [](const auto& f,double rho){return f.density_g_cm3<rho;});
        if(high==last || (high==first && high->density_g_cm3!=density))
            throw std::invalid_argument("Geometry density outside electron response coverage");
        const auto index=static_cast<std::size_t>(high-files.begin());needed[index]=true;
        if(high->density_g_cm3!=density)needed[index-1]=true;
    }
    std::vector<std::size_t> selected;
    for(std::size_t i=0;i<needed.size();++i)if(needed[i])selected.push_back(i);
    return selected;
}
WaterElectronResponseTable MaterialElectronResponseIndex::load_table(std::size_t i) const {
    const auto& file=files.at(i);
    auto table=WaterElectronResponseTable::load(file.path,file.data_sha256,file.metadata_sha256,true);
    if(table.material_section!=file.section || table.reference_density_g_cm3!=file.reference_density_g_cm3)
        throw std::invalid_argument("Material response index/table identity mismatch");
    return table;
}
std::vector<unsigned char> MaterialElectronResponseIndex::load_state_references(std::size_t i,std::size_t byte_budget) const {
    const auto& file=files.at(i);
    auto metadata=file.state_reference_file;metadata.replace_extension(".metadata.json");
    if(file.state_reference_file.empty() || !file_sha256_matches(metadata,file.state_reference_metadata_sha256) ||
       !file_sha256_matches(file.state_reference_file,file.state_reference_sha256))
        throw std::invalid_argument("Missing or changed material state references");
    std::ifstream input(metadata);std::stringstream text;text<<input.rdbuf();
    const auto m=minjson::Parser(text.str()).parse();
    const auto bytes=std::filesystem::file_size(file.state_reference_file);
    if(bytes>byte_budget || m.at("schema_version").number!=1 ||
       m.at("status").str!="STATE_REFERENCE_MAP_NOT_BOUNDARY_MODEL" ||
       m.at("response_metadata_sha256").str!=file.metadata_sha256 ||
       m.at("data_sha256").str!=file.state_reference_sha256 ||
       m.at("data_filename").str!=file.state_reference_file.filename().string() ||
       m.at("data_size_bytes").number!=bytes)
        throw std::invalid_argument("Material state-reference binding/size mismatch");
    std::vector<unsigned char> payload(static_cast<std::size_t>(bytes));
    std::ifstream raw(file.state_reference_file,std::ios::binary);
    raw.read(reinterpret_cast<char*>(payload.data()),payload.size());
    const auto view=ElectronStateReferenceView::parse(payload.data(),payload.size());
    if(!raw || !view.valid || view.samples!=m.at("samples").number || view.nodes!=m.at("nodes").number ||
       m.at("sources").type!=minjson::Value::Type::Array || view.sources!=m.at("sources").arr.size())
        throw std::invalid_argument("Invalid material state-reference layout");
    for(bool nodes:{false,true})for(std::uint64_t n=0;n<(nodes?view.nodes:view.samples);++n)
        if(!view.at(n,nodes).valid)throw std::invalid_argument("Invalid material state-reference row");
    return payload;
}
std::vector<std::vector<unsigned char>> MaterialElectronResponseIndex::load_state_sources(std::size_t i,std::size_t byte_budget) const {
    const auto& file=files.at(i);
    auto metadata=file.state_reference_file;metadata.replace_extension(".metadata.json");
    if(file.state_reference_file.empty() || !file_sha256_matches(metadata,file.state_reference_metadata_sha256))
        throw std::invalid_argument("Missing or changed state-source manifest");
    std::ifstream input(metadata);std::stringstream text;text<<input.rdbuf();
    const auto m=minjson::Parser(text.str()).parse();
    if(m.at("response_metadata_sha256").str!=file.metadata_sha256 || m.at("sources").type!=minjson::Value::Type::Array)
        throw std::invalid_argument("State-source response binding mismatch");
    std::vector<std::vector<unsigned char>> result;
    for(const auto& source:m.at("sources").arr) {
        std::filesystem::path path=source.at("path").str;
        if(path.is_relative())path=metadata.parent_path()/path;
        const auto bytes=std::filesystem::file_size(path);
        const auto rows=source.at("rows").number;
        if(!bytes || bytes>byte_budget || bytes%kElectronStateV4RecordBytes ||
           rows!=bytes/kElectronStateV4RecordBytes ||
           !file_sha256_matches(path,source.at("sha256").str))
            throw std::invalid_argument("State source exceeds budget or has wrong SHA/record count");
        std::vector<unsigned char> payload(static_cast<std::size_t>(bytes));
        std::ifstream raw(path,std::ios::binary);
        raw.read(reinterpret_cast<char*>(payload.data()),payload.size());
        if(!raw)throw std::invalid_argument("Truncated state source");
        byte_budget-=payload.size();result.push_back(std::move(payload));
    }
    if(result.empty())throw std::invalid_argument("Empty state-source list");
    return result;
}
} // namespace carbon

#pragma once
#include "carbon/electron_material_registry.hpp"
#include "carbon/electron_fp32.hpp"
#include "carbon/electron_cut_birth.hpp"
#include "carbon/electron_composition26_data.hpp"
#include "carbon/electron_projectile26_data.hpp"
#include <sstream>
#include <iomanip>
#include "carbon/electron_birth_bank_device.hpp"
#include "carbon/electron_charged_bank_device.hpp"
#include "carbon/electron_joint_pilot_device.hpp"
#include "carbon/photon_response_bank_device.hpp"
#include <cstdlib>
#ifdef CARBON_ELECTRON_CERTIFIED_RATES
#include "carbon/electron_rate_bound.hpp"
#include <iostream>
#endif

namespace carbon {
inline ElectronCutBirthState sample_cut_electron_birth_seeded(
    const ElectronCutBirthComposition* compositions,unsigned section,unsigned a,
    float energy,float cut,std::uint64_t seed) {
    if(!compositions||section>=kElectronMaterialCount)return {};
    auto uniform=[&](){return electron_fp32::random_draw(seed);};
    return sample_cut_electron_birth(compositions[section],energy,a,cut,uniform);
}
inline ElectronCutBirthState sample_cut_electron_birth_seeded(
    const ElectronCutBirthComposition* compositions,unsigned section,ElectronProjectileBirthParameters parameters,
    float energy,float cut,std::uint64_t seed) {
    if(!compositions||section>=kElectronMaterialCount)return {};
    auto uniform=[&](){return electron_fp32::random_draw(seed);};
    return sample_cut_electron_birth(compositions[section],energy,parameters,cut,uniform);
}
namespace electron_fp32_convert {
// These conversions run once at loading, never in the transport kernel. File
// parsing and physical schema validation remain with the disk readers.
inline electron_fp32::Probability probability(double x){
    return {x<=0?0:x>=1?UINT64_MAX:static_cast<std::uint64_t>(std::ldexp(x,64))};
}
inline float value(double x){return float(x);}
inline electron_fp32::joint_math::V value(electron_joint_research::V v){return {float(v.x),float(v.y),float(v.z)};}
inline electron_fp32::ElectronUrbanMaterial value(ElectronUrbanMaterial x){return {float(x.mass),float(x.excitation),float(x.e0),float(x.e1),float(x.e2),float(x.f1),float(x.f2)};}
inline electron_fp32::ElectronBirthChannel value(ElectronBirthChannel x){return {float(x.low),float(x.high),float(x.fraction),x.offset,x.count};}
inline electron_fp32::ElectronBirthStateSample value(ElectronBirthStateSample x){return {probability(x.cdf),float(x.energy),float(x.cosine)};}
inline electron_fp32::ChargedParameter value(ChargedParameter x){return {float(x.energy),float(x.elastic_mfp),float(x.dedx),float(x.range),{float(x.diagnostic_raw_rate[0]),float(x.diagnostic_raw_rate[1])}};}
inline electron_fp32::ChargedRatePiece value(ChargedRatePiece x){return {float(x.low),float(x.high),{float(x.value[0]),float(x.value[1]),float(x.value[2]),float(x.value[3])}};}
inline electron_fp32::ChargedChannel value(ChargedChannel x){return {x.offset,x.count};}
inline electron_fp32::ChargedOutcome value(ChargedOutcome x){return {probability(x.cdf),float(x.fraction),float(x.cosine),x.pdg,x.pad};}
inline electron_fp32::PhotonRateRow value(PhotonRateRow x){return {float(x.energy),{float(x.rate[0]),float(x.rate[1]),float(x.rate[2])}};}
inline electron_fp32::PhotonChannel value(PhotonChannel x){return {x.offset,x.count};}
inline electron_fp32::PhotonOutcome value(PhotonOutcome x){return {probability(x.cdf),float(x.fraction),float(x.cosine),x.pdg,x.pad};}
inline electron_fp32::joint_math::SampledStep value(electron_joint_research::SampledStep x){
    electron_fp32::joint_math::SampledStep y{};y.cdf=probability(x.cdf);y.delta=value(x.delta);y.post=value(x.post);y.source=value(x.source);
    y.energy=x.energy;y.length=x.length;double total=0;
    for(unsigned k=0;k<4;++k){y.fraction[k]=x.fraction[k];total+=x.fraction[k];if(k<3)y.cuts[k]=probability(total).threshold;}
    for(unsigned k=0;k<2;++k)y.child[k]=x.child[k];return y;
}
inline electron_fp32::joint_math::GraphStep value(electron_joint_research::GraphStep x){
    electron_fp32::joint_math::GraphStep y{};y.delta=value(x.delta);y.length=x.length;y.postdir=value(x.postdir);
    double total=0;for(unsigned k=0;k<4;++k){y.fraction[k]=x.fraction[k];total+=x.fraction[k];if(k<3)y.cuts[k]=probability(total).threshold;}
    y.next=x.next;for(unsigned k=0;k<2;++k)y.child[k]=x.child[k];y.pdg=x.pdg;y.inside=x.inside;y.pad=x.pad;return y;
}
} // namespace electron_fp32_convert

class IndependentElectronFloatDevice {
    sycl::queue queue_;std::vector<void*> allocations_;
    template<class T>T* allocate(std::size_t count,bool mapped=false){
        if(!count)return nullptr;
        auto* p=mapped?sycl::malloc_host<T>(count,queue_):sycl::malloc_device<T>(count,queue_);
        if(!p)throw std::bad_alloc();allocations_.push_back(p);bytes+=count*sizeof(T);if(mapped)mapped_bytes+=count*sizeof(T);return p;
    }
    template<class T>auto convert(const T* source,std::size_t n,bool mapped=false){
        using R=decltype(electron_fp32_convert::value(T{}));auto* p=allocate<R>(n,mapped);
        if(mapped && n) {
            // Photon outcomes are already validated host USM. Convert them on
            // the CPU: GPU stores to mapped host pages otherwise serialize a
            // multi-GB PCIe round trip during every first load.
            if(sycl::get_pointer_type(source,queue_.get_context())!=sycl::usm::alloc::host)
                throw std::runtime_error("FP32 mapped conversion requires a host USM source");
            for(std::size_t i=0;i<n;++i)p[i]=electron_fp32_convert::value(source[i]);
        } else if(n) {
            queue_.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){p[i]=electron_fp32_convert::value(source[i]);}).wait_and_throw();
        }
        return p;
    }
    template<class T>T* copy(const T* source,std::size_t n){auto* p=allocate<T>(n);if(n)queue_.copy(source,p,n).wait_and_throw();return p;}
    template<class T>T* upload(const std::vector<T>& x){return copy(x.data(),x.size());}
    template<class T>auto host_values(const T* source,std::size_t n,bool mapped=false){
        using R=decltype(electron_fp32_convert::value(T{}));std::vector<R> values(n);
        for(std::size_t i=0;i<n;++i)values[i]=electron_fp32_convert::value(source[i]);
        auto* p=allocate<R>(n,mapped);if(n)queue_.copy(values.data(),p,n).wait_and_throw();return p;
    }
    template<class Native,class Compact,class Channels>
    Compact* outcomes(const std::filesystem::path& path,std::uint64_t count,bool compact,
                      const Channels& channels,bool mapped=false){
        static_assert(sizeof(Compact)==24 && sizeof(Native)==32);
        const auto bytes=std::filesystem::file_size(path);const auto stride=compact?sizeof(Compact):sizeof(Native);
        if(count>bytes/stride)throw std::runtime_error("FP32 outcome count");
        std::ifstream input(path,std::ios::binary);input.seekg(bytes-count*stride);
        std::vector<Compact> values(count);
        if(compact){
            if(!input.read(reinterpret_cast<char*>(values.data()),count*sizeof(Compact)))
                throw std::runtime_error("Truncated compact outcomes");
        }else{
            std::vector<Native> chunk(4096);double previous=0;std::uint64_t channel=0,end=0;
            for(std::uint64_t offset=0;offset<count;){
                const auto n=std::min<std::uint64_t>(chunk.size(),count-offset);
                if(!input.read(reinterpret_cast<char*>(chunk.data()),n*sizeof(Native)))
                    throw std::runtime_error("Truncated native outcomes");
                for(unsigned j=0;j<n;++j){
                    const auto i=offset+j;
                    while(i>=end){if(channel>=channels.size())throw std::runtime_error("Outcome channel coverage");
                        end=channels[channel].offset+channels[channel].count;++channel;previous=0;}
                    const auto& x=chunk[j];
                    if(!std::isfinite(x.cdf)||x.cdf<previous||x.cdf>1||x.cdf<0||
                       !std::isfinite(x.fraction)||x.fraction<=0||x.fraction>1||
                       !std::isfinite(x.cosine)||std::abs(x.cosine)>1||
                       (x.pdg!=0&&x.pdg!=11&&x.pdg!=22))
                        throw std::runtime_error("Invalid native outcome");
                    previous=x.cdf;if(i+1==end&&previous!=1)throw std::runtime_error("Native final CDF");
                    values[i]=electron_fp32_convert::value(x);
                }
                offset+=n;
            }
        }
        std::uint64_t offset=0;
        for(const auto& c:channels){
            if(c.offset!=offset||offset>count||c.count>count-offset)throw std::runtime_error("Compact channel bounds");
            std::uint64_t previous=0;
            for(auto i=offset;i<offset+c.count;++i){const auto& x=values[i];
                if(x.cdf.threshold<previous||!std::isfinite(x.fraction)||x.fraction<=0||x.fraction>1||
                   !std::isfinite(x.cosine)||std::abs(x.cosine)>1||(x.pdg!=0&&x.pdg!=11&&x.pdg!=22))
                    throw std::runtime_error("Invalid FP32 outcome");
                previous=x.cdf.threshold;
            }
            if(c.count&&previous!=UINT64_MAX)throw std::runtime_error("Compact final CDF");offset+=c.count;
        }
        if(offset!=count)throw std::runtime_error("Compact unreferenced outcomes");
        auto* result=allocate<Compact>(count,mapped);
        if(count)queue_.copy(values.data(),result,count).wait_and_throw();return result;
    }
    electron_fp32::ElectronJointPilotView joint_node(ElectronJointPilotView x){
        electron_fp32::ElectronJointPilotView y{};y.section=x.section;y.density=x.density;y.radius=x.radius;y.density_scaling=x.density_scaling;
        const auto& a=x.model;auto& b=y.model;b.incident=a.incident;b.continuing=a.continuing;b.entry=a.entry;b.graph_count=a.graph_count;b.bins=a.bins;
        b.edges=convert(a.edges,a.bins+1);b.offsets=copy(a.offsets,a.bins+1);b.nearest=copy(a.nearest,a.bins);
        b.steps=convert(a.steps,a.continuing+a.entry);b.graph=convert(a.graph,a.graph_count);y.pre=convert(x.pre,a.graph_count);return y;
    }
    void load_complete26_charged(const std::filesystem::path& directory){
        ElectronComposition26Data source(directory);
        std::vector<electron_composition26::AtomicVacancy26> vacancies=source.atomic.vacancies;
        for(unsigned i=0;i<vacancies.size();++i){vacancies[i].radiative=upload(source.atomic.radiative[i]);vacancies[i].auger=upload(source.atomic.auger[i]);}
        auto atomic=source.atomic.view();atomic.vacancies=upload(vacancies);
        std::vector<electron_composition26::CollisionComposition26> compositions;
        for(unsigned s=0;s<26;++s){const auto& e=source.entries[s];electron_composition26::CollisionComposition26 v;
            v.scalar=e.scalar_view();v.scalar.energy=upload(e.scalar_energy);v.scalar.scalar=upload(e.scalar_coeff);v.scalar.cuts=upload(e.cut_coeff);
            v.penelope=e.penelope_view();v.penelope.oscillators=upload(e.oscillators);v.penelope.delta=upload(e.delta);
            v.brems=e.brems_view();v.brems.energies=upload(e.brems_energy);v.brems.fractions=upload(e.brems_fraction);v.brems.integral=upload(e.brems_integral);
            v.brems.element_z=upload(e.brems_z);v.brems.element_integral=upload(e.brems_elements);v.atomic=atomic;
            queue_.copy(cut_birth26+s,&v.shells,1).wait_and_throw();compositions.push_back(v);
        }
        auto* device=upload(compositions);
        std::vector<std::array<float,2>> bounds(26);std::vector<unsigned> order(26),offsets(27);
        for(unsigned s=0;s<26;++s){
            std::ostringstream filename;filename<<"elastic"<<std::setfill('0')<<std::setw(2)<<s<<".bin";
            std::ifstream f(directory/filename.str(),std::ios::binary);
            auto read=[&]<class T>(T& x){if(!f.read(reinterpret_cast<char*>(&x),sizeof(x)))throw std::runtime_error("Complete26 angular truncated");};
            char magic[8]{};f.read(magic,8);unsigned version,section;float density;read(version);read(section);read(density);
            if(std::string_view(magic,8)!="MAIEL026"||version!=1||section!=s)throw std::runtime_error("Complete26 angular schema");
            f.read(magic,8);unsigned elastic_version,count;double mass,bc,xc2;read(elastic_version);read(count);read(mass);read(bc);read(xc2);
            if(std::string_view(magic,8)!="MAIELS01"||elastic_version!=1||count!=31||!(mass>0&&bc>0&&xc2>0))throw std::runtime_error("Complete26 angular constants");
            std::vector<double> rows(31*38);f.read(reinterpret_cast<char*>(rows.data()),rows.size()*sizeof(double));ElectronUrbanMaterial urban;read(urban);
            char extra;if(f.read(&extra,1))throw std::runtime_error("Complete26 angular trailing bytes");
            for(double x:rows)if(!std::isfinite(x))throw std::runtime_error("Complete26 angular finite");
            electron_fp32::ChargedResponseView v{};v.composition26=device+s;v.rate26=experimental_rate26+s*2;v.moller26=cut_birth26+s;
            v.section=s;v.density=density;v.density_low=source.entries[s].density_low;v.density_high=source.entries[s].density_high;
            v.lowest_energy=.0001f;v.linear_loss_limit=.01f;v.elastic={host_values(rows.data(),rows.size()),float(mass),float(bc),float(xc2)};
            v.urban=electron_fp32_convert::value(urban);charged_nodes.push_back(v);order[s]=s;offsets[s]=s;
        }
        offsets[26]=26;charged={upload(charged_nodes),upload(bounds),26,upload(offsets),upload(order)};
        // Count only the components loaded here. Birth/rates/joint/photon
        // loaders own their counters; counting the whole directory duplicates them.
        for(const char* file:{"scalar26.bin","penelope26.bin","atomic26.bin","brems26.bin"})
            source_bytes+=std::filesystem::file_size(directory/file);
        for(unsigned s=0;s<26;++s){std::ostringstream name;name<<"elastic"<<std::setfill('0')<<std::setw(2)<<s<<".bin";
            source_bytes+=std::filesystem::file_size(directory/name.str());}
        std::cout<<"[electron-complete26] compositions=26 charged_density_nodes=0 photon_density_nodes=0 old82_reads=0 fallback=0 cut=continuous_fit hard_rates=actual_cut_mass_scale penelope=calibration photon_terminals=calibration status=dose_first_candidate\n";
    }
public:
    bool complete26{};
    ElectronProjectileBirthParameters projectile_parameters{};
    const electron_composition26::NativeGridRateView* experimental_rate26{};
    const ElectronCutBirthComposition* cut_birth26{};
    electron_fp32::ElectronBirthBankView birth{};
    electron_fp32::ElectronJointPilotView joint{};
    electron_fp32::ChargedBankView charged{};
    electron_fp32::PhotonResponseBankView photon{};
    std::vector<electron_fp32::ChargedResponseView> charged_nodes;
    std::vector<std::pair<int,float>> birth_materials;
    std::uint64_t bytes{},mapped_bytes{},source_bytes{};
    IndependentElectronFloatDevice(sycl::queue q,
        const std::filesystem::path& birth_path,
        const std::filesystem::path& joint_path,
        const std::filesystem::path& charged_path,
        const std::filesystem::path& photon_path,
        unsigned projectile_z=1,unsigned projectile_a=1):queue_(q){try{
        {std::ifstream index(charged_path);std::string magic;index>>magic;complete26=magic=="MAIEC026";}
        if(complete26)source_bytes+=std::filesystem::file_size(charged_path.parent_path()/(projectile_a==1?"proton26.bin":"carbon26.bin"));
        projectile_parameters=complete26?load_electron_projectile26(charged_path.parent_path()/(projectile_a==1?"proton26.bin":"carbon26.bin"),projectile_a):electron_birth_projectile_parameters(projectile_a);
        const std::string candidate_rates=complete26?(charged_path.parent_path()/"rates26.bin").string():"";
        const char* rate_file=complete26?candidate_rates.c_str():std::getenv("CARBON_ELECTRON_RATE26_FILE");
        if(const char* file=rate_file){
            std::ifstream input(file,std::ios::binary);char magic[8]{};input.read(magic,8);
            auto read=[&]<class T>(T& x){if(!input.read(reinterpret_cast<char*>(&x),sizeof(x)))throw std::runtime_error("Rate26 truncated");};
            auto array=[&]<class T>(std::vector<T>& x,std::size_t n){x.resize(n);if(n&&!input.read(reinterpret_cast<char*>(x.data()),n*sizeof(T)))throw std::runtime_error("Rate26 array truncated");};
            unsigned version,count;read(version);read(count);
            if(std::string_view(magic,8)!="MAIERG26"||version!=1||count!=26)throw std::runtime_error("Experimental rate26 schema");
            std::vector<electron_composition26::NativeGridRateView> views;
            for(unsigned s=0;s<26;++s)for(unsigned p=0;p<2;++p){
                unsigned section,process,n;electron_composition26::NativeGridRateView v;
                read(section);read(process);read(v.pieces);read(v.degree);read(n);read(v.cut_low);read(v.cut_high);read(v.mass_majorant);
                if(section!=s||process!=p||!v.pieces||v.pieces>4096||v.degree>32||!n||n>10000000||
                   !(v.cut_low>0&&v.cut_high>=v.cut_low&&std::isfinite(v.mass_majorant)))throw std::runtime_error("Rate26 descriptor");
                v.process=p;std::vector<float> bounds,majorants,c;std::vector<unsigned> bins,intervals,offsets,pole;
                array(bounds,v.pieces+1);array(bins,v.pieces);array(intervals,v.pieces);array(offsets,v.pieces+1);array(pole,v.pieces);array(majorants,v.pieces);array(c,n);
                // Source cut-domain endpoints and physical partition bounds
                // can round to adjacent FP32 values. The evaluator clamps
                // the normalized coordinate at that single-ulp endpoint.
                if(offsets.front()!=0||offsets.back()!=n||bounds.front()>std::nextafter(v.cut_low,INFINITY)||bounds.back()<=v.cut_high)throw std::runtime_error("Rate26 bounds");
                for(unsigned i=0;i<v.pieces;++i){
                    if(!(bounds[i]<bounds[i+1])||!bins[i]||intervals[i]>bins[i]||!intervals[i]||pole[i]>1||
                       offsets[i+1]-offsets[i]!=(v.degree+1)*intervals[i]*4||!(majorants[i]>0)||!std::isfinite(majorants[i]))throw std::runtime_error("Rate26 piece");
                    double maximum=0;
                    for(unsigned j=0;j<intervals[i]*4;++j){double sum=0;
                        for(unsigned k=0;k<=v.degree;++k){const auto x=c[offsets[i]+k*intervals[i]*4+j];if(!std::isfinite(x))throw std::runtime_error("Rate26 coefficient");sum+=std::abs(double(x));}
                        maximum=std::max(maximum,sum);}
                    if(double(majorants[i])<6*maximum*1.01)throw std::runtime_error("Rate26 majorant certificate");
                }
                v.bounds=upload(bounds);v.bins=upload(bins);v.intervals=upload(intervals);v.offsets=upload(offsets);v.pole=upload(pole);v.piece_majorants=upload(majorants);v.coefficients=upload(c);views.push_back(v);
            }
            char extra;if(input.read(&extra,1))throw std::runtime_error("Rate26 trailing bytes");
            experimental_rate26=upload(views);source_bytes+=std::filesystem::file_size(file);
            std::cout<<"[electron-rate26] compositions=26 density_nodes=0 explicit_cut=interpolated_terminal_cut certificate=coefficient_triangle reserve=0.02 status=experimental_active_rate_unqualified shared_terminals="<<(complete26?"composition26":"83")<<" joint=composition_spatial\n";
        }
        // Release each source device bank before loading the next component;
        // do not keep both full representations resident across batches.
        {
            std::ifstream probe(birth_path,std::ios::binary);char magic[8]{};probe.read(magic,8);
            if(std::string_view(magic,8)=="MAIBCA26") {
                auto read=[&]<class T>(T& x){if(!probe.read(reinterpret_cast<char*>(&x),sizeof(x)))throw std::runtime_error("Cut birth26 truncated");};
                unsigned version,count;read(version);read(count);
                if(version!=1||count!=kElectronMaterialCount||(projectile_z!=1&&projectile_z!=6)||
                   (projectile_a!=1&&projectile_a!=12))throw std::runtime_error("Cut birth26 schema/species");
                std::vector<ElectronCutBirthComposition> compositions;
                for(unsigned i=0;i<count;++i){
                    unsigned section,n;float low,high;read(section);read(low);read(high);read(n);
                    if(section!=i||!std::isfinite(low)||!std::isfinite(high)||low<=0||high<low||!n||n>256)
                        throw std::runtime_error("Cut birth26 composition/domain");
                    std::vector<ElectronBirthShell> shells(n);std::uint64_t previous=0;
                    for(auto& s:shells){read(s);
                        if(s.cdf<=previous||!std::isfinite(s.binding)||s.binding<=0.f||s.z<1||s.z>100)
                            throw std::runtime_error("Cut birth26 shell law");previous=s.cdf;}
                    if(previous!=UINT64_MAX)throw std::runtime_error("Cut birth26 incomplete CDF");
                    compositions.push_back({upload(shells),n,low,high});
                    birth_materials.emplace_back(section,low);birth_materials.emplace_back(section,high);
                }
                char extra;if(probe.read(&extra,1))throw std::runtime_error("Cut birth26 trailing data");
                cut_birth26=upload(compositions);source_bytes+=std::filesystem::file_size(birth_path);
                std::cout<<"[electron-cut-birth26] compositions=26 density_states=0 parent_energy=actual cut=ion_EM angular=G4DeltaAngle status=experimental\n";
            } else {
            ElectronBirthBankDevice source(q,birth_path,projectile_z,projectile_a);source_bytes+=source.bytes;
            std::vector<electron_fp32::ElectronBirthNodeView> nodes;
            for(const auto& x:source.host_nodes){electron_fp32::ElectronBirthNodeView y{};
                y.section=x.section;y.density=x.density;y.sample_count=x.sample_count;y.channel_count=x.channel_count;
                y.channels=convert(x.channels,x.channel_count);y.samples=convert(x.samples,x.sample_count);nodes.push_back(y);birth_materials.emplace_back(y.section,y.density);}
            birth.nodes=upload(nodes);birth.count=nodes.size();birth.section_offsets=copy(source.view.section_offsets,kElectronMaterialCount+1);
            }
        }
        {
            ElectronJointPilotDevice source(q,joint_path);source_bytes+=source.bytes;
            if(source.view.sections){std::vector<ElectronJointPilotView> raw(source.view.section_count);
                q.copy(source.view.sections,raw.data(),raw.size()).wait_and_throw();std::vector<electron_fp32::ElectronJointPilotView> nodes;
                for(auto x:raw)nodes.push_back(joint_node(x));joint.sections=upload(nodes);joint.section_count=nodes.size();
            }else joint=joint_node(source.view);
        }
        if(complete26){
            if(!cut_birth26||!experimental_rate26)throw std::runtime_error("Complete26 requires independent birth/rates");
            load_complete26_charged(charged_path.parent_path());
        }else {
            std::ifstream input(charged_path);std::string magic;unsigned count=0;
            if(!(input>>magic>>count)||magic!="MAIECB02"||!count||count>4096)throw std::runtime_error("Charged index");
            std::vector<std::array<float,2>> bounds;
#ifdef CARBON_ELECTRON_CERTIFIED_RATES
            double certificate_ratio=0;std::uint64_t certificate_pieces=0;
#endif
            for(unsigned i=0;i<count;++i){
                int section;double density;std::string file;
                if(!(input>>section>>density>>file))throw std::runtime_error("Charged index row");
                const auto path=charged_path.parent_path()/file;ChargedResponseData x(path,false);source_bytes+=x.bytes;
                if(section!=x.section||density!=x.density)throw std::runtime_error("Charged identity");
                for(const auto& old:charged_nodes)if(old.section==section&&old.density==float(density))throw std::runtime_error("Duplicate charged node");
                electron_fp32::ChargedResponseView y{};
                if(float(density)<1e-6f||float(density)>1e3f||!std::isfinite(float(density)))throw std::runtime_error("Charged density domain");
                y.parameters=host_values(x.parameters.data(),x.parameters.size());y.parameter_count=x.parameters.size();
                std::array<float,2> bound{};
                for(unsigned process=0;process<2;++process){
                    double raw=0;for(const auto& r:x.rate[process])for(double v:r.value)raw=std::max(raw,6*std::abs(v));
                    bound[process]=std::nextafter(float(raw),INFINITY);
#ifdef CARBON_ELECTRON_CERTIFIED_RATES
                    certificate_ratio=std::max(certificate_ratio,electron_rate_bound::certify(x.rate[process],float(x.parameters.front().energy),float(x.parameters.back().energy),bound[process]));
                    certificate_pieces+=x.rate[process].size();
#endif
                    y.rate[process]=host_values(x.rate[process].data(),x.rate[process].size());y.rate_count[process]=x.rate[process].size();
                }
                y.edges=host_values(x.edges.data(),x.edges.size());y.bins=x.edges.size()-1;
                y.channels=host_values(x.channels.data(),x.channels.size());
                y.outcomes=outcomes<ChargedOutcome,electron_fp32::ChargedOutcome>(path,x.outcome_count,x.compact_outcomes,x.channels);y.outcome_count=x.outcome_count;
                y.elastic={host_values(x.elastic_rows.data(),x.elastic_rows.size()),float(x.mass),float(x.bc),float(x.xc2)};
                y.urban=electron_fp32_convert::value(x.urban);
                y.section=section;y.density=density;y.electron_cut=x.electron_cut;y.photon_cut=x.photon_cut;y.lowest_energy=x.lowest;y.linear_loss_limit=x.linear;
                if(experimental_rate26)y.rate26=experimental_rate26+section*2;
                if(const char* flag=std::getenv("CARBON_ELECTRON_MOLLER26")){if(std::string_view(flag)!="1")throw std::runtime_error("Experimental Moller flag must be1");
                    if(!cut_birth26)throw std::runtime_error("Moller26 requires MAIBCA26 shell birth package");y.moller26=cut_birth26+section;}
                charged_nodes.push_back(y);bounds.push_back(bound);
            }
            std::string extra;if(input>>extra)throw std::runtime_error("Charged index trailing fields");
            std::vector<unsigned> order(count),offsets(kElectronMaterialCount+1);
            for(unsigned i=0;i<count;++i)order[i]=i;
            std::sort(order.begin(),order.end(),[&](unsigned a,unsigned b){const auto& x=charged_nodes[a];const auto& y=charged_nodes[b];return x.section<y.section||(x.section==y.section&&x.density<y.density);});
            unsigned pos=0;for(unsigned sec=0;sec<=kElectronMaterialCount;++sec){while(pos<count&&charged_nodes[order[pos]].section<int(sec))++pos;offsets[sec]=pos;}
            charged.nodes=upload(charged_nodes);charged.count=count;charged.node_order=upload(order);charged.section_offsets=upload(offsets);charged.majorants=upload(bounds);
#ifdef CARBON_ELECTRON_CERTIFIED_RATES
            std::cout<<"[independent-fp32-rate-certificate] nodes="<<count<<" pieces="<<certificate_pieces<<" max_bound_ratio="<<certificate_ratio<<" reserve=0.01 endpoint_check=retained\n";
#endif
        }
        {
            std::ifstream input(photon_path);std::string magic;unsigned count=0;
            if(!(input>>magic>>count)||(magic!=(complete26?"MAIPH026":"MAIPHB02"))||(complete26&&count!=26)||!count||count>4096)throw std::runtime_error("Photon index");
            std::vector<electron_fp32::PhotonResponseView> nodes;
            for(unsigned i=0;i<count;++i){
                int section;double density;std::string file;if(!(input>>section>>density>>file))throw std::runtime_error("Photon index row");
                const auto path=photon_path.parent_path()/file;PhotonResponseData x(path,false);source_bytes+=x.bytes;
                if(section!=x.section||density!=x.density)throw std::runtime_error("Photon identity");
                electron_fp32::PhotonResponseView y{};
                y.rates=host_values(x.rates.data(),x.rates.size());y.rate_count=x.rates.size();
                y.edges=host_values(x.edges.data(),x.edges.size());y.bins=x.edges.size()-1;
                y.channels=host_values(x.channels.data(),x.channels.size());
                y.outcomes=outcomes<PhotonOutcome,electron_fp32::PhotonOutcome>(path,x.outcome_count,x.compact_outcomes,x.channels,true);
                y.outcome_count=x.outcome_count;y.section=section;y.density=density;
                if(complete26){if(section!=int(i))throw std::runtime_error("Complete26 photon directory");y.density_low=charged_nodes[i].density_low;y.density_high=charged_nodes[i].density_high;}
                nodes.push_back(y);
            }
            std::string extra;if(input>>extra)throw std::runtime_error("Photon index trailing fields");
            std::sort(nodes.begin(),nodes.end(),[](const auto& a,const auto& b){return a.section<b.section||(a.section==b.section&&a.density<b.density);});
            for(unsigned i=1;i<count;++i)if(nodes[i].section==nodes[i-1].section&&nodes[i].density<=nodes[i-1].density)throw std::runtime_error("Duplicate photon node");
            std::vector<unsigned> offsets(kElectronMaterialCount+1);unsigned pos=0;
            for(unsigned sec=0;sec<=kElectronMaterialCount;++sec){while(pos<count&&nodes[pos].section<int(sec))++pos;offsets[sec]=pos;}
            photon.nodes=upload(nodes);photon.count=count;photon.section_offsets=upload(offsets);
        }
    }catch(...){for(auto* p:allocations_)sycl::free(p,queue_);throw;}}
    ~IndependentElectronFloatDevice(){for(auto* p:allocations_)sycl::free(p,queue_);}
    IndependentElectronFloatDevice(const IndependentElectronFloatDevice&)=delete;
};
} // namespace carbon

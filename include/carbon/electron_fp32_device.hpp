#pragma once
#include "carbon/electron_material_registry.hpp"
#include "carbon/electron_fp32.hpp"
#include "carbon/electron_birth_bank_device.hpp"
#include "carbon/electron_charged_bank_device.hpp"
#include "carbon/electron_joint_pilot_device.hpp"
#include "carbon/photon_response_bank_device.hpp"
#ifdef CARBON_ELECTRON_CERTIFIED_RATES
#include "carbon/electron_rate_bound.hpp"
#include <iostream>
#endif

namespace carbon {
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
    electron_fp32::ElectronJointPilotView joint_node(ElectronJointPilotView x){
        electron_fp32::ElectronJointPilotView y{};y.section=x.section;y.density=x.density;y.radius=x.radius;y.density_scaling=x.density_scaling;
        const auto& a=x.model;auto& b=y.model;b.incident=a.incident;b.continuing=a.continuing;b.entry=a.entry;b.graph_count=a.graph_count;b.bins=a.bins;
        b.edges=convert(a.edges,a.bins+1);b.offsets=copy(a.offsets,a.bins+1);b.nearest=copy(a.nearest,a.bins);
        b.steps=convert(a.steps,a.continuing+a.entry);b.graph=convert(a.graph,a.graph_count);y.pre=convert(x.pre,a.graph_count);return y;
    }
public:
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
        // Release each source device bank before loading the next component;
        // do not keep both full representations resident across batches.
        {
            ElectronBirthBankDevice source(q,birth_path,projectile_z,projectile_a);source_bytes+=source.bytes;
            std::vector<electron_fp32::ElectronBirthNodeView> nodes;
            for(const auto& x:source.host_nodes){electron_fp32::ElectronBirthNodeView y{};
                y.section=x.section;y.density=x.density;y.sample_count=x.sample_count;y.channel_count=x.channel_count;
                y.channels=convert(x.channels,x.channel_count);y.samples=convert(x.samples,x.sample_count);nodes.push_back(y);birth_materials.emplace_back(y.section,y.density);}
            birth.nodes=upload(nodes);birth.count=nodes.size();birth.section_offsets=copy(source.view.section_offsets,kElectronMaterialCount+1);
        }
        {
            ElectronJointPilotDevice source(q,joint_path);source_bytes+=source.bytes;
            if(source.view.sections){std::vector<ElectronJointPilotView> raw(source.view.section_count);
                q.copy(source.view.sections,raw.data(),raw.size()).wait_and_throw();std::vector<electron_fp32::ElectronJointPilotView> nodes;
                for(auto x:raw)nodes.push_back(joint_node(x));joint.sections=upload(nodes);joint.section_count=nodes.size();
            }else joint=joint_node(source.view);
        }
        {
            ChargedBankDevice source(q,charged_path);source_bytes+=source.bytes;
#ifdef CARBON_ELECTRON_CERTIFIED_RATES
            std::vector<std::array<double,2>> certified_raw_bound(source.host_nodes.size());
            q.copy(source.view.majorants,certified_raw_bound.data(),certified_raw_bound.size()).wait_and_throw();
            double certificate_ratio=0;std::uint64_t certificate_pieces=0;
#endif
            for(const auto& x:source.host_nodes){electron_fp32::ChargedResponseView y{};
#ifdef CARBON_ELECTRON_CERTIFIED_RATES
                if(float(x.density)<1e-6f||float(x.density)>1e3f||!std::isfinite(float(x.density)))
                    throw std::runtime_error("FP32 hard-rate certificate: density outside proof domain");
                ChargedParameter first,last;
                q.copy(x.parameters,&first,1).wait_and_throw();
                q.copy(x.parameters+x.parameter_count-1,&last,1).wait_and_throw();
                if(!std::isfinite(float(first.energy))||!std::isfinite(float(last.energy)))
                    throw std::runtime_error("FP32 hard-rate certificate: nonfinite energy domain");
                for(unsigned p=0;p<2;++p){
                    std::vector<ChargedRatePiece> rows(x.rate_count[p]);q.copy(x.rate[p],rows.data(),rows.size()).wait_and_throw();
                    const float bound=std::nextafter(float(certified_raw_bound[charged_nodes.size()][p]),INFINITY);
                    certificate_ratio=std::max(certificate_ratio,electron_rate_bound::certify(rows,float(first.energy),float(last.energy),bound));
                    certificate_pieces+=rows.size();
                }
#endif
                y.parameters=convert(x.parameters,x.parameter_count);y.parameter_count=x.parameter_count;
                for(unsigned p=0;p<2;++p){y.rate[p]=convert(x.rate[p],x.rate_count[p]);y.rate_count[p]=x.rate_count[p];}
                y.edges=convert(x.edges,x.bins+1);y.bins=x.bins;y.channels=convert(x.channels,2*x.bins);y.outcomes=convert(x.outcomes,x.outcome_count);y.outcome_count=x.outcome_count;
                y.elastic={convert(x.elastic.rows,31*38),float(x.elastic.mass),float(x.elastic.bc),float(x.elastic.xc2)};y.urban=electron_fp32_convert::value(x.urban);
                y.section=x.section;y.density=x.density;y.electron_cut=x.electron_cut;y.photon_cut=x.photon_cut;y.lowest_energy=x.lowest_energy;y.linear_loss_limit=x.linear_loss_limit;
                charged_nodes.push_back(y);
            }
            charged.nodes=upload(charged_nodes);charged.count=charged_nodes.size();charged.node_order=copy(source.view.node_order,charged.count);charged.section_offsets=copy(source.view.section_offsets,kElectronMaterialCount+1);
            std::vector<std::array<double,2>> raw(charged.count);q.copy(source.view.majorants,raw.data(),raw.size()).wait_and_throw();
            std::vector<std::array<float,2>> bound(raw.size());for(unsigned i=0;i<raw.size();++i)for(unsigned p=0;p<2;++p)
                bound[i][p]=std::nextafter(float(raw[i][p]),INFINITY);
            charged.majorants=upload(bound);
#ifdef CARBON_ELECTRON_CERTIFIED_RATES
            std::cout<<"[independent-fp32-rate-certificate] nodes="<<charged.count<<" pieces="<<certificate_pieces
                     <<" max_bound_ratio="<<certificate_ratio<<" reserve=0.01 endpoint_check=retained\n";
#endif
        }
        {
            PhotonResponseBankDevice source(q,photon_path);source_bytes+=source.bytes;
            std::vector<electron_fp32::PhotonResponseView> nodes;
            for(const auto& x:source.host_nodes){electron_fp32::PhotonResponseView y{};
                y.rates=convert(x.rates,x.rate_count);y.rate_count=x.rate_count;y.edges=convert(x.edges,x.bins+1);y.bins=x.bins;
                y.channels=convert(x.channels,3*x.bins);y.outcomes=convert(x.outcomes,x.outcome_count,true);y.outcome_count=x.outcome_count;y.section=x.section;y.density=x.density;nodes.push_back(y);}
            photon.nodes=upload(nodes);photon.count=nodes.size();photon.section_offsets=copy(source.view.section_offsets,kElectronMaterialCount+1);
        }
    }catch(...){for(auto* p:allocations_)sycl::free(p,queue_);throw;}}
    ~IndependentElectronFloatDevice(){for(auto* p:allocations_)sycl::free(p,queue_);}
    IndependentElectronFloatDevice(const IndependentElectronFloatDevice&)=delete;
};
} // namespace carbon

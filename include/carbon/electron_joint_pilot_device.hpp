#pragma once
#include "carbon/electron_joint_pilot.hpp"
#include <sycl/sycl.hpp>
#include <fstream>
#include <vector>
#include <filesystem>
#include <stdexcept>
#include <cstring>
namespace carbon {
// Shared material-specific spatial steps with density scaling.
class ElectronJointPilotDevice {
    sycl::queue queue_;std::vector<void*> allocations_;
    static void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
    template<class T> static void read(std::ifstream& f,T* p,std::size_t n=1) {
        if(n && !f.read(reinterpret_cast<char*>(p),sizeof(T)*n))throw std::runtime_error("Truncated joint pilot");
    }
    template<class T> T* upload(const std::vector<T>& host) {
        if(host.empty())return nullptr;
        auto p=sycl::malloc_device<T>(host.size(),queue_);if(!p)throw std::bad_alloc();
        allocations_.push_back(p);queue_.copy(host.data(),p,host.size()).wait_and_throw();return p;
    }
    template<class T> static std::vector<T> array(std::ifstream& f,std::size_t n) {
        std::vector<T> v(n);read(f,v.data(),n);return v;
    }
    void model(std::ifstream& f,ElectronJointPilotView& v,std::uint64_t bytes) {
        using namespace electron_joint_research;
        auto& m=v.model;char magic[8];read(f,magic,8);
        require(!std::memcmp(magic,"JTFAM002",8),"Joint operator format");
        read(f,&m.incident);read(f,&m.continuing);read(f,&m.entry);read(f,&m.graph_count);read(f,&m.bins);
        const std::uint64_t expected=32ull+8ull*(m.bins+1ull)+4ull*(m.bins+1ull)+4ull*m.bins+
            sizeof(SampledStep)*std::uint64_t(m.continuing)+(sizeof(GraphStep)+sizeof(V))*std::uint64_t(m.graph_count);
        require(!m.entry && m.continuing && m.bins && m.bins<=65536 && expected==bytes,"Joint operator counts/size");
        auto edges=array<double>(f,m.bins+1);auto offsets=array<std::uint32_t>(f,m.bins+1);
        auto nearest=array<std::uint32_t>(f,m.bins);auto steps=array<SampledStep>(f,m.continuing);
        auto graph=array<GraphStep>(f,m.graph_count);auto pre=array<V>(f,m.graph_count);
        require(offsets.front()==0 && offsets.back()==m.continuing,"Joint offsets");
        auto vector_ok=[](V x){return std::isfinite(x.x)&&std::isfinite(x.y)&&std::isfinite(x.z);};
        auto direction_ok=[&](V x){return vector_ok(x)&&std::abs(dot(x,x)-1)<1e-5;};
        auto link=[&](int x){return x==-1 || (x>=0 && unsigned(x)<m.graph_count);};
        auto fractions=[&](const double* f,const std::int32_t* child){
            double total=0;for(int k=0;k<4;++k){require(std::isfinite(f[k])&&f[k]>=0&&f[k]<=1.000001,"Joint fraction");total+=f[k];}
            require(std::abs(total-1)<1e-5,"Joint energy sum");
            for(int k=0;k<2;++k)require(link(child[k]) && (f[k+2]==0 || child[k]>=0),"Joint child link");
        };
        for(unsigned b=0;b<m.bins;++b) {
            require(std::isfinite(edges[b])&&edges[b]>0&&edges[b+1]>edges[b]&&std::isfinite(edges[b+1]),"Joint edges");
            require(offsets[b]<=offsets[b+1]&&offsets[b+1]<=m.continuing,"Joint offsets");
            require(nearest[b]<m.bins&&offsets[nearest[b]]<offsets[nearest[b]+1],"Joint nearest");
            double cdf=0;
            for(auto k=offsets[b];k<offsets[b+1];++k){require(std::isfinite(steps[k].cdf)&&steps[k].cdf>=cdf&&steps[k].cdf<=1.000001,"Joint CDF");cdf=steps[k].cdf;}
            require(offsets[b]==offsets[b+1]||std::abs(cdf-1)<1e-7,"Joint CDF end");
        }
        for(const auto& s:steps) {
            require(vector_ok(s.delta)&&dot(s.delta,s.delta)<=v.radius*v.radius &&
                std::isfinite(s.length)&&s.length>=0&&s.length<=v.radius&&
                std::isfinite(s.energy)&&s.energy>0&&direction_ok(s.post)&&direction_ok(s.source),"Joint charged support");
            fractions(s.fraction,s.child);
        }
        for(unsigned k=0;k<m.graph_count;++k) {
            const auto& g=graph[k];require((g.pdg==11||g.pdg==22)&&vector_ok(g.delta)&&std::isfinite(g.length)&&g.length>=0,"Joint graph state");
            require(direction_ok(pre[k])&&vector_ok(g.postdir),"Joint graph direction");
            require(link(g.next)||(g.next==-2&&g.pdg==11),"Joint continuation link");
            require(g.fraction[1]==0||g.next>=0||g.next==-2||(!g.inside&&g.pdg==22),"Joint missing continuation");
            if(g.pdg==11 && g.inside)require(g.length<=v.radius&&dot(g.delta,g.delta)<=v.radius*v.radius,"Joint graph charged support");
            fractions(g.fraction,g.child);
        }
        m.edges=upload(edges);m.offsets=upload(offsets);m.nearest=upload(nearest);
        m.steps=upload(steps);m.graph=upload(graph);v.pre=upload(pre);
    }
public:
    ElectronJointPilotView view;
    std::uint64_t bytes{};unsigned model_count{1};bool density_scaling{};
    ElectronJointPilotDevice(sycl::queue q,const std::filesystem::path& path):queue_(q) {
        try {
            bytes=std::filesystem::file_size(path);
            std::ifstream f(path,std::ios::binary);char magic[8];read(f,magic,8);
            read(f,&model_count);
            require(!std::memcmp(magic,"MAIJNB02",8)&&model_count==25,"Joint material bank header");
            std::vector<ElectronJointPilotView> views(model_count);
            for(unsigned i=0;i<model_count;++i) {
                std::uint32_t section;std::uint64_t payload;auto& v=views[i];
                read(f,&section);read(f,&v.density);read(f,&v.radius);read(f,&payload);
                require(section==i&&std::isfinite(v.density)&&v.density>0&&v.radius==.05&&payload<=bytes,"Joint material model scope");
                v.section=i;v.radius=.050001;v.density_scaling=true;model(f,v,payload);
            }
            view.sections=upload(views);view.section_count=model_count;density_scaling=true;
            require(f.peek()==std::char_traits<char>::eof(),"Joint pilot trailing bytes");
        }catch(...){for(auto p:allocations_)sycl::free(p,queue_);throw;}
    }
    ~ElectronJointPilotDevice(){for(auto p:allocations_)sycl::free(p,queue_);}
    ElectronJointPilotDevice(const ElectronJointPilotDevice&)=delete;
};
} // namespace carbon

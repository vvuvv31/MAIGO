#include "joint_family_transport.hpp"
#include <sycl/sycl.hpp>
#include <openssl/evp.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>

#if !defined(CARBON_DOSE_FP32) || defined(CARBON_DOSE_FP64)
#error "This research executable requires FP32 dose scoring"
#endif
using namespace electron_joint_research;
template<class T> void read(std::ifstream& in,T* p,std::size_t n){
    in.read(reinterpret_cast<char*>(p),sizeof(T)*n);if(!in)throw std::runtime_error("Truncated operator");}
template<class T> T read(std::ifstream& in){T value;read(in,&value,1);return value;}
std::string digest(const char* path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cannot open operator");
    auto ctx=std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)>(EVP_MD_CTX_new(),EVP_MD_CTX_free);
    if(!ctx || !EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr))throw std::runtime_error("SHA initialization");
    std::vector<char> data(1<<20);
    while(in){in.read(data.data(),data.size());if(!EVP_DigestUpdate(ctx.get(),data.data(),in.gcount()))throw std::runtime_error("SHA read");}
    if(!in.eof())throw std::runtime_error("SHA IO");
    unsigned char hash[32];unsigned n;
    if(!EVP_DigestFinal_ex(ctx.get(),hash,&n) || n!=32)throw std::runtime_error("SHA finish");
    std::ostringstream out;for(auto c:hash)out<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(c);return out.str();
}
void fractions(const double* f){double total=0;for(int c=0;c<4;++c){
    if(!std::isfinite(f[c]) || f[c]<0)throw std::runtime_error("Invalid partition");total+=f[c];}
    if(std::abs(total-1)>1e-7)throw std::runtime_error("Energy closure");}
bool finite(V v){return std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z);}
int main(int argc,char** argv){try{
    if(argc!=6)throw std::runtime_error("Usage: joint_family OPERATOR SHA256 HISTORIES SEED OUTPUT_DIR");
    const auto begin=std::chrono::steady_clock::now();
    if(digest(argv[1])!=argv[2])throw std::runtime_error("Operator SHA mismatch");
    const std::uint64_t histories=std::stoull(argv[3]),seed=std::stoull(argv[4]);
    if(histories<200 || histories%20 || histories>200000000)throw std::runtime_error("History limit");
    const std::filesystem::path out(argv[5]);if(!std::filesystem::create_directory(out))throw std::runtime_error("Output must be new");
    std::ifstream in(argv[1],std::ios::binary);char magic[8];read(in,magic,8);
    if(std::string(magic,8)!="JTFAM002")throw std::runtime_error("Wrong format");
    View view;view.incident=read<double>(in);view.continuing=read<std::uint32_t>(in);
    view.entry=read<std::uint32_t>(in);view.graph_count=read<std::uint32_t>(in);view.bins=read<std::uint32_t>(in);
    if(!std::isfinite(view.incident)||view.incident<=0||view.incident>.65||!view.continuing||
       !view.graph_count||!view.bins||view.bins>4096||view.continuing>5000000||view.entry>1000000||view.graph_count>20000000)
        throw std::runtime_error("Invalid operator bounds");
    const std::uint64_t bytes=32+(view.bins+1ULL)*12+view.bins*4ULL+
        (view.continuing+std::uint64_t(view.entry))*sizeof(SampledStep)+std::uint64_t(view.graph_count)*sizeof(GraphStep);
    if(std::filesystem::file_size(argv[1])!=bytes)throw std::runtime_error("Operator byte count");
    std::vector<double> edges(view.bins+1);std::vector<std::uint32_t> offsets(view.bins+1),nearest(view.bins);
    std::vector<SampledStep> steps(view.continuing+view.entry);std::vector<GraphStep> graph(view.graph_count);
    read(in,edges.data(),edges.size());read(in,offsets.data(),offsets.size());read(in,nearest.data(),nearest.size());
    read(in,steps.data(),steps.size());read(in,graph.data(),graph.size());
    if(offsets.front()!=0||offsets.back()!=view.continuing)throw std::runtime_error("Bad index extent");
    for(std::uint32_t b=0;b<view.bins;++b){
        if(!std::isfinite(edges[b])||edges[b]<=0||!std::isfinite(edges[b+1])||edges[b+1]<=edges[b]||
           offsets[b]>offsets[b+1]||offsets[b+1]>view.continuing||nearest[b]>=view.bins||
           offsets[nearest[b]]>=offsets[nearest[b]+1])throw std::runtime_error("Bad energy index");
    }
    auto check_cdf=[&](std::uint32_t lo,std::uint32_t hi){double last=0.;for(auto i=lo;i<hi;++i){
        if(!std::isfinite(steps[i].cdf)||steps[i].cdf<=last||steps[i].cdf>1)throw std::runtime_error("Bad CDF");last=steps[i].cdf;}
        if(hi>lo&&last!=1.)throw std::runtime_error("CDF terminal");};
    for(std::uint32_t b=0;b<view.bins;++b)check_cdf(offsets[b],offsets[b+1]);
    check_cdf(view.continuing,view.continuing+view.entry);
    auto check_child=[&](std::int32_t row,double fraction){if(row < -1 || (row>=0&&std::uint32_t(row)>=view.graph_count)||
        (fraction>0&&row<0))throw std::runtime_error("Bad child link");};
    for(auto s:steps){fractions(s.fraction);
        if(!finite(s.delta)||!finite(s.post)||!finite(s.source)||std::abs(dot(s.post,s.post)-1)>1e-7||
           std::abs(dot(s.source,s.source)-1)>1e-7||!std::isfinite(s.length)||s.length<0||s.length>.250001||
           !std::isfinite(s.energy)||s.energy<=0)throw std::runtime_error("Bad sampled step");
        for(int c=0;c<2;++c)check_child(s.child[c],s.fraction[c+2]);}
    for(auto g:graph){fractions(g.fraction);
        if(!finite(g.delta)||!std::isfinite(g.length)||g.length<0||(g.inside&&g.pdg==11&&g.length>.250001)||
           (g.pdg!=11&&g.pdg!=22)||(g.inside!=0&&g.inside!=1)||g.pad||g.next < -2||
           (g.next>=0&&std::uint32_t(g.next)>=view.graph_count))throw std::runtime_error("Bad graph row");
        if(g.next==-2 && (g.pdg!=11||!g.inside||!finite(g.postdir)||std::abs(dot(g.postdir,g.postdir)-1)>1e-7))
            throw std::runtime_error("Invalid electron continuation state");
        for(int c=0;c<2;++c)check_child(g.child[c],g.fraction[c+2]);}
    sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::enable_profiling{});
    auto upload=[&]<class T>(const std::vector<T>& values){auto* p=sycl::malloc_device<T>(values.size(),q);
        if(!p)throw std::bad_alloc();q.copy(values.data(),p,values.size()).wait_and_throw();return p;};
    view.edges=upload(edges);view.offsets=upload(offsets);view.nearest=upload(nearest);
    view.steps=upload(steps);view.graph=upload(graph);
    const std::size_t n=histories/20;auto* results=sycl::malloc_device<Result>(n,q);
    auto* scores=sycl::malloc_device<float>(32*72,q);if(!results||!scores)throw std::bad_alloc();
    std::vector<Result> host(n);std::vector<float> host_scores(32*72);
    std::ofstream profile(out/"profile.f32",std::ios::binary),counts(out/"counts.u64",std::ios::binary);
    std::uint64_t total[6]{},advances=0,graph_advances=0,below=0;double seconds=0;
    const auto setup=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    for(unsigned block=0;block<20;++block){
        q.fill(scores,0.f,32*72).wait_and_throw();
        auto event=q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){
            const auto i=std::uint64_t(id[0]);const auto r=transport(view,seed,block*n+i);results[i]=r;
            if(r.status==0||r.status==2){
                const double folded=r.x-3.6*sycl::floor((r.x+1.8)/3.6);
                int bin=int(sycl::floor((folded+1.8)/.1));bin=bin<0?0:(bin>35?35:bin);
                const auto index=(i%32)*72+(r.status==2?36:0)+bin;
                sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(scores[index]).fetch_add(1.f);
            }
        });event.wait_and_throw();
        const double time=(event.get_profiling_info<sycl::info::event_profiling::command_end>()-
                           event.get_profiling_info<sycl::info::event_profiling::command_start>())*1e-9;
        seconds+=time;q.copy(results,host.data(),n).wait_and_throw();q.copy(scores,host_scores.data(),host_scores.size()).wait_and_throw();
        std::uint64_t count[6]{};for(auto r:host){if(r.status>5||!std::isfinite(r.x)||!std::isfinite(r.y)||!std::isfinite(r.z))throw std::runtime_error("Invalid result");
            ++count[r.status];advances+=r.advances;graph_advances+=r.graph_advances;below+=r.below;}
        float h[72]{};for(unsigned i=0;i<32*72;++i)h[i%72]+=host_scores[i];
        double sum=0;for(float x:h)sum+=x;if(sum!=count[0]+count[2])throw std::runtime_error("FP32 score closure");
        profile.write(reinterpret_cast<const char*>(h),sizeof(h));counts.write(reinterpret_cast<const char*>(count),sizeof(count));
        for(int i=0;i<6;++i)total[i]+=count[i];
        std::cout<<"block="<<block+1<<" seconds="<<time<<" coverage="<<count[3]<<" invalid="<<count[4]<<" unfinished="<<count[5]<<std::endl;
        if(count[3]||count[4]||count[5]){
            for(std::size_t i=0;i<host.size();++i){const auto r=host[i];if(r.status<3)continue;
                std::cerr<<std::setprecision(17)<<"failed_history="<<block*n+i<<" reason="<<r.reason<<" iteration="<<r.iteration
                    <<" cursor="<<r.cursor<<" sampled_row="<<r.sampled_row<<" energy="<<r.energy<<'\n';}
            throw std::runtime_error("Research packet coverage/validity gate");}
    }
    profile.close();counts.close();if(!profile||!counts)throw std::runtime_error("Output write");
    std::ofstream report(out/"run.json");report<<std::setprecision(17)
        <<"{\"status\":\"GPU_RESEARCH_SCREEN_ONLY\",\"ct_acceptance\":false,\"production_promoted\":false,\"scorer\":\"FP32\","
        <<"\"histories\":"<<histories<<",\"seed\":"<<seed<<",\"kernel_seconds\":"<<seconds<<",\"setup_seconds\":"<<setup
        <<",\"operator_bytes\":"<<bytes<<",\"advances\":"<<advances<<",\"graph_advances\":"<<graph_advances<<",\"below_minimum\":"<<below
        <<",\"operator_sha256\":\""<<argv[2]<<"\",\"device\":\""<<q.get_device().get_info<sycl::info::device::name>()<<"\"}\n";
    report.close();if(!report)throw std::runtime_error("Report write");
    sycl::free(const_cast<double*>(view.edges),q);sycl::free(const_cast<std::uint32_t*>(view.offsets),q);
    sycl::free(const_cast<std::uint32_t*>(view.nearest),q);sycl::free(const_cast<SampledStep*>(view.steps),q);
    sycl::free(const_cast<GraphStep*>(view.graph),q);sycl::free(results,q);sycl::free(scores,q);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

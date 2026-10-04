#include "carbon/electron_fp32_device.hpp"
#include <iostream>
struct Pair { unsigned low,high; float density; };
int main(int argc,char** argv){try{
    if(argc!=9)throw std::runtime_error("usage: birth sha joint sha charged sha photon sha");
    namespace f=carbon::electron_fp32;
    sycl::queue q{sycl::gpu_selector_v};
    carbon::IndependentElectronFloatDevice data(q,argv[1],argv[2],argv[3],argv[4],argv[5],argv[6],argv[7],argv[8]);
    std::vector<Pair> pairs;
    for(unsigned i=0;i<data.charged_nodes.size();++i)pairs.push_back({i,i,data.charged_nodes[i].density});
    for(unsigned sec=0;sec<25;++sec){
        std::vector<unsigned> ids;for(unsigned i=0;i<data.charged_nodes.size();++i)if(data.charged_nodes[i].section==int(sec))ids.push_back(i);
        std::sort(ids.begin(),ids.end(),[&](auto a,auto b){return data.charged_nodes[a].density<data.charged_nodes[b].density;});
        for(unsigned j=1;j<ids.size();++j)pairs.push_back({ids[j-1],ids[j],.5f*(data.charged_nodes[ids[j-1]].density+data.charged_nodes[ids[j]].density)});
    }
    auto* spec=sycl::malloc_shared<Pair>(pairs.size(),q);std::copy(pairs.begin(),pairs.end(),spec);
    const unsigned n=pairs.size()*8192;auto* bad=sycl::malloc_shared<unsigned>(n,q);const auto bank=data.charged;const unsigned np=pairs.size();
    q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){
        const unsigned i=id[0];const auto pair=spec[i%np];
        const auto model=f::charged_density_model(bank.nodes+pair.low,bank.nodes+pair.high,pair.density);
        std::uint64_t rng=0x482abcd789ULL+i*0x9e3779b97f4a7c15ULL;
        auto u=[&](){return f::random_draw(rng);};
        const float energy=std::exp(std::log(1e-6f)+float(u())*std::log(.65f/1e-6f));
        const auto p=f::charged_parameters(model,energy);
        const unsigned regime=(i/np)%4;
        const float length=regime==0?0.f:regime==1?.25f:regime==2?std::min(.25f,p.range):std::min(.05f,.005f*energy/p.dedx);
        std::uint64_t rng_b=rng;auto v=[&](){return f::random_draw(rng_b);};
        const auto a=f::sample_charged_loss(model,energy,length,u);
        const auto b=f::sample_charged_loss_prepared(model,energy,length,v,p);
        bad[i]=a.valid!=b.valid||a.stopped!=b.stopped||a.mean!=b.mean||a.raw!=b.raw||a.deposit!=b.deposit||a.remaining!=b.remaining||rng!=rng_b;
    }).wait_and_throw();
    std::uint64_t mismatches=0;for(unsigned i=0;i<n;++i)mismatches+=bad[i];
    std::cout<<"{\"passed\":"<<(mismatches?"false":"true")<<",\"density_cases\":"<<pairs.size()<<",\"samples\":"<<n<<",\"field_or_rng_mismatches\":"<<mismatches<<"}\n";
    sycl::free(spec,q);sycl::free(bad,q);return mismatches?1:0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

#include "carbon/electron_urban_fluctuation.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
struct LossCase{double energy,mean,cut;};
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("usage: index");
    const std::filesystem::path path=argv[1];
    std::ifstream index(path);std::string magic;unsigned nodes;index>>magic>>nodes;
    if(magic!="MAIURB02"||!nodes||nodes>4096)throw std::runtime_error("Index format");
    sycl::queue q{sycl::gpu_selector_v};std::uint64_t total=0;double maxError=0;
    for(unsigned node=0;node<nodes;++node){
        std::string file,caseFile;index>>file>>caseFile;
        const auto materialPath=path.parent_path()/file,casePath=path.parent_path()/caseFile;
        if(!index)throw std::runtime_error("Fixture index mismatch");
        carbon::ElectronUrbanMaterial material;std::ifstream mf(materialPath,std::ios::binary);mf.read(reinterpret_cast<char*>(&material),sizeof(material));
        if(!mf||mf.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Material size");
        std::ifstream cf(casePath);std::string line;std::getline(cf,line);std::vector<LossCase> cases;
        while(std::getline(cf,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);unsigned cell;LossCase c;row>>cell>>c.energy>>c.mean>>c.cut;if(!row||cell!=cases.size())throw std::runtime_error("Case row");cases.push_back(c);}
        if(cases.size()!=40)throw std::runtime_error("Expected40 law cases");
        constexpr unsigned samples=1024;const unsigned count=cases.size()*samples;
        auto* inputs=sycl::malloc_shared<LossCase>(cases.size(),q);auto* output=sycl::malloc_shared<carbon::ElectronUrbanLoss>(count,q);auto* score=sycl::malloc_shared<float>(1,q);
        if(!inputs||!output||!score)throw std::bad_alloc();std::copy(cases.begin(),cases.end(),inputs);*score=0;
        q.parallel_for(sycl::range<1>(count),[=](sycl::id<1> id){
            const auto i=id[0];const unsigned cell=i/samples,j=i%samples;
            std::uint64_t rng=(0x435674235aULL+std::uint64_t(cell)*0x123465789ULL+j*0x9e3779b97f4a7c15ULL)^0x81742345123ULL;
            auto uniform=[&](){rng=rng*6364136223846793005ULL+1442695040888963407ULL;return (double(rng>>12)+.5)*0x1p-52;};
            const auto c=inputs[cell];output[i]=carbon::sample_electron_urban_loss(material,c.energy,c.mean,c.cut,uniform);
            sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> sum(*score);sum.fetch_add(1.f);
        }).wait_and_throw();
        for(unsigned i=0;i<count;++i){const unsigned cell=i/samples,j=i%samples;
            std::uint64_t rng=(0x435674235aULL+std::uint64_t(cell)*0x123465789ULL+j*0x9e3779b97f4a7c15ULL)^0x81742345123ULL;
            auto uniform=[&](){rng=rng*6364136223846793005ULL+1442695040888963407ULL;return (double(rng>>12)+.5)*0x1p-52;};
            const auto c=cases[cell];const auto ref=carbon::sample_electron_urban_loss(material,c.energy,c.mean,c.cut,uniform);
            if(!ref.valid||!output[i].valid||ref.random_draws!=output[i].random_draws)throw std::runtime_error("GPU/CPU sampler state mismatch");
            maxError=std::max(maxError,std::abs(ref.loss-output[i].loss)/std::max(1e-5,ref.loss));
        }
        if(*score!=count||maxError>1e-9)throw std::runtime_error("GPU/CPU loss/FP32 tally mismatch");
        total+=count;sycl::free(inputs,q);sycl::free(output,q);sycl::free(score,q);
    }
    std::string extra;if(index>>extra)throw std::runtime_error("Trailing index fields");
    std::cout<<std::setprecision(17)<<"{\"passed\":true,\"nodes\":"<<nodes<<",\"samples\":"<<total<<",\"maximum_scaled_loss_error\":"<<maxError<<",\"random_draw_counts_equal\":true,\"scorer\":\"FP32\"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

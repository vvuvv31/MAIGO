#include "electron_multiple_fixture.hpp"
#include <iomanip>
int main(int argc,char** argv){try{
 if(argc!=7)throw std::runtime_error("usage: charged sha metadata sha samples output");
 namespace x=multiple_fixture;namespace m=carbon::electron_multiple_cdf_research;namespace f=carbon::electron_fp32;
 sycl::queue queue{sycl::gpu_selector_v};x::Data data(queue,argv[1],argv[2],argv[3],argv[4]);const unsigned samples=std::stoul(argv[5]);
 const unsigned counts=data.max_count+2; // includes N=0,1 and N=max+1 fallback
 const std::size_t count=std::size_t(data.count)*counts*samples;const auto* laws=data.laws;
 auto* output=sycl::malloc_shared<float>(count,queue);if(!output)throw std::bad_alloc();
 for(unsigned arm=0;arm<2;++arm){
  queue.parallel_for(sycl::range<1>(count),[=](sycl::id<1> id){const std::size_t i=id[0];const unsigned ci=i/samples,n=ci%counts;const auto law=laws[ci/counts];
   auto rng=x::seed(0x5971ab43ULL+i+arm*0x6a09e667f3bcc909ULL);auto uniform=[&](){return f::random_draw(rng);};
   output[i]=arm?m::angle(law.soft,law.table,n,uniform):m::explicit_angle(law.soft,n,uniform);
  }).wait_and_throw();
  std::uint64_t bad=0;for(std::size_t i=0;i<count;++i)bad+=!std::isfinite(output[i])||output[i]<0||output[i]>2;
  std::ofstream out(std::string(argv[6])+(arm?".candidate.bin":".reference.bin"),std::ios::binary);out.write(reinterpret_cast<const char*>(output),count*sizeof(float));if(!out)throw std::runtime_error("output write");
  std::cout<<"{\"arm\":"<<arm<<",\"draws\":"<<count<<",\"law_count\":"<<data.count<<",\"counts_per_law\":"<<counts<<",\"invalid\":"<<bad<<"}"<<std::endl;if(bad)return 1;
 }
 sycl::free(output,queue);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

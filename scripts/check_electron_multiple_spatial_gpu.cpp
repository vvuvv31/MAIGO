#include "electron_multiple_fixture.hpp"
#include <iomanip>
#include <chrono>
struct Output {float position[3],direction[3];unsigned events,mode;};
static_assert(sizeof(Output)==32);
int main(int argc,char** argv){try{
 if(argc!=7)throw std::runtime_error("usage: charged sha metadata sha samples output");
 namespace x=multiple_fixture;namespace m=carbon::electron_multiple_cdf_research;namespace c=carbon::electron_condensed_research;namespace f=carbon::electron_fp32;
 sycl::queue queue(sycl::gpu_selector_v,sycl::property::queue::enable_profiling{});x::Data data(queue,argv[1],argv[2],argv[3],argv[4]);const unsigned samples=std::stoul(argv[5]);
 const unsigned nc=data.count*7;const std::size_t count=std::size_t(nc)*samples;const auto* laws=data.laws;
 auto* output=sycl::malloc_shared<Output>(count,queue);if(!output)throw std::bad_alloc();
 // Carry the incident direction as a kernel argument, as in transport. Do
 // not give only an inlined arm a compile-time constant initial direction.
 auto launch=[&]<bool candidate>(c::V initial){
  sycl::event event=queue.parallel_for(sycl::range<1>(count),[=](sycl::id<1> id){const std::size_t i=id[0];const unsigned ci=i/samples;const auto law=laws[ci/7];
   constexpr float mus[]={.1f,.5f,1,2,4,8,16};const float mu=mus[ci%7];
   auto rng=x::seed(0x37abcdef912ULL+i+(candidate?0x6a09e667f3bcc909ULL:0));auto uniform=[&](){return f::random_draw(rng);};
   const auto r=candidate?m::segment(law.soft,law.table,mu,.05f,initial,uniform):c::explicit_segment(law.soft,mu,.05f,initial,uniform);
   output[i]={{r.position.x,r.position.y,r.position.z},{r.direction.x,r.direction.y,r.direction.z},r.events,r.valid?r.mode+1:0};
  });event.wait_and_throw();return 1e-6*double(event.get_profiling_info<sycl::info::event_profiling::command_end>()-event.get_profiling_info<sycl::info::event_profiling::command_start>());
 };
 for(unsigned arm=0;arm<2;++arm){
  if(arm)launch.template operator()<true>({0,0,1});else launch.template operator()<false>({0,0,1});
  std::uint64_t bad=0;for(std::size_t i=0;i<count;++i)bad+=!output[i].mode;
  std::ofstream out(std::string(argv[6])+(arm?".candidate.bin":".reference.bin"),std::ios::binary);out.write(reinterpret_cast<const char*>(output),count*sizeof(Output));if(!out)throw std::runtime_error("output write");
  std::cout<<"{\"type\":\"distribution\",\"arm\":"<<arm<<",\"draws\":"<<count<<",\"cases\":"<<nc<<",\"invalid\":"<<bad<<"}"<<std::endl;if(bad)return 1;
 }
 // Warm both complete segment kernels after host sample readback. No data
 // loading or CPU compilation overlaps these isolated component timings.
 const auto warm_start=std::chrono::steady_clock::now();unsigned warm_pairs=0;
 do{launch.template operator()<false>({0,0,1});launch.template operator()<true>({0,0,1});++warm_pairs;}
 while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm_start).count()<1.0);
 std::cout<<"{\"type\":\"warmup\",\"pairs\":"<<warm_pairs<<",\"minimum_seconds\":1}"<<std::endl;
 for(unsigned i=0;i<3;++i){
  const double a1=launch.template operator()<false>({0,0,1}),b1=launch.template operator()<true>({0,0,1}),b2=launch.template operator()<true>({0,0,1}),a2=launch.template operator()<false>({0,0,1});
  std::cout<<std::setprecision(17)<<"{\"type\":\"component_timing\",\"repeat\":"<<i<<",\"reference_ms\":"<<(a1+a2)/2<<",\"candidate_ms\":"<<(b1+b2)/2<<",\"ratio\":"<<(b1+b2)/(a1+a2)<<"}"<<std::endl;
 }
 sycl::free(output,queue);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

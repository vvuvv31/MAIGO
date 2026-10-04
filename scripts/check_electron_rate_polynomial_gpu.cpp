#include "carbon/electron_charged_data.hpp"
#include "carbon/electron_fp32.hpp"
#include "carbon/electron_rate_polynomial.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
namespace f=carbon::electron_fp32;
namespace p=carbon::electron_rate_polynomial;
struct Row {f::ChargedRatePiece raw;p::Piece poly;float majorant;};
struct Result {float absolute{},relative{};unsigned invalid{},support{};};
struct Lookup {unsigned first{},count{};float low{},high{};};
template<class Piece>float lookup(const Piece* rows,unsigned count,float energy){
    if(energy<rows[0].low||energy>rows[count-1].high)return NAN;
    unsigned lo=0,hi=count;
    while(lo+1<hi){const auto m=lo+(hi-lo)/2;if(rows[m].low>energy)hi=m;else lo=m;}
    const auto& r=rows[lo];return p::evaluate(r,(energy-r.low)/(r.high-r.low));
}
int main(int argc,char** argv){try{
    if(argc!=3)throw std::runtime_error("usage: charged index SHA");
    const std::filesystem::path path=argv[1];
    if(carbon::compute_file_sha256_hex(path)!=argv[2])throw std::runtime_error("index SHA");
    std::ifstream in(path);std::string magic;unsigned nodes;in>>magic>>nodes;
    if(magic!="MAIECB01")throw std::runtime_error("index magic");
    std::vector<Row> host;std::vector<Lookup> queries;
    for(unsigned i=0;i<nodes;++i){
        int section;double density;std::string file,pin;in>>section>>density>>file>>pin;
        carbon::ChargedResponseData data(path.parent_path()/file,pin);
        if(!in||section!=data.section||density!=data.density)throw std::runtime_error("node metadata");
        for(unsigned process=0;process<2;++process){
            double bound=0;for(const auto& r:data.rate[process])for(double y:r.value)bound=std::max(bound,6*std::abs(y));
            float majorant=std::nextafter(float(bound),INFINITY);
            p::certify(data.rate[process],float(data.parameters.front().energy),float(data.parameters.back().energy),majorant);
            queries.push_back({unsigned(host.size()),unsigned(data.rate[process].size()),float(data.parameters.front().energy),float(data.parameters.back().energy)});
            for(const auto& r:data.rate[process])host.push_back({{float(r.low),float(r.high),{float(r.value[0]),float(r.value[1]),float(r.value[2]),float(r.value[3])}},p::convert(r),majorant});
        }
    }
    sycl::queue queue(sycl::gpu_selector_v,sycl::property::queue::enable_profiling{});
    const unsigned n=host.size(),nq=queries.size();
    auto* rows=sycl::malloc_device<Row>(n,queue);queue.copy(host.data(),rows,n).wait_and_throw();
    auto* result=sycl::malloc_shared<Result>(n,queue);
    queue.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){
        const auto row=rows[id[0]];Result r{};
        f::ChargedResponseView view{};view.rate[0]=&row.raw;view.rate_count[0]=1;
        for(unsigned j=0;j<=4096;++j){
            const float e=std::fma(float(j)/4096,row.raw.high-row.raw.low,row.raw.low);
            const float a=f::charged_raw_rate(view,0,e),b=p::evaluate(row.poly,(e-row.raw.low)/(row.raw.high-row.raw.low));
            r.invalid+=!std::isfinite(a)||!std::isfinite(b);r.support+=(a>0)!=(b>0);
            r.absolute=std::max(r.absolute,std::abs(a-b)/row.majorant);
            if(std::abs(a)>1e-6f*row.majorant)r.relative=std::max(r.relative,std::abs(a-b)/std::abs(a));
        }
        result[id[0]]=r;
    }).wait_and_throw();
    Result all{};for(unsigned i=0;i<n;++i){all.invalid+=result[i].invalid;all.support+=result[i].support;all.absolute=std::max(all.absolute,result[i].absolute);all.relative=std::max(all.relative,result[i].relative);}
    const bool pass=!all.invalid&&!all.support&&all.absolute<2e-6f&&all.relative<2e-5f;
    std::cout<<std::setprecision(17)<<"{\"type\":\"component\",\"passed\":"<<(pass?"true":"false")<<",\"pieces\":"<<n<<",\"points\":"<<std::uint64_t(n)*4097<<",\"invalid\":"<<all.invalid<<",\"support_changes\":"<<all.support<<",\"max_absolute_to_majorant\":"<<all.absolute<<",\"max_relative_above_1e_6_majorant\":"<<all.relative<<"}\n";
    if(!pass)return 1;
    // Isolated runtime lookup screen includes binary search and the changed
    // record layout. This is not a CT throughput prediction.
    std::vector<f::ChargedRatePiece> raw;std::vector<p::Piece> poly;
    for(const auto& r:host){raw.push_back(r.raw);poly.push_back(r.poly);}
    auto* a=sycl::malloc_device<f::ChargedRatePiece>(n,queue);auto* b=sycl::malloc_device<p::Piece>(n,queue);
    auto* q=sycl::malloc_device<Lookup>(nq,queue);
    queue.copy(raw.data(),a,n).wait_and_throw();queue.copy(poly.data(),b,n).wait_and_throw();queue.copy(queries.data(),q,nq).wait_and_throw();
    const unsigned count=1u<<24;auto* output=sycl::malloc_device<float>(count,queue);
    auto launch=[&]<bool candidate>(){
        sycl::event event=queue.parallel_for(sycl::range<1>(count),[=](sycl::id<1> id){
            const unsigned i=id[0];const auto spec=q[i%nq];unsigned state=i*747796405u+2891336453u;state=((state>>((state>>28u)+4u))^state)*277803737u;state=(state>>22u)^state;
            const float u=float(state>>8)*(1.f/16777216.f);
            const float e=std::exp(std::log(spec.low)+u*std::log(spec.high/spec.low));
            if constexpr(candidate)output[i]=lookup(b+spec.first,spec.count,e);
            else{f::ChargedResponseView v{};v.rate[0]=a+spec.first;v.rate_count[0]=spec.count;output[i]=f::charged_raw_rate(v,0,e);}
        });
        event.wait_and_throw();return 1e-6*double(event.get_profiling_info<sycl::info::event_profiling::command_end>()-event.get_profiling_info<sycl::info::event_profiling::command_start>());
    };
    launch.template operator()<false>();launch.template operator()<true>();
    for(unsigned i=0;i<3;++i){
        const double a1=launch.template operator()<false>(),b1=launch.template operator()<true>(),b2=launch.template operator()<true>(),a2=launch.template operator()<false>();
        std::cout<<"{\"type\":\"lookup_timing\",\"repeat\":"<<i<<",\"lookups\":"<<count<<",\"reference_ms\":"<<(a1+a2)/2<<",\"candidate_ms\":"<<(b1+b2)/2<<",\"ratio\":"<<(b1+b2)/(a1+a2)<<"}\n";
    }
    for(void* ptr:{static_cast<void*>(rows),static_cast<void*>(result),static_cast<void*>(a),static_cast<void*>(b),static_cast<void*>(q),static_cast<void*>(output)})sycl::free(ptr,queue);
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

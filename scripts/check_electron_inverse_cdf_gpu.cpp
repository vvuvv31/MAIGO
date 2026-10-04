#include "carbon/electron_fp32.hpp"
#include "carbon/electron_elastic_inverse_cdf.hpp"
#include "carbon/electron_charged_data.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
struct Case {unsigned node;float energy;};
int main(int argc,char** argv){try{
    if(argc!=9)throw std::runtime_error("usage: index sha cases tables bins samples repeats output-prefix");
    namespace f=carbon::electron_fp32;
    const std::filesystem::path index=argv[1];
    if(carbon::compute_file_sha256_hex(index)!=argv[2])throw std::runtime_error("index SHA");
    std::ifstream in(index);std::string magic;unsigned nodes;in>>magic>>nodes;
    if(magic!="MAIECB01"||nodes!=82)throw std::runtime_error("index header");
    const unsigned bins=std::stoul(argv[5]),samples=std::stoul(argv[6]),repeats=std::stoul(argv[7]);
    if((bins!=16&&bins!=32&&bins!=64)||!samples||samples>1000000||repeats<2||repeats>20)throw std::runtime_error("arguments");
    sycl::queue queue{sycl::gpu_selector_v,sycl::property::queue::enable_profiling{}};
    auto* rows=sycl::malloc_shared<float>(nodes*31*38,queue);
    auto* views=sycl::malloc_shared<f::ElectronElasticView>(nodes,queue);
    if(!rows||!views)throw std::bad_alloc();
    for(unsigned i=0;i<nodes;++i){int section;double density;std::string path,sha;
        in>>section>>density>>path>>sha;if(!in)throw std::runtime_error("index row");
        carbon::ChargedResponseData data(index.parent_path()/path,sha);
        if(data.section!=section||data.density!=density)throw std::runtime_error("node identity");
        for(unsigned k=0;k<31*38;++k)rows[i*31*38+k]=float(data.elastic_rows[k]);
        views[i]={rows+i*31*38,float(data.mass),float(data.bc),float(data.xc2)};
    }
    std::ifstream cf(argv[3]);unsigned nc;cf>>nc;if(!nc)throw std::runtime_error("case count");
    const unsigned knots=48*bins+1;
    const std::size_t count=std::size_t(nc)*samples,nt=std::size_t(nc)*knots;
    auto* cases=sycl::malloc_shared<Case>(nc,queue);
    auto* tables=sycl::malloc_shared<float>(nt,queue);
    auto* output=sycl::malloc_shared<float>(count,queue);
    if(!cases||!tables||!output)throw std::bad_alloc();
    for(unsigned i=0;i<nc;++i){cf>>cases[i].node>>cases[i].energy;if(!cf||cases[i].node>=nodes)throw std::runtime_error("case row");}
    std::ifstream tf(argv[4],std::ios::binary);tf.read(reinterpret_cast<char*>(tables),nt*sizeof(float));
    if(!tf||tf.peek()!=std::char_traits<char>::eof())throw std::runtime_error("table length");
    for(unsigned c=0;c<nc;++c){
        const auto* q=tables+std::size_t(c)*knots;
        if(q[0]!=0||q[knots-1]!=2)throw std::runtime_error("table support");
        for(unsigned i=0;i<knots;++i)if(!std::isfinite(q[i])||(i&&q[i]<q[i-1]))throw std::runtime_error("table order");
    }
    auto run=[&]<bool table>(){
        for(unsigned repeat=0;repeat<repeats;++repeat){
            auto event=queue.parallel_for(sycl::range<1>(count),[=](sycl::id<1> id){
                const auto i=id[0];const unsigned c=i/samples;
                std::uint64_t rng=0x44149abcdefULL+i+(table?0x6a09e667f3bcc909ULL:0x9e3779b97f4a7c15ULL);
                rng=(rng^(rng>>30))*0xbf58476d1ce4e5b9ULL;rng=(rng^(rng>>27))*0x94d049bb133111ebULL;rng^=rng>>31;
                auto uniform=[&](){return f::random_draw(rng);};
                if constexpr(table) output[i]=carbon::electron_inverse_cdf_research::sample(tables+std::size_t(c)*knots,bins,uniform());
                else{const auto draw=f::sample_electron_elastic(views[cases[c].node],cases[c].energy,uniform);output[i]=draw.valid?draw.one_minus_cosine:NAN;}
            });event.wait_and_throw();
            const auto elapsed=event.template get_profiling_info<sycl::info::event_profiling::command_end>()-event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            std::cout<<std::setprecision(9)<<"{\"arm\":"<<table<<",\"repeat\":"<<repeat<<",\"warmup\":"<<(repeat==0?"true":"false")<<",\"samples\":"<<count<<",\"kernel_seconds\":"<<elapsed*1e-9<<"}"<<std::endl;
        }
        for(std::size_t i=0;i<count;++i)if(!std::isfinite(output[i])||output[i]<0||output[i]>2)throw std::runtime_error("invalid angle");
        std::ofstream out(std::string(argv[8])+(table?".candidate.bin":".reference.bin"),std::ios::binary);
        out.write(reinterpret_cast<const char*>(output),count*sizeof(float));if(!out)throw std::runtime_error("output write");
    };
    run.template operator()<false>();run.template operator()<true>();
    sycl::free(output,queue);sycl::free(tables,queue);sycl::free(cases,queue);sycl::free(views,queue);sycl::free(rows,queue);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

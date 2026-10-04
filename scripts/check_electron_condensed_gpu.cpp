#include "carbon/electron_condensed_elastic.hpp"
#include "carbon/electron_charged_data.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
namespace c=carbon::electron_condensed_research;
struct Case {unsigned node;float energy,mu,length; c::SoftLaw law;};
struct Output {float position[3],direction[3];unsigned events,mode;};
static_assert(sizeof(Output)==32);
int main(int argc,char** argv){try{
    if(argc!=8)throw std::runtime_error("usage: charged-index SHA cases samples output-prefix subsegments method(bridge|gs-spatial|gs-exact-one|gs-exact-low)");
    const unsigned parts=std::stoul(argv[6]);if(!parts||parts>64)throw std::runtime_error("subsegment count");
    const std::string method=argv[7];if(method!="bridge"&&method!="gs-spatial"&&method!="gs-exact-one"&&method!="gs-exact-low")throw std::runtime_error("method");
    const bool gs=method!="bridge";
    const unsigned exact_limit=method=="gs-exact-one"?1:method=="gs-exact-low"?4:0;
    const std::filesystem::path index=argv[1];
    if(carbon::compute_file_sha256_hex(index)!=argv[2])throw std::runtime_error("index SHA");
    std::ifstream in(index);std::string magic;unsigned nodes;in>>magic>>nodes;
    if(magic!="MAIECB01"||nodes!=82)throw std::runtime_error("index header");
    sycl::queue queue{sycl::gpu_selector_v};
    std::ifstream input(argv[3]);unsigned nc;input>>nc;unsigned samples=std::stoul(argv[4]);
    if(!nc||!samples||samples>1000000)throw std::runtime_error("case count");
    auto* cases=sycl::malloc_shared<Case>(nc,queue);
    auto* tables=sycl::malloc_shared<float>(nodes*31*38,queue);
    auto* score=sycl::malloc_shared<float>(nc,queue);
    const std::size_t count=std::size_t(nc)*samples;
    auto* output=sycl::malloc_shared<Output>(count,queue);
    if(!cases||!tables||!score||!output)throw std::bad_alloc();
    bool used[82]{};
    for(unsigned i=0;i<nc;++i){auto& x=cases[i];auto& l=x.law;
        input>>x.node>>x.energy>>x.mu>>x.length>>l.screening>>l.cutoff_q>>l.lower_row_probability>>l.mean_q>>l.second_q>>l.lower_row;
        if(!input||x.node>=nodes||l.lower_row>=30)throw std::runtime_error("case");used[x.node]=true;
    }
    for(unsigned i=0;i<nodes;++i){int section;double density;std::string file,sha;in>>section>>density>>file>>sha;
        if(!in)throw std::runtime_error("index row");if(!used[i])continue;
        carbon::ChargedResponseData data(index.parent_path()/file,sha);
        if(data.section!=section||data.density!=density)throw std::runtime_error("identity");
        for(unsigned k=0;k<31*38;++k)tables[i*31*38+k]=float(data.elastic_rows[k]);
        for(unsigned j=0;j<nc;++j)if(cases[j].node==i)cases[j].law.elastic={tables+i*31*38,float(data.mass),float(data.bc),float(data.xc2)};
    }
    for(unsigned arm=0;arm<2;++arm){
        for(unsigned i=0;i<nc;++i)score[i]=0;
        queue.parallel_for(sycl::range<1>(count),[=](sycl::id<1> id){
            const auto i=id[0];const unsigned ci=i/samples;const auto spec=cases[ci];
            std::uint64_t rng=0x37abcdef912ULL+i+arm*0x6a09e667f3bcc909ULL;
            rng=(rng^(rng>>30))*0xbf58476d1ce4e5b9ULL;rng=(rng^(rng>>27))*0x94d049bb133111ebULL;rng^=rng>>31;
            auto uniform=[&](){return carbon::electron_fp32::random_draw(rng);};
            c::Segment r;
            if(!arm)r=c::explicit_segment(spec.law,spec.mu,spec.length,{0,0,1},uniform);
            else if(parts==1)r=gs?c::gs_spatial_segment(spec.law,spec.mu,spec.length,{0,0,1},uniform,exact_limit):c::condensed_segment(spec.law,spec.mu,spec.length,{0,0,1},uniform);
            else{
                r.valid=true;r.direction={0,0,1};bool multiple=false;
                for(unsigned part=0;part<parts;++part){
                    const auto step=gs?c::gs_spatial_segment(spec.law,spec.mu/parts,spec.length/parts,r.direction,uniform,exact_limit):c::condensed_segment(spec.law,spec.mu/parts,spec.length/parts,r.direction,uniform);
                    r.valid=r.valid&&step.valid;if(!r.valid)break;
                    r.position=r.position+step.position;r.direction=step.direction;r.events+=step.events;
                    multiple=multiple||step.mode>=3;
                }
                r.mode=multiple?5:0;
            }
            output[i]={{r.position.x,r.position.y,r.position.z},{r.direction.x,r.direction.y,r.direction.z},r.events,r.valid?r.mode+1:0};
            if(r.valid){sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> sum(score[ci]);sum.fetch_add(1.f);}
        }).wait_and_throw();
        unsigned bad=0;for(unsigned i=0;i<nc;++i)if(score[i]!=samples){++bad;std::cerr<<"case="<<i<<" valid="<<score[i]<<'\n';}
        const std::string path=std::string(argv[5])+(arm?".candidate.bin":".reference.bin");
        std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(output),count*sizeof(Output));if(!out)throw std::runtime_error("output write");
        std::cout<<"{\"arm\":"<<arm<<",\"method\":\""<<method<<"\",\"subsegments\":"<<parts<<",\"cases\":"<<nc<<",\"samples_per_case\":"<<samples<<",\"invalid_cases\":"<<bad<<",\"scorer\":\"FP32\"}"<<std::endl;
        if(bad)return 1;
    }
    sycl::free(cases,queue);sycl::free(tables,queue);sycl::free(score,queue);sycl::free(output,queue);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

#include "carbon/electron_charged_data.hpp"
#include "carbon/electron_rate_bound.hpp"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv){try{
    if(argc!=3)throw std::runtime_error("usage: charged index SHA");
    const std::filesystem::path path=argv[1];if(carbon::compute_file_sha256_hex(path)!=argv[2])throw std::runtime_error("index SHA");
    std::ifstream f(path);std::string magic;unsigned n;f>>magic>>n;
    if(magic!="MAIECB01")throw std::runtime_error("index magic");
    double ratio=0;std::uint64_t pieces=0;
    unsigned rejected_bad_certificates=0;
    const auto must_reject=[&](std::vector<carbon::ChargedRatePiece> rows,float low,float high,float bound){
        try{carbon::electron_rate_bound::certify(rows,low,high,bound);}
        catch(const std::runtime_error&){++rejected_bad_certificates;return;}
        throw std::runtime_error("Invalid certificate was accepted");
    };
    const carbon::ChargedRatePiece unit{0,1,{1,1,1,1}};
    must_reject({unit},0,1,.5f); // Bound below an actual constant rate.
    must_reject({unit},0,2,6);   // Parameter domain outside rate coverage.
    must_reject({{0,.4,{1,1,1,1}},{.5,1,{1,1,1,1}}},0,1,6); // Gap.
    must_reject({{0,.5,{1,1,1,1}},{.4,1,{1,1,1,1}}},0,1,6); // Overlap.
    must_reject({{1,std::nextafter(1.,2.),{1,1,1,1}}},1,1,6); // Collapses in FP32.
    must_reject({{0,1,{NAN,1,1,1}}},0,1,6);
    for(unsigned i=0;i<n;++i){int section;double density;std::string file,sha;f>>section>>density>>file>>sha;
        carbon::ChargedResponseData d(path.parent_path()/file,sha);
        if(!f||section!=d.section||density!=d.density||float(density)<1e-6f||float(density)>1e3f)throw std::runtime_error("density domain");
        for(unsigned p=0;p<2;++p){double m=0;for(const auto& r:d.rate[p])for(double v:r.value)m=std::max(m,6*std::abs(v));
            ratio=std::max(ratio,carbon::electron_rate_bound::certify(d.rate[p],float(d.parameters.front().energy),float(d.parameters.back().energy),std::nextafter(float(m),INFINITY)));pieces+=d.rate[p].size();
        }
    }
    std::cout<<std::setprecision(17)<<"{\"passed\":true,\"nodes\":"<<n<<",\"pieces\":"<<pieces<<",\"rejected_bad_certificates\":"<<rejected_bad_certificates<<",\"max_certified_bound_to_majorant\":"<<ratio<<"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

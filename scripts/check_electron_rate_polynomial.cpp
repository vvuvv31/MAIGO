#include "carbon/electron_charged_data.hpp"
#include "carbon/electron_fp32.hpp"
#include "carbon/electron_rate_polynomial.hpp"
#include <iostream>
#include <iomanip>
struct PreparedPiece {float low{},high{},value[4]{};unsigned polynomial{};};

int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("usage: charged index");
    namespace f=carbon::electron_fp32;namespace p=carbon::electron_rate_polynomial;
    const std::filesystem::path path=argv[1];

    std::ifstream in(path);std::string magic;unsigned nodes;in>>magic>>nodes;
    if(magic!="MAIECB02")throw std::runtime_error("index magic");
    std::uint64_t pieces=0,polynomial_pieces=0,points=0,support_changes=0,significant_support_changes=0,invalid=0;
    double max_absolute_to_majorant=0,max_relative=0,max_reference_error=0,max_candidate_error=0,max_bound_ratio=0;
    for(unsigned node=0;node<nodes;++node){
        int section;double density;std::string file;in>>section>>density>>file;
        carbon::ChargedResponseData data(path.parent_path()/file);
        if(!in||section!=data.section||density!=data.density)throw std::runtime_error("node metadata");
        for(unsigned process=0;process<2;++process){
            double bound=0;for(const auto& r:data.rate[process])for(double y:r.value)bound=std::max(bound,6*std::abs(y));
            const float majorant=std::nextafter(float(bound),INFINITY);
            for(const auto& src:data.rate[process]){
                f::ChargedRatePiece raw{float(src.low),float(src.high),{float(src.value[0]),float(src.value[1]),float(src.value[2]),float(src.value[3])}};
                const auto poly=p::prepare(PreparedPiece{raw.low,raw.high,{raw.value[0],raw.value[1],raw.value[2],raw.value[3]}});++pieces;polynomial_pieces+=poly.polynomial;
                const auto upper=p::piece_upper(poly);max_bound_ratio=std::max(max_bound_ratio,double(upper)/majorant);
                f::ChargedResponseView view{};view.rate[process]=&raw;view.rate_count[process]=1;
                for(unsigned j=0;j<=4096;++j){
                    const float energy=std::fma(float(j)/4096,raw.high-raw.low,raw.low);
                    const float t=(energy-raw.low)/(raw.high-raw.low);
                    const float baseline=f::charged_raw_rate(view,process,energy),candidate=p::evaluate(poly,t);
                    constexpr long double x[]={.125L,.375L,.625L,.875L};long double exact=0;
                    for(unsigned a=0;a<4;++a){long double term=raw.value[a];for(unsigned b=0;b<4;++b)if(a!=b)term*=(static_cast<long double>(t)-x[b])/(x[a]-x[b]);exact+=term;}
                    ++points;if(!std::isfinite(baseline)||!std::isfinite(candidate)){++invalid;continue;}
                    max_absolute_to_majorant=std::max(max_absolute_to_majorant,std::abs(double(baseline)-candidate)/majorant);
                    max_reference_error=std::max(max_reference_error,double(std::abs(baseline-exact))/majorant);
                    max_candidate_error=std::max(max_candidate_error,double(std::abs(candidate-exact))/majorant);
                    if(std::abs(exact)>1e-6*majorant)max_relative=std::max(max_relative,std::abs(double(baseline)-candidate)/double(std::abs(exact)));
                    if((baseline>0)!=(candidate>0)){
                        ++support_changes;
                        if(std::max(std::abs(baseline),std::abs(candidate))>1e-6*majorant)++significant_support_changes;
                        if(support_changes<=8)std::cerr<<std::setprecision(17)<<"support node="<<node<<" process="<<process<<" energy="<<energy<<" t="<<t<<" old="<<baseline<<" new="<<candidate<<" exact="<<double(exact)<<'\n';
                    }
                }
            }
        }
    }
    const bool pass=!invalid&&!support_changes&&max_absolute_to_majorant<2e-6&&max_relative<2e-5&&max_bound_ratio<.99;
    std::cerr<<"polynomial_pieces="<<polynomial_pieces<<" original_pieces="<<pieces-polynomial_pieces<<'\n';
    std::cout<<std::setprecision(17)<<"{\"passed\":"<<(pass?"true":"false")<<",\"nodes\":"<<nodes<<",\"pieces\":"<<pieces<<",\"points\":"<<points<<",\"invalid\":"<<invalid<<",\"support_changes\":"<<support_changes<<",\"significant_support_changes\":"<<significant_support_changes<<",\"max_absolute_to_majorant\":"<<max_absolute_to_majorant<<",\"max_relative_above_1e_6_majorant\":"<<max_relative<<",\"reference_error_to_majorant\":"<<max_reference_error<<",\"candidate_error_to_majorant\":"<<max_candidate_error<<",\"polynomial_bound_ratio\":"<<max_bound_ratio<<"}\n";
    return pass?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

#include "carbon/electron_fp32.hpp"
#include "carbon/electron_charged_data.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
// Compare against the frozen actual native AlongStep moments. Inputs are
// narrowed as at the FP32 runtime boundary; failures are retained, not hidden
// by changing the reference's threshold inputs.
int main(int argc,char** argv){try{
    if(argc!=6)throw std::runtime_error("usage: charged index SHA native_directory samples output_csv");
    const std::filesystem::path path=argv[1],native=argv[3];const unsigned samples=std::stoul(argv[4]);
    if(carbon::compute_file_sha256_hex(path)!=argv[2]||!samples)throw std::runtime_error("input");
    std::ifstream in(path);std::string magic;unsigned nodes;in>>magic>>nodes;
    if(magic!="MAIECB01")throw std::runtime_error("index magic");
    namespace f=carbon::electron_fp32;
    std::ofstream out(argv[5]);out<<std::setprecision(17)<<"node,cell,energy,length,native_m1,native_m2,native_n,m1,m2,n,invalid,stops,delta_incident_fraction,se,passed\n";
    unsigned cells=0,failed=0;std::uint64_t invalid=0;double max_delta=0;
    for(unsigned node=0;node<nodes;++node){int sec;double density;std::string file,sha;in>>sec>>density>>file>>sha;
        carbon::ChargedResponseData data(path.parent_path()/file,sha);const auto v=data.view();
        std::vector<f::ChargedParameter> par;for(auto p:data.parameters)par.push_back({float(p.energy),float(p.elastic_mfp),float(p.dedx),float(p.range),{float(p.diagnostic_raw_rate[0]),float(p.diagnostic_raw_rate[1])}});
        std::vector<f::ChargedRatePiece> rates[2];for(unsigned k=0;k<2;++k)for(auto r:data.rate[k])rates[k].push_back({float(r.low),float(r.high),{float(r.value[0]),float(r.value[1]),float(r.value[2]),float(r.value[3])}});
        f::ChargedResponseView model{};model.parameters=par.data();model.parameter_count=par.size();
        for(unsigned k=0;k<2;++k){model.rate[k]=rates[k].data();model.rate_count[k]=rates[k].size();}
        const auto u=v.urban;model.urban={float(u.mass),float(u.excitation),float(u.e0),float(u.e1),float(u.e2),float(u.f1),float(u.f2)};
        model.lowest_energy=v.lowest_energy;model.linear_loss_limit=v.linear_loss_limit;model.electron_cut=v.electron_cut;
        std::ostringstream name;name<<"node"<<std::setw(2)<<std::setfill('0')<<node;
        std::ifstream csv(native/name.str()/"alongstep_checks.csv");std::string line;if(!std::getline(csv,line))throw std::runtime_error("native CSV missing");
        unsigned node_cells=0;
        while(std::getline(csv,line)){
            std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);double x[11];for(auto& a:x)row>>a;if(!row)throw std::runtime_error("native row");
            const unsigned cell=unsigned(x[0]);double sum=0,square=0;unsigned bad=0,stops=0;
            for(unsigned i=0;i<samples;++i){std::uint64_t rng=0x783abcdef123ULL+node*0x9e3779b97f4a7c15ULL+cell*747796405ULL+i*277803737ULL;
                auto uniform=[&](){return f::random_draw(rng);};
                const auto r=f::sample_charged_loss(model,float(x[1]),float(x[2]),uniform);
                if(!r.valid){++bad;continue;}stops+=r.stopped;const double deposit=r.deposit/x[1];sum+=deposit;square+=deposit*deposit;
            }
            const double mean=sum/samples,m2=square/samples,delta=mean-x[4];
            const double se=std::sqrt(std::max(0.,x[5]-x[4]*x[4])/x[3]+std::max(0.,m2-mean*mean)/samples);
            const bool pass=!bad&&std::abs(delta)<=7*se+2e-4;
            out<<node<<','<<cell<<','<<x[1]<<','<<x[2]<<','<<x[4]<<','<<x[5]<<','<<x[3]<<','<<mean<<','<<m2<<','<<samples<<','<<bad<<','<<stops<<','<<delta<<','<<se<<','<<pass<<'\n';
            ++cells;++node_cells;failed+=!pass;invalid+=bad;max_delta=std::max(max_delta,std::abs(delta));
        }
        if(node_cells!=66)throw std::runtime_error("native cell coverage");
    }
    if(!out)throw std::runtime_error("output write");
    std::cout<<std::setprecision(17)<<"{\"passed\":"<<(failed?"false":"true")<<",\"nodes\":"<<nodes<<",\"cells\":"<<cells<<",\"samples_per_cell\":"<<samples<<",\"failed_cells\":"<<failed<<",\"invalid\":"<<invalid<<",\"maximum_absolute_incident_fraction_difference\":"<<max_delta<<"}\n";
    return failed?2:0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

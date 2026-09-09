// Standalone raw-source check: stdin contains next rows, CSR offsets, child rows
// as little-endian uint64. This is not a transport accuracy test.
#include "carbon/electron_continuation_replay.hpp"
#include "carbon/electron_continuation_links.hpp"
#include "carbon/electron_continuation_sampling.hpp"
#include "carbon/electron_state_refs.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc,char** argv) {
    if(argc!=2 && argc!=3)return 2;
    std::ifstream input(argv[1],std::ios::binary);
    if(!input)return 2;
    std::vector<unsigned char> payload((std::istreambuf_iterator<char>(input)),{});
    const auto n=payload.size()/carbon::kElectronStateV4RecordBytes;
    if(!n || payload.size()%carbon::kElectronStateV4RecordBytes)return 2;
    if(argc==3 && std::string(argv[2])=="--build-links") {
        try {
            const auto links=carbon::ElectronContinuationLinks::build(payload.data(),payload.size(),
                std::numeric_limits<std::size_t>::max());
            // Compare every generated link against the independent Python CSR,
            // supplied on stdin in the same layout as the standalone check.
            std::uint64_t compared=0;
            for(const auto* array:{&links.next,&links.child_offsets,&links.child_rows}) {
                std::vector<std::uint64_t> expected(array->size());
                std::cin.read(reinterpret_cast<char*>(expected.data()),expected.size()*8);
                if(!std::cin || expected!=*array) {
                    std::cerr<<"Generated links differ from independent reference\n";return 1;
                }
                compared+=array->size();
            }
            std::uint64_t continuous=0,discrete=0;
            const auto view=links.view(payload.data(),payload.size());
            for(std::uint64_t row=0;row<n;++row) {
                const auto step=carbon::read_electron_continuation_step(payload.data(),payload.size(),row);
                if(step.pdg!=11 || step.deposited_MeV<=1e-10)continue;
                const auto energy=step.pre_energy_MeV-0.5*step.deposited_MeV;
                const auto draw=carbon::inspect_electron_energy_crossing(view,row,energy);
                if(draw.status!=carbon::ElectronCrossingStatus::ready ||
                   std::abs(draw.continuous_loss_fraction-0.5)>1e-5) {
                    std::cerr<<"Continuous crossing failed at row "<<row<<'\n';return 1;
                }
                ++continuous;
                if(draw.replay.child_energy_MeV>1e-6) {
                    const auto gap=step.post_energy_MeV+0.5*draw.replay.child_energy_MeV;
                    if(carbon::inspect_electron_energy_crossing(view,row,gap).status!=carbon::ElectronCrossingStatus::rejected) {
                        std::cerr<<"Discrete child KE was treated as continuous loss\n";return 1;
                    }
                    ++discrete;
                }
            }
            std::cout<<"source_rows="<<n<<" children="<<links.child_rows.size()
                     <<" exact_links_compared="<<compared<<" continuous_crossings="<<continuous
                     <<" discrete_gaps_rejected="<<discrete<<" failed=0\n";
            return 0;
        } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
    }
    if(argc==3) {
        std::ifstream mapping(argv[2],std::ios::binary);
        if(!mapping)return 2;
        std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(mapping)),{});
        const auto refs=carbon::ElectronStateReferenceView::parse(bytes.data(),bytes.size());
        if(!refs.valid || refs.sources!=1)return 2;
        std::uint64_t checked=0,failed=0;
        for(bool nodes:{false,true}) {
            const auto count=nodes?refs.nodes:refs.samples;
            for(std::uint64_t i=0;i<count;++i) {
                ++checked;
                const auto ref=refs.at(i,nodes);
                if(!ref.valid){++failed;continue;}
                const auto state=carbon::read_electron_continuation_step(payload.data(),payload.size(),ref.row);
                if(!state.valid || (state.pdg!=11 && state.pdg!=22)){++failed;continue;}
                if(!nodes) {
                    if(!(state.deposited_MeV>0))++failed;
                } else if(ref.post) {
                    if(ref.incoming_edge!=ref.row)++failed;
                } else if(ref.incoming_edge!=carbon::kElectronContinuationEnd) {
                    const auto parent=carbon::read_electron_continuation_step(payload.data(),payload.size(),ref.incoming_edge);
                    if(!carbon::generated_by_electron_step(parent,state))++failed;
                } else if(state.step!=1)++failed;
            }
        }
        std::cout<<"state_references="<<checked<<" failed="<<failed<<'\n';
        return failed?1:0;
    }
    // The actual target sm75/x86 host ABI is little-endian; refuse other hosts.
    const std::uint64_t one=1;
    if(*reinterpret_cast<const unsigned char*>(&one)!=1)return 2;
    std::vector<std::uint64_t> next(n),offsets(n+1);
    auto read=[](auto& values) {
        std::cin.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(values[0]));
        return static_cast<bool>(std::cin);
    };
    if(!read(next) || !read(offsets) || offsets.back()>n)return 2;
    std::vector<std::uint64_t> children(offsets.back());
    if(!read(children))return 2;
    carbon::ElectronContinuationView view{payload.data(),payload.size(),next.data(),offsets.data(),children.data(),children.size()};
    std::uint64_t checked=0,failed=0;
    for(std::uint64_t row=0;row<n;++row) {
        const auto state=carbon::read_electron_continuation_step(view.payload,view.bytes,row);
        if(!state.valid){++failed;continue;}
        if(state.pdg!=11 && state.pdg!=22)continue;
        const auto result=carbon::prepare_electron_replay_step(view,row);
        ++checked;
        if(result.status!=carbon::ElectronReplayStatus::ready) {
            if(failed<5)std::cerr<<"row "<<row<<" status "<<static_cast<int>(result.status)<<'\n';
            ++failed;
        }
    }
    std::cout<<"checked="<<checked<<" failed="<<failed<<'\n';
    return failed?1:0;
}

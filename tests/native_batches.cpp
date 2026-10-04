#include "carbon/batch_run.hpp"
#include "carbon/plan_run.hpp"
#include <fstream>
#include <iostream>
#include <set>
#include <source_location>
#include <cstring>
#include <stdexcept>
using namespace carbon;
void check(bool ok, std::source_location loc=std::source_location::current()) { if (!ok) throw std::runtime_error("native batch test failed at line "+std::to_string(loc.line())); }
template<class F> void rejects(F f) { bool threw=false;try{f();}catch(const std::exception&){threw=true;}check(threw); }
int main(int argc,char** argv) {
    if(argc!=3)throw std::runtime_error("Usage: native_batches BASE_CONFIG EMPTY_OUTPUT_ROOT");
    ImmutableInputCache inputs;
    int loads=0;
    const auto& a=inputs.get<int>("x",[&]{++loads;return 7;});
    check(&a==&inputs.get<int>("x",[&]{++loads;return 8;}));check(a==7&&loads==1);
    check(inputs.get<double>("x",[]{return 2.;})==2.);
    rejects([&]{inputs.get<int>("fail",[]()->int{throw std::runtime_error("expected");});});
    check(!inputs.contains<int>("fail"));
    RunShard base;base.config_path=argv[1];base.config=load_config(argv[1]);
    CliState cli;cli.batch_output=argv[2];cli.batch_count=3;cli.histories_per_batch=1;
    auto files=base.config.topas_spots_files;
    if(files.empty())files.push_back(base.config.topas_spots_file);
    auto plan=TopasSpotPlan::from_files(files);
    const auto total_weight=plan.total_histories();
    for(auto n:{1U,2U,3U,17U})for(auto h:{1U,2U,1193U,1983920U}) {
        cli.batch_count=n;cli.histories_per_batch=h;
        const auto batches=make_repeated_batches(base,cli,inputs);
        const auto again=make_repeated_batches(base,cli,inputs);
        std::vector<std::uint64_t> totals(plan.spots.size());std::set<std::uint64_t> seeds;
        for(std::size_t i=0;i<n;++i) {
            const auto& c=batches[i].config;
            check(c.number_of_histories==h&&c.material_electron_retain_bank);
            check(c.primary_spot_batch.back().history_end==h);
            std::uint64_t previous=0;
            for(std::size_t j=0;j<c.primary_spot_batch.size();++j) {
                const auto& e=c.primary_spot_batch[j];const auto& f=again[i].config.primary_spot_batch[j];
                check(e.history_begin==previous&&e.history_end>previous);previous=e.history_end;
                check(seeds.insert(e.random_seed).second);
                check(e.random_seed==f.random_seed&&e.history_end==f.history_end&&std::memcmp(e.floats,f.floats,sizeof(e.floats))==0);
                auto spot=(e.random_seed-c.random_seed)/1000003;
                check(spot<totals.size());totals[spot]+=e.history_end-e.history_begin;
            }
        }
        for(std::size_t s=0;s<totals.size();++s) {
            const auto exact=static_cast<long double>(plan.spots[s].number_of_histories)*n*h/total_weight;
            check(std::abs(static_cast<long double>(totals[s])-exact)<1.000001L);
        }
    }
    cli.batch_count=0;rejects([&]{make_repeated_batches(base,cli,inputs);});
    cli.batch_count=2;cli.histories_per_batch=std::numeric_limits<std::size_t>::max();
    rejects([&]{make_repeated_batches(base,cli,inputs);});
    cli.histories_per_batch=1;base.histories_overridden=true;
    rejects([&]{make_repeated_batches(base,cli,inputs);});base.histories_overridden=false;
    base.config.random_seed=std::numeric_limits<std::uint64_t>::max();
    rejects([&]{make_repeated_batches(base,cli,inputs);});
    for (const auto path : {"config/proton_water_broadbeam.yaml", "config/carbon_water_broadbeam.yaml",
                           "config/proton_ct_broadbeam.yaml", "config/carbon_ct_broadbeam.yaml"}) {
        RunShard other;other.config_path=path;other.config=load_config(path);
        cli.batch_count=3;cli.histories_per_batch=7;
        const auto batches=make_repeated_batches(other,cli,inputs);
        for (const auto& shard:batches) {
            check(shard.config.number_of_histories==7);
            if(other.config.enable_tps_source)check(shard.config.primary_spot_batch.back().history_end==7);
            if(!other.config.enable_ct_grid)check(!shard.config.material_electron_retain_bank);
        }
    }
    check(!std::filesystem::exists(cli.batch_output));
    std::cout<<"PASS: allocation, tiny/zero spot counts, deterministic disjoint seeds, cache types/hits/exceptions, overflow/conflicts, no preflight output\n";
}

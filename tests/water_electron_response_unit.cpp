#include "carbon/water_electron_response.hpp"
#include "carbon/transport_config.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc,char** argv) {
    using namespace carbon;
    auto check=[](bool condition) {if(!condition)throw std::runtime_error("Water electron unit failure");};
    if(argc==4) {
        const auto table=WaterElectronResponseTable::load(argv[1],argv[2],argv[3]);
        check(!table.channels.empty());
        for(const auto& channel:table.channels) {
            const auto draw=sample_water_electron_response((channel.low+channel.high)/2,.5,.5,
                table.channels.data(),table.samples.data(),table.heads.data(),table.channels.size());
            check(draw.valid);
        }
        check(!sample_water_electron_response(table.channels.back().high+1,.5,.5,
            table.channels.data(),table.samples.data(),table.heads.data(),table.channels.size()).valid);
        std::cout<<"Loaded and sampled "<<table.channels.size()<<" real response channels\n";
    }
    std::array<WaterElectronChannel,60> c{};
    for(int i=0;i<60;++i)c[i]={double(i*5),double((i+1)*5),.1,0,0,2};
    const WaterElectronSample s[2]={{.5,{0,0,0},{0,0,2}},{1,{0,0,1},{0,0,3}}};
    const std::uint32_t heads[2]={0,1};
    auto draw=sample_water_electron_response(300,1,.5,c.data(),s,heads);
    check(draw.valid && draw.head==1 && draw.point[2]==2);
    check(!sample_water_electron_response(300.1,.5,.5,c.data(),s,heads).valid);
    std::array<WaterElectronChannel,90> extended{};
    for(int i=0;i<90;++i)extended[i]={double(i*5),double((i+1)*5),.1,0,0,2};
    for(double energy : {300.1,350.,400.,449.99,450.})
        check(sample_water_electron_response(energy,.5,.5,extended.data(),s,heads,extended.size()).valid);
    check(!sample_water_electron_response(450.01,.5,.5,extended.data(),s,heads,extended.size()).valid);
    check(!sample_water_electron_response(100,.5,.5,c.data(),s,heads,0).valid);
    extended[61].low=307.;
    check(!sample_water_electron_response(306.,.5,.5,extended.data(),s,heads,extended.size()).valid);
    check(!sample_water_electron_response(-.01,.5,.5,c.data(),s,heads).valid);
    check(!sample_water_electron_response(100,1.01,.5,c.data(),s,heads).valid);
    check(!sample_water_electron_response(100,.5,-1,c.data(),s,heads).valid);
    check(sample_water_electron_response(100,0,0,c.data(),s,heads).point[2]==0);
    c[0].count=0;c[0].fraction=0;
    check(sample_water_electron_response(0,.5,.5,c.data(),s,heads).valid);
    WaterElectronPathNode nodes[2]={{kWaterPathNone,0,{0,0,0}},{0,0,{0,0,-2}}};
    draw.point={0,0,1};draw.head=1;
    check(water_electron_path_in_slab(draw,1,20,{0,0,1},nodes,2)==WaterElectronPathStatus::escaped);
    nodes[1].point={1000,0,0};
    check(water_electron_path_in_slab(draw,1,20,{0,0,1},nodes,2)==WaterElectronPathStatus::contained);
    nodes[1].previous=1;
    check(water_electron_path_in_slab(draw,1,20,{0,0,1},nodes,2)==WaterElectronPathStatus::invalid);
    nodes[1].previous=0;nodes[1].point={0,0,-2};
    const double radius[2]={0,2};
    check(water_electron_path_in_slab(draw,10,20,{0,0,1},nodes,2,radius)==WaterElectronPathStatus::contained);
    check(water_electron_path_in_slab(draw,1,20,{0,0,1},nodes,2,radius)==WaterElectronPathStatus::escaped);
    if(argc==2) {
        auto config=load_config(argv[1]);config.run_mode=RunMode::smoke;
        config.enable_inelastic=false;config.enable_secondary_transport=false;config.enable_let_scoring=false;
        config.initial_energy_MeVu=300;config.beam_energy_spread=0;
        config.water_electron_response_diagnostic_file="not_loaded_by_config.bin";
        config.water_electron_response_sha256=std::string(64,'a');
        config.water_electron_response_metadata_sha256=std::string(64,'b');config.validate();
        auto rejected=[&](auto change) {
            auto bad=config;change(bad);bool failed=false;
            try {bad.validate();}catch(const std::invalid_argument& e) {
                failed=std::string(e.what()).find("Water electron")!=std::string::npos;
            }
            check(failed);
        };
        rejected([](auto& c){c.run_mode=RunMode::production;});
        rejected([](auto& c){c.run_mode=RunMode::research;});
        rejected([](auto& c){c.enable_inelastic=true;});
        rejected([](auto& c){c.initial_energy_MeVu=301;});
        rejected([](auto& c){c.beam_energy_spread=.01;});
        auto spread=config;spread.initial_energy_MeVu=250;spread.beam_energy_spread=.01;spread.validate();
        auto high=spread;high.water_electron_high_energy_diagnostic=true;
        high.initial_energy_MeVu=400;high.validate();
        auto batch=spread;
        PrimarySpotBatchEntry first;first.history_end=10;first.initial_energy_MeV()=1800;first.beam_energy_spread()=.01F;
        PrimarySpotBatchEntry second=first;second.history_begin=10;second.history_end=20;second.initial_energy_MeV()=3000;
        batch.number_of_histories=20;batch.primary_spot_batch={first,second};batch.validate();
        rejected([](auto& c){PrimarySpotBatchEntry s;s.initial_energy_MeV()=3600;s.beam_energy_spread()=.01F;c.primary_spot_batch={s};});
        rejected([](auto& c){c.water_electron_response_sha256.clear();});
        rejected([](auto& c){c.water_electron_response_diagnostic_file.clear();});
        rejected([](auto& c){c.water_electron_nuclear_diagnostic=true;});
        auto nuclear=config;nuclear.water_electron_nuclear_diagnostic=true;
        nuclear.enable_inelastic=true;nuclear.enable_secondary_transport=true;nuclear.validate();
        nuclear.run_mode=RunMode::production;bool production_rejected=false;
        try {nuclear.validate();}catch(const std::invalid_argument&) {production_rejected=true;}
        check(production_rejected);
        std::cout<<"Water response configuration guard checks passed\n";
    }
    std::cout<<"Water response sampler / closed CDF / full path / physical-vs-ROI boundary checks passed\n";
}

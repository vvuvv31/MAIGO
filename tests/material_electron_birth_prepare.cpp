#include "carbon/material_electron_birth_prepare.hpp"
#include <bit>
#include <cassert>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
using namespace carbon;
int main() {
    std::size_t checks=0;
    std::mt19937_64 rng(20261001);
    for(int fixture=0;fixture<10;++fixture) {
        ElectronBirthChannel low[]={{0,1,0,0,0},{1,2,0.12,0,3},{3,4,0.3,3,2}};
        ElectronBirthChannel high[]={{0,1,0,0,0},{1,2,0.25,0,3},{3,4,0.4,3,2}};
        ElectronBirthSample samples[]={{0.1,10,0},{0.7,20,1},{1.0,30,0},{0.5,40,1},{1.0,50,0}};
        if(fixture==1)high[1].fraction=std::numeric_limits<double>::quiet_NaN();
        if(fixture==2)high[1].fraction=1.01;
        if(fixture==3)high[1].count=0;
        if(fixture==4)high[1].offset=100;
        if(fixture==5)high[1].count=100;
        if(fixture==6)high[1].fraction=0;
        if(fixture==7)high[0].count=1;
        if(fixture==8)high[1].fraction=-0.01;
        MaterialElectronResponseView tables[2]{};
        for(int j=0;j<2;++j) {tables[j].section=2;tables[j].density_g_cm3=1+j;
            tables[j].birth_channels=j?high:low;tables[j].channel_count=3;
            tables[j].birth_samples=(fixture==9&&j)?nullptr:samples;tables[j].birth_sample_count=5;}
        const std::vector<double> edge={-1,0,std::nextafter(0.,1.),0.1,0.5,0.7,std::nextafter(1.,0.),1,2,3,4,5,std::numeric_limits<double>::quiet_NaN()};
        auto check=[&](int section,double density,double energy,double mix,double u) {
            const auto expected=sample_material_electron_birth(section,density,energy,mix,u,tables,2);
            auto actual=prepare_material_electron_birth(section,density,energy,mix,u,tables,2);
            if(actual.birth.valid)actual.birth=resolve_prepared_electron_birth(actual.birth,tables[actual.table],energy,u);
            assert(expected.table==actual.table && expected.birth.valid==actual.birth.valid);
            assert(std::bit_cast<std::uint64_t>(expected.birth.fraction)==std::bit_cast<std::uint64_t>(actual.birth.fraction));
            assert(expected.birth.reference.valid==actual.birth.reference.valid);
            assert(expected.birth.reference.row==actual.birth.reference.row && expected.birth.reference.source==actual.birth.reference.source);
            auto serialized=expected.birth;
            serialized.reference.incoming_edge=rng();serialized.reference.post=(rng()&1)!=0;
            const QueuedElectronBirth queued(serialized,energy,u);
            const auto restored=queued.draw();
            assert(restored.valid==serialized.valid && restored.reference.valid==serialized.reference.valid);
            assert(std::bit_cast<std::uint64_t>(restored.fraction)==std::bit_cast<std::uint64_t>(serialized.fraction));
            assert(restored.reference.source==serialized.reference.source && restored.reference.row==serialized.reference.row);
            assert(restored.reference.incoming_edge==serialized.reference.incoming_edge && restored.reference.post==serialized.reference.post);
            assert(std::bit_cast<std::uint64_t>(queued.energy)==std::bit_cast<std::uint64_t>(energy));
            assert(std::bit_cast<std::uint64_t>(queued.uniform)==std::bit_cast<std::uint64_t>(u));
            ++checks;
        };
        for(double e:edge)for(double u:edge)for(double m:edge)for(double rho:{0.,1.,1.3,2.,3.})check(2,rho,e,m,u);
        for(int i=0;i<100000;++i)check(i%11?2:3,1+std::generate_canonical<double,53>(rng),
            4*std::generate_canonical<double,53>(rng),std::generate_canonical<double,53>(rng),std::generate_canonical<double,53>(rng));
    }
    std::cout<<checks<<" exact prepared/resolved birth checks passed\n";
}

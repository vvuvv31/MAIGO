#include "carbon/upstream_air_scattering.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>

int main() {
    char pattern[]="/tmp/maigo-air-moments-XXXXXX";
    const char* created=mkdtemp(pattern);
    if (!created) return 1;
    const std::filesystem::path root(created),file=root/"moments.csv";
    const std::string header="energy_MeVu,length_mm,position_variance_mm2,position_angle_cov_mm_rad,angle_variance_rad2\n";
    auto check=[](bool ok){if(!ok)throw std::runtime_error("air moment test failed");};
    auto reject=[&](auto f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught);};
    auto write=[&](const std::string& s){std::ofstream f(file);f<<s;};
    try {
        write(header+"100,300,1,.01,.01\n100,315,1,.01,.01\n200,300,3,.01,.01\n200,315,3,.01,.01\n");
        const auto table=carbon::AirMomentTable::read(file);
        check(table.sample(100,300).xx==1.);
        check(table.sample(200,315).xx==3.);
        check(std::abs(table.sample(150,307.5).xx-2.)<1e-12);
        reject([&]{table.sample(99.,307.);});
        reject([&]{table.sample(150.,316.);});
        reject([&]{table.sample(std::nan(""),307.);});
        write(header+"100,300,1,.01,.01\n100,315,1,.01,.01\n200,300,3,.01,.01\n200,300,3,.01,.01\n");
        reject([&]{carbon::AirMomentTable::read(file);});
        write(header+"100,300,1,2,.01\n");
        reject([&]{carbon::AirMomentTable::read(file);});
        write("wrong schema\n");reject([&]{carbon::AirMomentTable::read(file);});
        std::filesystem::remove(file);std::filesystem::remove(root);
        std::cout<<"Air moment nodes/interpolation/domain/schema/PSD rejection: PASS\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<'\n';return 1;
    }
}

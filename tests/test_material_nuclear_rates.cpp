#include "carbon/material_nuclear_rates.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <fstream>
#include <cstring>
#include <unistd.h>
#ifdef CARBON_HAS_SYCL
#include <sycl/sycl.hpp>
#endif

namespace {
void require(bool pass,const char* message) { if(!pass) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
    bool threw=false;try { f(); } catch(const std::invalid_argument&) { threw=true; }
    require(threw,"Expected fail-closed rejection");
}
void synthetic() {
    // Header-only retired schemas must fail at the version gate, not later
    // at a missing payload/metadata check. Never mutate the frozen binaries.
    char filename[]="/tmp/maigo-retired-rate-XXXXXX";
    const int fd=mkstemp(filename);
    require(fd>=0,"Cannot create isolated schema fixture");
    close(fd);
    struct Cleanup {
        const char* path;
        ~Cleanup() { std::error_code ec; std::filesystem::remove(path,ec); }
    } cleanup{filename};
    for (const unsigned version : {0U,1U,2U,4U,99U}) {
        carbon::SchneiderRateHeader primary{};
        std::memcpy(primary.magic,"SCHNRATE",8);primary.version=version;
        { std::ofstream out(filename,std::ios::binary|std::ios::trunc);
          out.write(reinterpret_cast<const char*>(&primary),sizeof(primary)); }
        bool rejected=false;
        try { (void)carbon::SchneiderRateTable::from_binary(filename); }
        catch(const std::runtime_error& e) {
            rejected=std::string(e.what()).find("requires SCHNRATE v3")!=std::string::npos;
        }
        require(rejected,"Primary retired schema did not fail at version gate");
        carbon::SecondaryRateHeader secondary{};
        std::memcpy(secondary.magic,"SCHN2RAT",8);secondary.version=version;
        { std::ofstream out(filename,std::ios::binary|std::ios::trunc);
          out.write(reinterpret_cast<const char*>(&secondary),sizeof(secondary)); }
        rejected=false;
        try { (void)carbon::SecondaryRateTable::from_binary(filename); }
        catch(const std::runtime_error& e) {
            rejected=std::string(e.what()).find("requires SCHN2RAT v3")!=std::string::npos;
        }
        require(rejected,"Secondary retired schema did not fail at version gate");
    }
    std::array<double,25> w{},r{};w[1]=.2;w[2]=.4;r[1]=.002;r[2]=.004;
    require(std::abs(carbon::recover_element_mass_rate(r,w).per_unit_mass_fraction-.01)<1.e-15,"Unmixing units");
    auto bad=r;bad[2]*=1.01;rejects([&]{carbon::recover_element_mass_rate(bad,w);});
    bad=r;bad[0]=.001;rejects([&]{carbon::recover_element_mass_rate(bad,w);});
    bad=r;bad[1]=std::numeric_limits<double>::quiet_NaN();rejects([&]{carbon::recover_element_mass_rate(bad,w);});
    bad=r;bad[1]=-1;rejects([&]{carbon::recover_element_mass_rate(bad,w);});
    w[1]=0;r[1]=0;rejects([&]{carbon::recover_element_mass_rate(r,w);});
    w[1]=.2;r.fill(0);require(carbon::recover_element_mass_rate(r,w).per_unit_mass_fraction==0,"Zero channel");
    const double values[26]={1,2,3,4}; // 13 targets, 2 energies
    carbon::MaterialRateDomain domains[13]{};domains[0]={.4,.8,true};
    carbon::MaterialNuclearRateView view{values,domains,1,2,0,1};
    require(view.evaluate(0,.3).total==0,"Mask must precede interpolation");
    require(std::abs(view.evaluate(0,.5).total-1.5)<1.e-15,"Masked interpolation");
    require(view.evaluate(0,.9).total==0,"Above domain");
    require(view.evaluate(1,.5).total==0,"Invalid projectile");
    require(view.evaluate(0,std::numeric_limits<double>::quiet_NaN()).total==0,"NaN energy");
}
void check_view(const carbon::MaterialNuclearRates& table) {
    auto view=table.view();
    for(std::size_t p=0;p<view.projectiles;++p) {
        for(std::size_t e=0;e<view.energies;++e) {
            const double energy=view.minimum+e*view.step;
            const auto rates=view.evaluate(p,energy);
            double sum=0;
            for(std::size_t t=0;t<13;++t) {
                if(t!=0 && t!=3) require(rates.partials[t]==0,"Non-H/O target in pure water");
                require(rates.partials[t]>=0 && std::isfinite(rates.partials[t]),"Invalid water partial");
                sum+=rates.partials[t];
            }
            require(sum==rates.total,"Hazard and sampler sum differ");
        }
        for(std::size_t t:{std::size_t{0},std::size_t{3}}) {
            const auto& d=table.domains()[p*13+t];
            require(view.evaluate(p,std::nextafter(d.minimum,-std::numeric_limits<double>::infinity())).partials[t]==0,"Lower domain leakage");
            require(view.evaluate(p,std::nextafter(d.maximum,std::numeric_limits<double>::infinity())).partials[t]==0,"Upper domain leakage");
        }
    }
#ifdef CARBON_HAS_SYCL
    sycl::queue q{sycl::gpu_selector_v};
    auto* data=sycl::malloc_shared<double>(table.partials().size(),q);
    auto* domains=sycl::malloc_shared<carbon::MaterialRateDomain>(table.domains().size(),q);
    const auto n=view.projectiles*view.energies;
    auto* out=sycl::malloc_shared<carbon::MaterialMaskedRates>(n,q);
    if(!data || !domains || !out) throw std::bad_alloc();
    std::copy(table.partials().begin(),table.partials().end(),data);
    std::copy(table.domains().begin(),table.domains().end(),domains);
    view.partials=data;view.domains=domains;
    q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){
        const auto i=id[0];out[i]=view.evaluate(i/view.energies,view.minimum+(i%view.energies+.37)*view.step);
    }).wait_and_throw();
    for(std::size_t i=0;i<n;++i) {
        const auto ref=table.view().evaluate(i/view.energies,view.minimum+(i%view.energies+.37)*view.step);
        for(std::size_t t=0;t<13;++t)
            require(std::abs(out[i].partials[t]-ref.partials[t])<=1.e-13*std::max(1.,ref.total),"Host/device mismatch");
    }
    sycl::free(out,q);sycl::free(domains,q);sycl::free(data,q);
    std::cout<<"GPU checked "<<n<<" off-node queries\n";
#endif
}
}
int main(int argc,char** argv) {
    try {
        synthetic();
        if(argc>1 && std::string(argv[1])=="--real-data") {
            const auto materials=carbon::SchneiderMaterialTable::from_topas_file("data/HUtoMaterialSchneider.txt");
            const auto p=carbon::SchneiderRateTable::from_binary("data/schneider/schneider_inelastic_rates_v2_1.bin");
            const auto s=carbon::SecondaryRateTable::from_binary("data/schneider/secondary_inelastic_rates_v2_1.bin");
            const auto water=carbon::PureWaterMaterial::from_probe("data/water_unified/g4_water_material.json",
                "60be17929880fe18f1758edc02350b3fa7140b817ab0d21cb75bd87dbc891f31");
            const double h=water.hydrogen_mass_fraction;
            require(water.density_g_cm3==1 && water.mean_excitation_energy_eV==78,"Probe units");
            require(std::abs(water.radiation_length_g_cm2-36.082977464022328)<1.e-12,"Probe radiation length");
            rejects([&]{carbon::PureWaterMaterial::from_probe("data/water_unified/g4_water_material.json",std::string(64,'0'));});
            const auto waterp=carbon::MaterialNuclearRates::water_primary(p,materials,h);
            const auto waters=carbon::MaterialNuclearRates::water_secondary(s,materials,h);
            require(waters.projectiles().size()==14,"Registry changed");
            for(std::size_t e=0;e<p.num_energies();++e) for(std::size_t t:{std::size_t{0},std::size_t{3}}) {
                const double w=t==0?h:1-h;
                const double expected=p.mass_partial_rate(1,t,e)/materials.sections[1].mass_fraction[t]*w;
                require(std::abs(waterp.partials()[t*p.num_energies()+e]-expected)<1.e-10*std::max(1.,expected),"Independent section reconstruction mismatch");
                for(std::size_t proj=0;proj<s.num_projectiles();++proj) {
                    const double x=s.mass_partial_rate(proj,1,t,e)/materials.sections[1].mass_fraction[t]*w;
                    require(std::abs(waters.partials()[(proj*13+t)*s.num_energies()+e]-x)<1.e-10*std::max(1.,x),"Secondary reconstruction mismatch");
                }
            }
            check_view(waterp);check_view(waters);
            rejects([&]{carbon::MaterialNuclearRates::water_primary(p,materials,0);});
            rejects([&]{carbon::MaterialNuclearRates::water_secondary(s,materials,1);});
            std::cout<<"primary elemental max relative disagreement="<<waterp.maximum_relative_disagreement()
                     <<" secondary="<<waters.maximum_relative_disagreement()<<"\n";
        }
        std::cout<<"Material rate adapter tests PASS (not transport/production validation)\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}

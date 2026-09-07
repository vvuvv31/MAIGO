#include "carbon/material_nuclear_rates.hpp"
#include <iomanip>
#include <iostream>

int main() {
    try {
        const auto materials=carbon::SchneiderMaterialTable::from_topas_file("data/HUtoMaterialSchneider.txt");
        const auto rates=carbon::SecondaryRateTable::from_binary("data/schneider/secondary_inelastic_rates_v2_1.bin");
        const auto water=carbon::PureWaterMaterial::from_probe("data/water_unified/g4_water_material.json",
            "60be17929880fe18f1758edc02350b3fa7140b817ab0d21cb75bd87dbc891f31");
        const auto table=carbon::MaterialNuclearRates::water_secondary(rates,materials,water.hydrogen_mass_fraction);
        const auto view=table.view();
        std::cout<<"Z,A,E_MeVu,total_per_mm,H_per_mm,O_per_mm,HO_in_domain\n"<<std::setprecision(17);
        for(std::size_t p=0;p<table.projectiles().size();++p) {
            const auto ion=table.projectiles()[p];
            for(int e=0;e<=400;++e) {
                const double energy=.01+e;
                const auto x=view.evaluate(p,energy);
                const auto h=table.domains()[p*13],o=table.domains()[p*13+3];
                const bool covered=h.supported && o.supported && energy>=h.minimum && energy<=h.maximum && energy>=o.minimum && energy<=o.maximum;
                std::cout<<ion.z<<','<<ion.a<<','<<energy<<','<<x.total*water.density_g_cm3<<','
                    <<x.partials[0]*water.density_g_cm3<<','<<x.partials[3]*water.density_g_cm3<<','<<covered<<'\n';
            }
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

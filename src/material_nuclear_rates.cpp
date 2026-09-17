#include "carbon/material_nuclear_rates.hpp"
#include "carbon/min_json.hpp"
#include "carbon/sha256.hpp"
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace carbon {
PureWaterMaterial PureWaterMaterial::from_probe(const std::filesystem::path& path,const std::string& expected) {
    if(expected.size()!=64 || compute_file_sha256_hex(path)!=expected)
        throw std::invalid_argument("Pure-water material SHA mismatch");
    std::ifstream in(path);
    const std::string text((std::istreambuf_iterator<char>(in)),{});
    const auto doc=minjson::Parser(text).parse();
    if(minjson::require_uint(doc.at("schema_version"),"schema")!=1 ||
       minjson::require_uint(doc.at("geant4_version_number"),"G4 version")!=1132 ||
       minjson::require_string(doc.at("material"),"material")!="G4_WATER")
        throw std::invalid_argument("Wrong water material/schema/Geant4 version");
    const auto positive=[&](const minjson::Value& value) {
        const double x=minjson::require_number(value,"water property");
        if(!std::isfinite(x) || x<=0) throw std::invalid_argument("Nonpositive water property");
        return x;
    };
    PureWaterMaterial out;
    out.density_g_cm3=positive(doc.at("density_g_cm3"));
    out.radiation_length_g_cm2=positive(doc.at("radiation_length_g_cm2"));
    out.mean_excitation_energy_eV=positive(doc.at("mean_excitation_energy_eV"));
    const auto& elements=doc.at("elements");
    if(elements.type!=minjson::Value::Type::Array || elements.arr.size()!=2 ||
       minjson::require_uint(elements.at(0).at("z"),"H Z")!=1 ||
       minjson::require_uint(elements.at(1).at("z"),"O Z")!=8)
        throw std::invalid_argument("Pure water requires H/O only, canonical order");
    const double h=positive(elements.at(0).at("mass_fraction"));
    const double o=positive(elements.at(1).at("mass_fraction"));
    const double ah=positive(elements.at(0).at("atomic_mass_g_mol"));
    const double ao=positive(elements.at(1).at("atomic_mass_g_mol"));
    if(std::abs(h+o-1)>1.e-12 || std::abs((h/ah)/(o/ao)-2)>1.e-10)
        throw std::invalid_argument("Water fractions must sum to one and have H:O atom ratio 2:1");
    out.hydrogen_mass_fraction=h;
    return out;
}
ElementRateRecovery recover_element_mass_rate(const std::array<double,25>& rates,
                                              const std::array<double,25>& weights) {
    std::size_t ref=0,present=0;
    for(std::size_t s=0;s<25;++s) {
        if(!std::isfinite(rates[s]) || rates[s]<0 || !std::isfinite(weights[s]) || weights[s]<0 || weights[s]>1)
            throw std::invalid_argument("Invalid elemental rate/composition");
        if(weights[s]>0) ++present;
        else if(rates[s]!=0) throw std::invalid_argument("Nonzero rate on absent element");
        if(weights[s]>weights[ref]) ref=s;
    }
    if(present<2) throw std::invalid_argument("Insufficient sections for H/O consistency audit");
    // Deterministic largest-weight reference, not a fitted mean.
    ElementRateRecovery out{rates[ref]/weights[ref],0};
    if(!std::isfinite(out.per_unit_mass_fraction)) throw std::invalid_argument("Elemental recovery overflow");
    for(std::size_t s=0;s<25;++s) {
        if(weights[s]==0) continue;
        const double v=rates[s]/weights[s],den=std::max(v,out.per_unit_mass_fraction);
        if(!std::isfinite(v)) throw std::invalid_argument("Elemental recovery overflow");
        const double err=den>0?std::abs(v-out.per_unit_mass_fraction)/den:0;
        out.maximum_relative_disagreement=std::max(out.maximum_relative_disagreement,err);
        // Frozen dump serialization differences are ~1e-11, not physics fits.
        if(err>1.e-8) throw std::invalid_argument("Section-dependent elemental rate: refuse water reconstruction");
    }
    return out;
}
template<class Partial,class Domain>
MaterialNuclearRates MaterialNuclearRates::build(const SchneiderMaterialTable& materials,double h,
    std::vector<SecondaryProjectileKey> projectiles,std::size_t ne,double emin,double step,
    Partial partial,Domain domain) {
    if(!std::isfinite(h) || h<=0 || h>=1) throw std::invalid_argument("Water needs explicit positive H/O mass fractions");
    for(std::size_t t=0;t<13;++t)
        if(materials.elements[t].z!=kSchneiderCanonicalZ[t]) throw std::invalid_argument("Composition target order mismatch");
    for(const auto& row:materials.sections) {
        double sum=0;
        for(double w:row.mass_fraction) {
            if(!std::isfinite(w) || w<0 || w>1) throw std::invalid_argument("Invalid source composition");
            sum+=w;
        }
        if(std::abs(sum-1)>1.e-8) throw std::invalid_argument("Source composition sum mismatch");
    }
    MaterialNuclearRates out;
    out.projectiles_=std::move(projectiles);out.energies_=ne;out.minimum_=emin;out.step_=step;
    out.partials_.assign(out.projectiles_.size()*13*ne,0);out.domains_.resize(out.projectiles_.size()*13);
    for(std::size_t p=0;p<out.projectiles_.size();++p) for(std::size_t t:{std::size_t{0},std::size_t{3}}) {
        const auto d=domain(p,t);
        out.domains_[p*13+t]={d.energy_min_mevu,d.energy_max_mevu,d.has_support!=0};
        std::array<double,25> weights{},rates{};
        for(std::size_t s=0;s<25;++s) weights[s]=materials.sections[s].mass_fraction[t];
        for(std::size_t e=0;e<ne;++e) {
            for(std::size_t s=0;s<25;++s) rates[s]=partial(p,s,t,e);
            const auto recovered=recover_element_mass_rate(rates,weights);
            out.disagreement_=std::max(out.disagreement_,recovered.maximum_relative_disagreement);
            out.partials_[(p*13+t)*ne+e]=recovered.per_unit_mass_fraction*(t==0?h:1-h);
        }
    }
    return out;
}
MaterialNuclearRates MaterialNuclearRates::water_primary(const SchneiderRateTable& src,
    const SchneiderMaterialTable& materials,double h) {
    if(src.binary_version()!=3 || !src.has_channel_domains()) throw std::invalid_argument("Water requires v3 primary domains");
    return build(materials,h,{{6,12}},src.num_energies(),src.energy_min_mevu(),src.energy_step_mevu(),
        [&](auto,auto s,auto t,auto e){return src.mass_partial_rate(s,t,e);},
        [&](auto,auto t){return src.channel_domain(t);});
}
MaterialNuclearRates MaterialNuclearRates::water_secondary(const SecondaryRateTable& src,
    const SchneiderMaterialTable& materials,double h) {
    if(src.binary_version()!=3 || src.num_projectiles()!=14 || !src.has_channel_domains())
        throw std::invalid_argument("Water requires v3 14-projectile secondary domains");
    return build(materials,h,src.projectiles(),src.num_energies(),src.energy_min_mevu(),src.energy_step_mevu(),
        [&](auto p,auto s,auto t,auto e){return src.mass_partial_rate(p,s,t,e);},
        [&](auto p,auto t){return src.channel_domain(p,t);});
}
} // namespace carbon

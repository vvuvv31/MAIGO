#pragma once
#include "carbon/ct_grid.hpp"
#include "carbon/schneider_rate_table.hpp"
#include "carbon/secondary_rate_table.hpp"
#include <array>
#include <cmath>
#include <vector>

namespace carbon {
struct PureWaterMaterial {
    double density_g_cm3{}, radiation_length_g_cm2{}, mean_excitation_energy_eV{};
    double hydrogen_mass_fraction{};
    static PureWaterMaterial from_probe(const std::filesystem::path&, const std::string& expected_sha256);
};
// Material rows have no geometry identity: water is NOT a Schneider section.
struct MaterialRateDomain { double minimum{}, maximum{}; bool supported{}; };
struct ElementRateRecovery { double per_unit_mass_fraction{}, maximum_relative_disagreement{}; };
ElementRateRecovery recover_element_mass_rate(const std::array<double,25>& partials,
                                              const std::array<double,25>& fractions);
struct MaterialMaskedRates { std::array<double,13> partials{}; double total{}; };
// Shared host/device view [projectile][target][energy]. Density is NOT stored.
struct MaterialNuclearRateView {
    const double* partials{};
    const MaterialRateDomain* domains{};
    std::size_t projectiles{}, energies{};
    double minimum{}, step{};
    MaterialMaskedRates evaluate(std::size_t p, double energy) const noexcept {
        MaterialMaskedRates out;
        if (!partials || !domains || p>=projectiles || energies<2 || !(step>0) ||
            !std::isfinite(energy) || energy<minimum || energy>minimum+(energies-1)*step) return out;
        const double pos=(energy-minimum)/step;
        auto node=static_cast<std::size_t>(pos);
        if(node>=energies-1) node=energies-2;
        const double f=pos-node;
        for(std::size_t t=0;t<13;++t) {
            const auto& d=domains[p*13+t];
            // Mask at query energy BEFORE interpolation, matching CT v3.
            if(!d.supported || energy<d.minimum || energy>d.maximum) continue;
            const auto i=(p*13+t)*energies+node;
            const double v=partials[i]+f*(partials[i+1]-partials[i]);
            out.partials[t]=v>0?v:0; out.total+=out.partials[t];
        }
        return out.total>1.e-12 ? out : MaterialMaskedRates{};
    }
};
class MaterialNuclearRates {
public:
    // Explicit TOPAS-pinned water composition required. Rounded Schneider
    // parser atomic masses must not silently stand in for exact G4_WATER.
    static MaterialNuclearRates water_primary(const SchneiderRateTable&,
        const SchneiderMaterialTable&, double hydrogen_mass_fraction,
        int projectile_z, int projectile_a);
    static MaterialNuclearRates water_secondary(const SecondaryRateTable&,
        const SchneiderMaterialTable&, double hydrogen_mass_fraction);
    MaterialNuclearRateView view() const noexcept {
        return {partials_.data(),domains_.data(),projectiles_.size(),energies_,minimum_,step_};
    }
    const auto& partials() const noexcept { return partials_; }
    const auto& domains() const noexcept { return domains_; }
    const auto& projectiles() const noexcept { return projectiles_; }
    double maximum_relative_disagreement() const noexcept { return disagreement_; }
private:
    template<class Partial,class Domain>
    static MaterialNuclearRates build(const SchneiderMaterialTable&,double,
        std::vector<SecondaryProjectileKey>,std::size_t,double,double,Partial,Domain);
    std::vector<double> partials_;
    std::vector<MaterialRateDomain> domains_;
    std::vector<SecondaryProjectileKey> projectiles_;
    std::size_t energies_{};
    double minimum_{},step_{},disagreement_{};
};
} // namespace carbon

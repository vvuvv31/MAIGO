#pragma once
#include "carbon/restricted_fluctuation_candidate.hpp"
#include "carbon/discrete_delta_candidate.hpp"
#include "carbon/copper_ion_constants.hpp"
#include <algorithm>
#include <cmath>

namespace carbon {
// Model selection is separate from source-species support. Ion parameters
// come from the active Urban model and the extracted projectile registry.
enum class CopperFluctuationModel { universal_proton, ion };
enum class CopperMeanLoss { total, restricted };
// Native fragment tables retain their extracted ion calibration, including H1.
// Primary calibration is selected from the projectile identity.
enum class CopperFluctuationContext { primary, native_ion_fragment };
struct CopperFluctuationStep {
    // Total kinetic/rest energy in MeV; length is accepted true path in mm,
    // density is g/cm^3, and effective_charge_squared is dimensionless.
    float kinetic_MeV, mass_MeV;
    int z, a;
    float effective_charge_squared, mean_MeV, length_mm, density;
    // The mean must declare whether it already includes the hard-delta mean.
    // Restricted fluctuations and hard-delta production are separate: turning
    // off restricted_fluctuations does not suppress native ion delta sampling.
    CopperMeanLoss mean_kind;
    bool restricted_fluctuations;
    CopperFluctuationContext context{CopperFluctuationContext::primary};
};

// Native Geant4 11.3.2 G4IonFluctuations::Factor in solid G4_Cu (Zeff=29).
// kinetic/mass are MeV, q2 is effective charge squared, beta2 is v^2/c^2.
// The fitted constants and evaluation order belong to the extracted model.
inline float copper_ion_dispersion_factor(float kinetic, float mass, int z, float q2, float beta2) {
    constexpr float zeff = 29.f, electron_mass = .51099891f;
    const float fermi_beta2 = 2 * copper_fermi_MeV / electron_mass;
    float s1 = .4f * (1 - beta2) / ((1 - .5f * beta2) * zeff);
    s1 *= beta2 > fermi_beta2
              ? std::log(2 * electron_mass * beta2 / copper_excitation_MeV) * fermi_beta2 / beta2
              : std::log(4 * copper_fermi_MeV / copper_excitation_MeV);
    s1 += 1;
    float energy = kinetic * 931.49410242f / mass;
    if (beta2 < 3 * (.05f / 938.272013f) * zeff) {
        const float ss =
            1 - .4861f * std::pow(energy, -.6508f) + .8712f * std::pow(energy, -.8432f);
        if (ss < .001f)
            s1 = 1000;
        else if (s1 * ss < 1)
            s1 = 1 / ss;
    }
    const float charge = static_cast<float>(z);
    const bool hadron = z == 1;
    const float factor = hadron ? 1 : charge * std::cbrt(charge / zeff);
    if (!hadron)
        energy /= charge * std::sqrt(charge * zeff);
    const float b0 = hadron ? .1955f : .01273f;
    const float b1 = hadron ? .6941f : .03458f;
    const float b2 = hadron ? 2.522f : .3951f;
    const float b3 = hadron ? 1.040f : 3.812f;
    const float y = energy * b3;
    const float x = b2 * (y <= .2f ? y * (1 - .5f * y) : 1 - std::exp(-y));
    const float offset = energy - b1;
    return s1 * q2 / (charge * charge) + factor * x * b0 / (offset * offset + x * x);
}

// Bounded Poisson component shared by proton and ion hard-delta production.
// Larger ion means are split into independent components by the caller.
struct CopperPoissonDraw {
    unsigned count{};
    bool valid{false};
};
template <class Uniform> CopperPoissonDraw copper_poisson_component(float mean, Uniform &uniform) {
    const float threshold = std::exp(-mean);
    float product = 1;
    unsigned n = 0;
    do {
        const float u = uniform();
        if (!(u >= 0 && u < 1) || ++n > 128)
            return {};
        product *= u;
    } while (product > threshold);
    return {n - 1, true};
}

template <CopperFluctuationModel Model, bool NativeMath = false, class Uniform>
RestrictedFluctuationDraw<float> sample_copper_fluctuation_model(const CopperFluctuationStep &p,
                                                                 Uniform &uniform) {
    constexpr bool ion = Model == CopperFluctuationModel::ion;
    const float kinetic = p.kinetic_MeV, mass = p.mass_MeV;
    const float length = p.length_mm, density = p.density;
    const float mean = p.mean_MeV, q2 = p.effective_charge_squared;
    if (!(kinetic > 0 && mass > 0 && mean >= 0 && length >= 0 && density > 0) ||
        !std::isfinite(kinetic + mass + mean + length + density) ||
        std::abs(density - 8.96f) > 1e-5f)
        return {};
    if constexpr (ion) {
        if (!(p.z > 0 && p.a >= p.z && q2 > 0) ||
            !std::isfinite(kinetic + mass + q2 + mean + length + density) ||
            p.mean_kind != CopperMeanLoss::restricted)
            return {};
    } else {
        if (p.z != 1 || p.a != 1 || q2 != 1.f)
            return {};
    }
    if (mean == 0 || length == 0)
        return {0, true};
    const CopperIonIdentity *identity = nullptr;
    if constexpr (ion) {
        for (const auto &item : copper_ion_identities)
            if (item.z == p.z && item.a == p.a) {
                identity = &item;
                break;
            }
        if (!identity)
            return {};
    }
    constexpr float electron_mass = .51099891f, radius = 2.8179403262e-12f;
    const float tau = kinetic / mass, ratio = electron_mass / mass;
    const float beta2 = tau * (tau + 2) / ((tau + 1) * (tau + 1));
    const float tmax =
        2 * electron_mass * tau * (tau + 2) / (1 + 2 * (tau + 1) * ratio + ratio * ratio);
    // Preserve each independently extracted projectile calibration and FP32
    // expression order; unifying the engine does not recalibrate the cut.
    const float cut = std::min(tmax, ion ? copper_cut_MeV : .16144811353831331f);
    float restricted_mean = mean, universal_variance = 0, ion_variance = 0;
    float delta_count = 0, coefficient = 0, form_factor = 0;
    if constexpr (ion) {
        // Ion preparation starts from restricted mean loss. Charge-dependent
        // dispersion corrects its variance; delta transfers are added below.
        coefficient = 6.283185307179586f * electron_mass * radius * radius *
                      copper_electron_density * length * (density / 8.96f) * q2;
        if (p.restricted_fluctuations) {
            universal_variance = (tmax / beta2 - .5f * cut) * coefficient;
            const float factor = copper_ion_dispersion_factor(kinetic, mass, p.z, q2, beta2);
            const float cut_factor =
                1 + (factor - 1) * 2 * electron_mass * beta2 / (tmax * (1 - beta2));
            ion_variance =
                universal_variance * ((cut_factor > .01f && factor > .01f) ? cut_factor : 1);
        }
    } else {
        // Proton preparation accepts either mean convention. Subtract the
        // analytic hard-delta mean before drawing restricted loss so the same
        // energy is not counted twice when explicit Cu transfers are added.
        universal_variance =
            (tmax / beta2 - .5f * cut) *
            (6.283185307179586f * electron_mass * radius * radius * copper_electron_density) *
            length * (density / 8.96f);
        ion_variance = universal_variance;
        if (cut < tmax) {
            constexpr float proton_form_factor = 2 * electron_mass / (842.6f * 842.6f);
            if (tmax * proton_form_factor > 1e-6f)
                return {};
            const float total2 = (kinetic + mass) * (kinetic + mass);
            const float factor =
                (6.283185307179586f * electron_mass * radius * radius * copper_electron_density) *
                length * (density / 8.96f) / beta2;
            const float logarithm = std::log(tmax / cut);
            delta_count = factor * (1 / cut - 1 / tmax - beta2 * logarithm / tmax +
                                    .5f * (tmax - cut) / total2);
            const float delta_mean = factor * (logarithm - beta2 * (1 - cut / tmax) +
                                               .25f * (tmax * tmax - cut * cut) / total2);
            const float total_mean =
                p.mean_kind == CopperMeanLoss::restricted ? mean + delta_mean : mean;
            if (!(delta_count >= 0 && delta_count <= 8 && delta_mean >= 0 &&
                  delta_mean <= total_mean))
                return {};
            restricted_mean = total_mean - delta_mean;
        }
    }
    RestrictedFluctuationDraw<float> draw{restricted_mean, true};
    if (p.restricted_fluctuations) {
        RestrictedFluctuationInput<float> input{kinetic,
                                                mass,
                                                restricted_mean,
                                                ion_variance,
                                                universal_variance,
                                                cut,
                                                tmax,
                                                ion ? copper_excitation_MeV : .000322f,
                                                .00001f};
        RestrictedFluctuationSampler<float, Uniform, NativeMath, NativeMath> sampler(uniform);
        draw = sampler.sample(input, ion && !(p.z == 1 && p.a == 1), static_cast<float>(p.z));
    }
    if (!draw.valid)
        return draw;
    if constexpr (ion) {
        if (cut >= tmax)
            return draw;
        const float total2 = (kinetic + mass) * (kinetic + mass);
        const float spin_term = identity->spin > 0 ? .5f * (tmax - cut) / total2 : 0;
        delta_count = coefficient / beta2 *
                      (1 / cut - 1 / tmax - beta2 * std::log(tmax / cut) / tmax + spin_term);
        if (!(delta_count >= 0 && delta_count <= 4096 && std::isfinite(delta_count)))
            return {};
        form_factor =
            2 * electron_mass * std::pow(static_cast<float>(p.z), 4.f / 7.f) / (842.6f * 842.6f);
    } else if (delta_count == 0)
        return draw;
    unsigned proposals = 0;
    if constexpr (ion) {
        // Independent bounded Poisson components sum to the same total mean.
        // Preserve their draw order: it defines the per-history RNG stream.
        while (delta_count > 0) {
            const float chunk = std::min(delta_count, 8.f);
            const auto n = copper_poisson_component(chunk, uniform);
            if (!n.valid)
                return {};
            proposals += n.count;
            delta_count -= chunk;
        }
    } else {
        const auto n = copper_poisson_component(delta_count, uniform);
        if (!n.valid)
            return {};
        proposals = n.count;
    }
    const float spin = ion ? identity->spin : .5f;
    const float magnetic_moment2 = ion ? identity->magnetic_moment2 : 0.f;
    for (unsigned i = 0; i < proposals; ++i) {
        const auto delta = sample_charged_delta_candidate(cut, tmax, beta2, form_factor, mass,
                                                          kinetic, spin, magnetic_moment2, uniform);
        if constexpr (ion)
            if (delta.status == DeltaDrawStatus::form_factor_null)
                continue;
        if (delta.status != DeltaDrawStatus::accepted)
            return {};
        draw.loss += delta.energy_MeV;
    }
    return {draw.loss, std::isfinite(draw.loss) && draw.loss >= 0};
}

// One transport-facing entry. A configuration string never selects particle
// parameters or fragment transport. Keep native math scoped to proton calls.
template <bool NativeProtonMath = false, class Uniform>
RestrictedFluctuationDraw<float> sample_copper_fluctuation(const CopperFluctuationStep &p,
                                                           Uniform &uniform) {
    if (p.context == CopperFluctuationContext::primary && p.z == 1 && p.a == 1)
        return sample_copper_fluctuation_model<CopperFluctuationModel::universal_proton,
                                               NativeProtonMath>(p, uniform);
    return sample_copper_fluctuation_model<CopperFluctuationModel::ion>(p, uniform);
}
} // namespace carbon

#pragma once
#include "carbon/unrestricted_csda.hpp"
#include "carbon/unified_em_package.hpp"
#include "carbon/energy_loss_fluctuation.hpp"
#include <algorithm>
#include <cmath>
#include <bit>
#include <limits>
#include <cstdint>
#include <sycl/sycl.hpp>
namespace carbon {
// Exponent buckets narrow only the native full-response node search.
inline constexpr unsigned unified_em_index_stride = 257;
inline std::vector<unsigned> build_unified_em_index(const UnifiedEmPackage &package) {
    std::vector<unsigned> index(package.records.size() * unified_em_index_stride);
    for (unsigned r=0; r<package.records.size(); ++r) {
        const auto& record=package.records[r]; unsigned cursor=0;
        for (unsigned bucket=0; bucket<=256; ++bucket) {
            const float edge=bucket==0 ? 0.f : bucket>=255
                ? std::numeric_limits<float>::infinity() : std::ldexp(1.f,int(bucket)-127);
            while (cursor<record.node_count && package.nodes[record.node_offset+cursor].energy<edge)
                ++cursor;
            index[r*unified_em_index_stride+bucket]=cursor;
        }
    }
    return index;
}

// One step-start full-response evaluation. Variances are native unrestricted
// electronic MeV^2/mm, not finite-production-cut soft or delta partitions.
struct UnifiedEmPointStep {
    float range{}, stopping{}, dispersion{}, universal_dispersion{}, tmax{};
    bool ion_fluctuation{};
};
struct UnifiedEmPoint {
    // Non-owning views into an uploaded package and optional per-track cache.
    // Callers must keep their owners alive. Do not checkpoint these temporary
    // views as a substitute for the track's UnifiedEmState material identity.
    const UnifiedEmRecord *record{};
    const UnifiedEmNode *nodes{};
    float density_scale{1};
    const unsigned *index{};
    const EmContinuousDescriptor *continuous{};
    const float *unrestricted_ranges{};
    float continuous_coordinate{};
    EmContinuousMaterial material{};
    void bounds(float x, unsigned &lo, unsigned &hi) const {
        if (index && x>0 && std::isfinite(x)) {
            const unsigned bucket=(std::bit_cast<unsigned>(x)>>23)&255;
            const unsigned begin=index[bucket], end=index[bucket+1];
            lo=std::min(hi,begin>0 ? begin-1 : 0);
            hi=std::min(hi,end);
        }
    }
    bool covers(float kinetic) const {
        float e = kinetic / record->a;
        return e >= nodes[0].energy && e <= nodes[record->node_count - 1].energy;
    }
    // The caller owns this interval only for the current energy evaluation.
    // Keep it out of persistent track/continuation state.
    UnifiedEmNode interpolate(float e, unsigned lo) const {
        const auto &a = nodes[lo];
        const auto &b = nodes[lo + 1];
        float w = std::clamp((e - a.energy) / (b.energy - a.energy), 0.f, 1.f);
        auto mix = [&](float x, float y) { return x + w * (y - x); };
        return {e, mix(a.full, b.full), mix(a.factor, b.factor),
                mix(a.dispersion, b.dispersion),
                mix(a.universal_dispersion, b.universal_dispersion), a.fluctuation};
    }
    UnifiedEmNode at(float e) const {
        return interpolate(e, energy_interval(e));
    }
    // Shared by transport, CSDA/Urban and LET. It is electronic-only, with
    // native effective charge already in full; no restricted/delta mean is added.
    float density_effect_difference(float kinetic) const {
        const bool alpha = record->z == 2 && record->a == 4;
        const float mass = alpha ? 3727.379f : 938.27208816f;
        const float tau = kinetic * record->ratio / mass;
        const float bg2 = tau * (tau + 2);
        if (bg2 <= 1 && material.d0 == 0 && continuous->base_d0 == 0 &&
            material.x0 > 0 && continuous->base_x0 > 0) return 0;
        const float x = .5f * std::log10(bg2);
        return em_density_effect(x, material.c, material.m, material.a, material.x0,
                                 material.x1, material.d0) -
               em_density_effect(x, continuous->base_c, continuous->base_m,
                                 continuous->base_a, continuous->base_x0,
                                 continuous->base_x1, continuous->base_d0);
    }
    float stopping_from_node(float kinetic, const UnifiedEmNode& node) const {
        const float difference = density_effect_difference(kinetic);
        if (difference == 0) return node.full * density_scale;
        const bool alpha = record->z == 2 && record->a == 4;
        const float mass = alpha ? 3727.379f : 938.27208816f;
        const float tau = kinetic * record->ratio / mass;
        const float bg2 = tau * (tau + 2), beta2 = bg2 / ((tau + 1) * (tau + 1));
        return node.full * density_scale - node.factor * material.coefficient *
                   (alpha ? 4.f : 1.f) / beta2 * difference;
    }
    float unrestricted_electronic_stopping(float kinetic) const {
        return stopping_from_node(kinetic, at(kinetic / record->a));
    }
    unsigned energy_interval(float energy_per_u) const {
        unsigned lo = 0, hi = record->node_count - 2;
        bounds(energy_per_u, lo, hi);
        while (lo < hi) {
            const auto m = (lo + hi + 1) / 2;
            if (nodes[m].energy <= energy_per_u) lo = m;
            else hi = m - 1;
        }
        return lo;
    }
    float reference_range(float kinetic, unsigned i) const {
        const float e = kinetic / record->a;
        if (e <= nodes[0].energy)
            return unrestricted_ranges[0] * std::sqrt(std::max(0.f, e / nodes[0].energy));
        const auto& a = nodes[i]; const auto& b = nodes[i+1];
        const float width = (e - a.energy) * record->a;
        const float stop = a.full + (b.full-a.full) * ((e-a.energy)/(b.energy-a.energy));
        return unrestricted_ranges[i] + linear_stopping_path(width, a.full, stop);
    }
    float reference_range(float kinetic) const {
        const float e = kinetic / record->a;
        return reference_range(kinetic, e <= nodes[0].energy ? 0 : energy_interval(e));
    }
    float reference_energy(float range) const {
        if (range <= unrestricted_ranges[0]) {
            const float w = std::max(0.f, range / unrestricted_ranges[0]);
            return nodes[0].energy * record->a * w * w;
        }
        unsigned lo = 0, hi = record->node_count - 2;
        while (lo < hi) {
            const auto m = (lo + hi + 1) / 2;
            if (unrestricted_ranges[m] <= range) lo = m;
            else hi = m - 1;
        }
        const auto& a = nodes[lo]; const auto& b = nodes[lo+1];
        const float width = (b.energy-a.energy) * record->a;
        const float step = linear_stopping_energy(range-unrestricted_ranges[lo], a.full,
                                                  (b.full-a.full)/width);
        return a.energy * record->a + std::clamp(step, 0.f, width);
    }
    // Prove that density contributes only linear scaling on [0,E]. All native
    // water/Schneider therapeutic-energy states satisfy this or identical laws.
    // Above the density-effect onset use the same continuous laws explicitly.
    bool linear_density_range(float kinetic) const {
        if (material.c == continuous->base_c && material.m == continuous->base_m &&
            material.a == continuous->base_a && material.x0 == continuous->base_x0 &&
            material.x1 == continuous->base_x1 && material.d0 == continuous->base_d0)
            return true;
        if (material.d0 != 0 || continuous->base_d0 != 0) return false;
        const bool alpha = record->z == 2 && record->a == 4;
        const float mass = alpha ? 3727.379f : 938.27208816f;
        const float tau = kinetic * record->ratio / mass, bg2 = tau * (tau+2);
        if (bg2 <= 1 && material.x0 >= 0 && continuous->base_x0 >= 0) return true;
        return .5f * std::log10(bg2) <= std::min(material.x0, continuous->base_x0);
    }
    // Above-onset density correction is rare for therapeutic water/CT ions.
    // Keep its quadrature out of each hot Urban/mean query's inline body.
    [[gnu::noinline]] float density_range_correction(float kinetic) const {
        // Integrate only the continuous density-effect residual. Native full
        // response knots and their exact linear primitive supply the baseline.
        float lower = nodes[0].energy * record->a;
        if (material.d0 == 0 && continuous->base_d0 == 0) {
            const float bg2 = std::pow(10.f, 2*std::min(material.x0, continuous->base_x0));
            const float mass = record->z == 2 && record->a == 4 ? 3727.379f : 938.27208816f;
            lower = std::max(lower, mass / record->ratio * (bg2/(std::sqrt(1+bg2)+1)));
        }
        if (kinetic <= lower) return 0;
        constexpr float x[4] = {.183434642f,.525532410f,.796666477f,.960289856f};
        constexpr float w[4] = {.362683783f,.313706646f,.222381034f,.101228536f};
        const float start = std::log(lower), span = std::log(kinetic/lower);
        const unsigned panels = static_cast<unsigned>(std::ceil(span/.125f));
        const float half = span / (2*panels);
        float correction = 0;
        for (unsigned i=0; i<panels; ++i) {
            const float center = start + (2*i+1)*half;
            for (unsigned k=0; k<4; ++k) for (int sign : {-1,1}) {
                const float energy = std::exp(center + sign*half*x[k]);
                const float base = at(energy/record->a).full*density_scale;
                const float actual = unrestricted_electronic_stopping(energy);
                correction += w[k]*half*energy*(base-actual)/(actual*base);
            }
        }
        return correction;
    }
    float unrestricted_range(float kinetic) const {
        const float reference = reference_range(kinetic)/density_scale;
        return kinetic<=0 || linear_density_range(kinetic) ? reference
            : reference+density_range_correction(kinetic);
    }
    [[gnu::noinline]] float density_corrected_energy(float range, float e) const {
        float lo = 0, hi = nodes[record->node_count-1].energy*record->a;
        for (unsigned i=0; i<32; ++i) {
            const float residual = unrestricted_range(e)-range;
            if (std::abs(residual) <= 2e-6f*range) return e;
            if (residual > 0) hi=e; else lo=e;
            const float next=e-residual*unrestricted_electronic_stopping(e);
            e = next>lo && next<hi ? next : .5f*(lo+hi);
        }
        return std::numeric_limits<float>::quiet_NaN();
    }
    float unrestricted_energy(float range) const {
        if (range<=0) return 0;
        const float e=reference_energy(range*density_scale);
        return linear_density_range(e) ? e : density_corrected_energy(range,e);
    }
    UnifiedEmPointStep prepare(float kinetic) const {
        const auto& r=*record;
        const float e=kinetic/r.a;
        const auto interval=energy_interval(e);
        const auto n=interpolate(e,interval);
        // Stopping, fluctuations and the analytic CSDA primitive use the same
        // native knots. Share their search without changing either formula.
        const float reference=reference_range(kinetic,interval)/density_scale;
        const float range=kinetic<=0 || linear_density_range(kinetic) ? reference
            : reference+density_range_correction(kinetic);
        const float tau=kinetic/r.mass, re=.51099891f/r.mass;
        const float bg2=tau*(tau+2), beta2=bg2/((tau+1)*(tau+1));
        const float tmax=2*.51099891f*bg2/(1+2*(tau+1)*re+re*re);
        // Exported Dispersion used native Tmax and the reference production
        // cut. Replace that cut term by Tmax. The ion charge-fluctuation
        // multiplier depends on Tmax, not tcut, and is already in dispersion.
        const float ratio=(tmax/beta2-.5f*tmax) /
            (tmax/beta2-.5f*std::min(tmax,continuous->base_cut));
        return {range,stopping_from_node(kinetic,n),
            n.dispersion*density_scale*ratio,n.universal_dispersion*density_scale*ratio,
            tmax,n.fluctuation==1};
    }
    float loss_step(float kinetic,const UnifiedEmPointStep& pre,float fraction) const {
        // Native final-range smoothing terminates the low-energy tail. Outside
        // that domain bound mean loss by an actual unrestricted range difference.
        const float f=record->step_fraction, x=record->final_range;
        if (pre.range<=x) return pre.range;
        const float native=f*pre.range+x*(1-f)*(2-x/pre.range);
        const float limit=pre.range-unrestricted_range(kinetic*(1-fraction));
        return limit>0 && std::isfinite(limit) ? std::min(native,limit)
            : std::numeric_limits<float>::quiet_NaN();
    }
    float mean(float kinetic, float length, const UnifiedEmPointStep& pre) const {
        if (!(length>0)) return 0;
        if (length>=pre.range) return kinetic;
        const float first=pre.stopping*length;
        if (first>record->linear_limit*kinetic)
            return std::clamp(kinetic-unrestricted_energy(pre.range-length), 0.f, kinetic);
        // Midpoint integration of total electronic stopping. Range inversion
        // handles the larger finite-loss steps and the stopping endpoint.
        const float mid=std::max(.5f*kinetic,kinetic-.5f*first);
        return std::min(kinetic,unrestricted_electronic_stopping(mid)*length);
    }

};
struct UnifiedEmDevice;
// Exact material fields evaluated on the GPU at a CT's actual density bits.
// This is a memoized polynomial evaluation, not a density response node.
struct UnifiedEmCtFields {
    EmContinuousMaterial material{};
    float coordinate{};
    bool valid{};
};
struct alignas(16) UnifiedEmState {
    const UnifiedEmDevice *tables{};
    unsigned material_index{};
    int ion{-1};
    float density{}, density_scale{1};
    int section{-2};
    bool valid{false};
    unsigned ct_fields_index{~0u};
    UnifiedEmPoint point() const;
    const UnifiedEmRecord *record() const {
        return tables ? point().record : nullptr;
    }
    bool covers(float t) const {
        return valid && point().covers(t);
    }
    UnifiedEmPointStep prepare(float t) const;
    float mean(float t, float h, const UnifiedEmPointStep &pre) const {
        return point().mean(t, h, pre);
    }

};
struct UnifiedEmDevice {
    const UnifiedEmMaterial *materials{};
    const UnifiedEmSpecies *species{};
    const UnifiedEmRecord *records{};
    const UnifiedEmNode *nodes{};
    const unsigned *energy_index{};
    const EmContinuousDescriptor *continuous_descriptors{};
    const float *continuous_scalars{};
    unsigned continuous_scalar_count{};
    const UnifiedEmCtFields *ct_fields{};
    const std::uint16_t *ct_fields_voxels{};
    unsigned ct_fields_count{};
    const float *unrestricted_ranges{};
    int species_index(unsigned z, unsigned a) const {
        for (int i = 0; i < 18; ++i)
            if (species[i].z == z && species[i].a == a)
                return i;
        return -1;
    }
    UnifiedEmPoint point(unsigned material, int ion, float density_scale,
                         unsigned ct_index = ~0u) const {
        auto *r = records + material * 18 + ion;
        UnifiedEmPoint p{r, nodes + r->node_offset, density_scale,
                         energy_index
                             ? energy_index + (material * 18 + ion) * unified_em_index_stride
                             : nullptr};

        p.continuous = continuous_descriptors + material * 18 + ion;
        const auto &d = *p.continuous;
        if (ct_index != ~0u && ct_fields) {
            const auto &cached = ct_fields[ct_index * 18 + ion];
            p.continuous_coordinate = cached.coordinate;
            p.material = cached.material;

        } else {
            const float lr = std::log(materials[material].density * density_scale);
            p.continuous_coordinate =
                d.log_max > d.log_min
                    ? std::clamp((lr - d.log_min) / (d.log_max - d.log_min), 0.f, 1.f)
                    : .5f;
            p.material =
                em_continuous_material(continuous_scalars + material * continuous_scalar_count * 8,
                                       continuous_scalar_count, p.continuous_coordinate);
        }
        p.unrestricted_ranges = unrestricted_ranges + r->node_offset;
        return p;
    }
    UnifiedEmState select(int section, float density, int ion) const {
        UnifiedEmState state;
        if (ion < 0 || ion >= 18 || section < -1 || section > 24 || !(density > 0))
            return state;

        const unsigned m = section + 1;
        const auto &d = continuous_descriptors[m * 18 + ion];
        const float lr = std::log(density);
        if (lr < d.log_min - 2e-6f || lr > d.log_max + 2e-6f)
            return state;
        state.tables = this;
        state.material_index = m;
        state.ion = ion;
        state.density_scale = density / materials[m].density;
        state.density = density;
        state.section = section;
        state.valid = true;
        return state;
    }
    UnifiedEmState select_ct(int section, float density, int ion, std::size_t voxel) const {
        if (!ct_fields || !ct_fields_voxels)
            return select(section, density, ion);
        UnifiedEmState state;
        if (ion < 0 || ion >= 18 || section < 0 || section > 24 || !(density > 0))
            return state;
        const unsigned index = ct_fields_voxels[voxel];
        if (index >= ct_fields_count || !ct_fields[index * 18 + ion].valid)
            return state;
        const unsigned m = section + 1;
        state.tables = this;
        state.material_index = m;
        state.ion = ion;
        state.density_scale = density / materials[m].density;
        state.density = density;
        state.section = section;
        state.valid = true;
        state.ct_fields_index = index;
        return state;
    }
};
inline UnifiedEmPoint UnifiedEmState::point() const {
    return tables ? tables->point(material_index, ion, density_scale, ct_fields_index)
                  : UnifiedEmPoint{};
}
// Searches are shared only inside this preparation, never cached across steps.
inline UnifiedEmPointStep UnifiedEmState::prepare(float t) const {
    return point().prepare(t);
}
struct UnifiedEmLoss {
    float loss{}, mean{}, variance{}, tmax{};
    bool valid{false};
    std::uint8_t failure_stage{};
};
// Mean, native unrestricted dispersion and the full kinematic transfer limit
// feed one Universal/Ion draw. No electron-production cut or separate hard loss.
template<bool NonIon=false,bool PreparedPoint=false,
         bool NativeNormal=(NonIon && PreparedPoint),bool NativeMath=(NativeNormal && NonIon),
         bool InverseNormal=NativeNormal,class Uniform>
UnifiedEmLoss unified_em_loss(const UnifiedEmState& state,float kinetic,float length,bool fluctuations,
        const UnifiedEmPointStep& pre,Uniform& uniform,const UnifiedEmPoint* prepared=nullptr) {
    UnifiedEmLoss out;
    const auto point=[&] { if constexpr(PreparedPoint) return *prepared; else return state.point(); }();
    const auto& record=*point.record;
    out.mean=point.mean(kinetic,length,pre);
    out.variance=(NonIon || !pre.ion_fluctuation ? pre.universal_dispersion : pre.dispersion)*length;
    out.tmax=pre.tmax;
    if (!(out.mean>=0 && out.mean<=kinetic && std::isfinite(out.mean) &&
          out.variance>=0 && std::isfinite(out.variance))) { out.failure_stage=2; return out; }
    float loss=out.mean;
    if (fluctuations && loss<kinetic) {
        EnergyLossFluctuationInput<float> input{kinetic,record.mass,out.mean,
            pre.dispersion*length,pre.universal_dispersion*length,pre.tmax,
            record.excitation,record.e0};
        EnergyLossFluctuationSampler<float,Uniform,NativeNormal,NativeMath,InverseNormal> sampler(uniform);
        const auto draw=sampler.sample(input,NonIon ? false : pre.ion_fluctuation,float(record.z));
        if (!draw.valid) { out.failure_stage=1; return out; }
        loss=std::min(kinetic,draw.loss);
    }
    if (kinetic-loss<=record.lowest_kinetic) loss=kinetic;
    out.loss=loss;
    out.valid=std::isfinite(loss) && loss>=0 && loss<=kinetic;
    if (!out.valid) out.failure_stage=4;
    return out;
}
} // namespace carbon

#pragma once
#include "carbon/delta_partition.hpp"
#include "carbon/delta_loss_sampling.hpp"
#include "carbon/unified_em_package.hpp"
#include "carbon/discrete_delta_candidate.hpp"
#include "carbon/restricted_fluctuation_candidate.hpp"
#include <algorithm>
#include <cmath>
#include <bit>
#include <limits>
#include <cstdint>
#include <sycl/sycl.hpp>
namespace carbon {
// Coarse exponent buckets only narrow the search; original knots and arithmetic stay intact.
inline constexpr unsigned unified_em_index_stride = 4 * 257;
inline std::vector<unsigned> build_unified_em_index(const UnifiedEmPackage &package) {
    std::vector<unsigned> index(package.records.size() * unified_em_index_stride);
    for (unsigned r = 0; r < package.records.size(); ++r) {
        const auto &record = package.records[r];
        for (unsigned kind = 0; kind < 4; ++kind) {
            unsigned cursor = 0, count = kind == 0 ? record.node_count : record.counts[kind - 1];
            auto lower = [&](unsigned i) {
                return kind == 0 ? package.nodes[record.node_offset + i].energy
                                 : package.segments[record.offsets[kind - 1] + i].lower;
            };
            for (unsigned bucket = 0; bucket <= 256; ++bucket) {
                const float edge = bucket == 0     ? 0.f
                                   : bucket >= 255 ? std::numeric_limits<float>::infinity()
                                                   : std::ldexp(1.f, int(bucket) - 127);
                while (cursor < count && lower(cursor) < edge)
                    ++cursor;
                index[r * unified_em_index_stride + kind * 257 + bucket] = cursor;
            }
        }
    }
    return index;
}

// Production transport uses only continuous26-composition.

// Values at one fixed step-start energy. Reuse across step selection, mean
// loss, fluctuations and the post-step delta acceptance without new searches.
struct UnifiedEmPointStep {
    // range is mm; stopping/delta_stopping are MeV/mm at the fixed start energy.
    // Keep restricted and delta moments separate until the loss estimator;
    // removing explicit electron tracking does not remove either energy term.
    float factor{}, range{}, stopping{}, q2{}, dispersion{}, universal_dispersion{};
    bool ion_fluctuation{};
    float delta_stopping{}, delta_variance{}, stopping_slope{}, delta_slope{},
        delta_partition_rate{};
};
struct UnifiedEmPoint {
    // Non-owning views into an uploaded package and optional per-track cache.
    // Callers must keep their owners alive. Do not checkpoint these temporary
    // views as a substitute for the track's UnifiedEmState material identity.
    const UnifiedEmRecord *record{};
    const UnifiedEmNode *nodes{};
    const EmCubicSegmentCandidate<float> *segments{};
    float density_scale{1};
    const unsigned *index{};
    const EmContinuousDescriptor *continuous{};
    const float *continuous_range{};
    const unsigned *continuous_lengths{};
    const float *cached_range_values{};
    unsigned continuous_degree{};
    float continuous_coordinate{};
    EmContinuousMaterial material{};
    // Integral correction is a continuous polynomial in log density, not a
    // collection of material-density transport states.
    float range_log_ratio(float energy, float *derivative = nullptr) const {
        // Interpolate between energy knots; evaluate each knot's continuous
        // density correction in Chebyshev coordinates x=2*coordinate-1.
        const auto *p = continuous_range;
        const unsigned stride = continuous_degree + 1;
        unsigned low = 0, high = continuous->range_count - 2;
        while (low < high) {
            const auto m = (low + high + 1) / 2;
            if (p[m * stride] <= energy)
                low = m;
            else
                high = m - 1;
        }
        const auto *a = p + low * stride;
        const auto *b = a + stride;
        const float w = std::clamp((energy - a[0]) / (b[0] - a[0]), 0.f, 1.f);
        const float x = 2 * continuous_coordinate - 1;
        const float left = cached_range_values
                               ? cached_range_values[low]
                               : em_chebyshev(a + 1,
                                              continuous_lengths ? continuous_lengths[low]
                                                                 : continuous_degree,
                                              x),
                    right = cached_range_values
                                ? cached_range_values[low + 1]
                                : em_chebyshev(b + 1,
                                               continuous_lengths ? continuous_lengths[low + 1]
                                                                  : continuous_degree,
                                               x);
        if (derivative)
            *derivative = energy >= a[0] && energy <= b[0] ? (right - left) / (b[0] - a[0]) : 0.f;
        return left + w * (right - left);
    }
    template <bool ZeroDensityGuard = false>
    float corrected_raw_stopping(float energy, float *derivative = nullptr) const {
        // Raw DEDX/range splines use a proton or alpha reference mass.
        const bool alpha = record->z == 2 && record->a == 4;
        const float mass = alpha ? 3727.379f : 938.27208816f;
        const bool spin = record->z == 1;
        auto correction = [&](float e) {
            const float u = e / mass, bg2 = u * (u + 2), beta = bg2 / ((u + 1) * (u + 1));
            const float re = .51099891f / mass,
                        tm = 2 * .51099891f * bg2 / (1 + 2 * (u + 1) * re + re * re);
            const float floor = .00025f * mass / 938.272013f;
            const bool bragg = e * (938.272013f / mass) <= 2;
            const float cb =
                std::min(tm, bragg ? std::max(continuous->base_cut, floor) : continuous->base_cut);
            const float ct = std::min(tm, bragg ? std::max(material.cut, floor) : material.cut);
            // For 0<bg2<=1, x=.5*log10(bg2)<=0. If both density
            // laws start strictly above zero and d0==0, both are exactly
            // zero. No threshold, curve or nonzero response is approximated.
            const bool zero_density = ZeroDensityGuard && bg2 > 0.f && bg2 <= 1.f &&
                                      material.d0 == 0.f && continuous->base_d0 == 0.f &&
                                      material.x0 > 0.f && material.x1 > 0.f &&
                                      continuous->base_x0 > 0.f && continuous->base_x1 > 0.f;
            // Coincident effective cuts make log(ct/cb), ct-cb and the
            // spin difference zero as well. Keep the derivative's +/-eps
            // evaluations separate: they each prove their own zero branch.
            if (zero_density && cb == ct && cb > 0.f)
                return 0.f;
            float dd = 0.f;
            if (!zero_density) {
                const float x = .5f * std::log10(bg2);
                dd = em_density_effect(x, material.c, material.m, material.a, material.x0,
                                       material.x1, material.d0) -
                     em_density_effect(x, continuous->base_c, continuous->base_m,
                                       continuous->base_a, continuous->base_x0, continuous->base_x1,
                                       continuous->base_d0);
            }
            const float coeff = material.coefficient * (alpha ? 4.f : 1.f) / beta;
            float v = std::log(ct / cb) - beta * (ct - cb) / tm - dd;
            if (spin)
                v += .25f * (ct * ct - cb * cb) / ((e + mass) * (e + mass));
            return coeff * v;
        };
        float base_slope = 0;
        const float result =
            raw(0, energy, derivative ? &base_slope : nullptr) * density_scale + correction(energy);
        if (derivative) {
            const float eps = energy * .001f;
            *derivative = base_slope * density_scale +
                          (correction(energy + eps) - correction(energy - eps)) / (2 * eps);
        }
        return std::max(result, 1e-30f);
    }
    float corrected_raw_range(float energy) const {
        return raw(1, energy) * std::exp(range_log_ratio(energy)) / density_scale;
    }
    float continuous_inverse(float scaled_range) const {
        // Newton solves the same restricted-range integral; never replace the
        // near-endpoint range branch with local/linear deposition.
        float e = raw(2, scaled_range * density_scale);
        const float lower = segments[record->offsets[0]].lower * .000001f;
        const auto &last = segments[record->offsets[0] + record->counts[0] - 1];
        // The measured proton CT primary cache uses two analytic updates;
        // carbon retains three. Both keep the same residual guard below.
        const unsigned iterations =
            cached_range_values ? ((record->z == 1 && record->a == 1) ? 2u : 3u) : 5u;
        for (unsigned k = 0; k < iterations; ++k) {
            if (cached_range_values) {
                float raw_slope = 0, log_slope = 0;
                const float base = raw(1, e, &raw_slope);
                const float log_ratio = range_log_ratio(e, &log_slope);
                const float scale = std::exp(log_ratio) / density_scale;
                // Differentiate the implemented spline times exponential
                // correction. Preserve its value evaluation.
                const float value = base * std::exp(log_ratio) / density_scale;
                const float slope = scale * (raw_slope + base * log_slope);
                const float residual = value - scaled_range;
                const float correction = slope > 0 && std::isfinite(slope)
                                             ? residual / slope
                                             : residual * corrected_raw_stopping(e);
                e = std::clamp(e - correction, lower, last.upper);
            } else {
                const float residual = corrected_raw_range(e) - scaled_range;
                e = std::clamp(e - residual * corrected_raw_stopping(e), lower, last.upper);
            }
        }
        // A cached analytic solve must retain a small residual. Unusual
        // states use the existing five-update evaluator instead.
        if (cached_range_values &&
            (!std::isfinite(e) || std::abs(corrected_raw_range(e) - scaled_range) >
                                      1e-6f * std::max(std::abs(scaled_range), 1e-30f))) {
            e = raw(2, scaled_range * density_scale);
            for (unsigned k = 0; k < 5; ++k) {
                const float residual = corrected_raw_range(e) - scaled_range;
                e = std::clamp(e - residual * corrected_raw_stopping(e), lower, last.upper);
            }
        }
        return e;
    }
    float continuous_rate(float kinetic, float f) const {
        // kinetic is the track's total MeV. record->ratio maps it to the
        // reference projectile energy used by the packaged loss process.
        const bool alpha = record->z == 2 && record->a == 4;
        const float mass = alpha ? 3727.379f : 938.27208816f, e = kinetic * record->ratio;
        const float u = e / mass, bg2 = u * (u + 2), beta = bg2 / ((u + 1) * (u + 1)),
                    re = .51099891f / mass;
        const float tm = 2 * .51099891f * bg2 / (1 + 2 * (u + 1) * re + re * re);
        const float floor = .00025f * mass / 938.272013f;
        const float cut =
            e * (938.272013f / mass) <= 2 ? std::max(material.cut, floor) : material.cut;
        if (tm <= cut)
            return 0;
        float den = 1 / cut - 1 / tm - beta / tm * std::log(tm / cut);
        if (record->z == 1)
            den += .5f * (tm - cut) / ((e + mass) * (e + mass));
        const float c = .5f * material.cut / .51099891f;
        float minimum = record->z == 1
                            ? mass * (c * re + std::sqrt((1 + c) * (1 + c * re * re)) - 1)
                            : mass * (std::sqrt(1 + c) - 1);
        minimum = std::max(minimum, segments[record->offsets[0]].lower);
        if (e <= minimum)
            return 0;
        const float pos =
            std::clamp(std::log(e / minimum) / std::log(continuous->rate_max_energy / minimum), 0.f,
                       1.f) *
            (continuous->rate_count - 1);
        const unsigned i = std::min(static_cast<unsigned>(pos), continuous->rate_count - 2),
                       stride = continuous_degree + 1;
        const auto *a = continuous_range +
                        (continuous->rate_offset - continuous->range_offset + i) * stride + 1;
        const float x = 2 * continuous_coordinate - 1;
        const unsigned row = continuous->rate_offset - continuous->range_offset + i;
        const float left = em_chebyshev(
                        a, continuous_lengths ? continuous_lengths[row] : continuous_degree, x),
                    right = em_chebyshev(
                        a + stride,
                        continuous_lengths ? continuous_lengths[row + 1] : continuous_degree, x);
        const float correction = std::max(0.f, left + (pos - i) * (right - left));
        return std::max(0.f,
                        f * material.coefficient * (alpha ? 4.f : 1.f) / beta * den * correction);
    }
    void continuous_moments(float kinetic, float rate, float &mean, float &variance) const {
        const float mass = record->mass, u = kinetic / mass, re = .51099891f / mass,
                    bg2 = u * (u + 2), beta = bg2 / ((u + 1) * (u + 1));
        const float tm = 2 * .51099891f * bg2 / (1 + 2 * (u + 1) * re + re * re), c = material.cut;
        mean = variance = 0;
        if (!(tm > c * 1.0000001f && rate > 0))
            return;
        const float total2 = (kinetic + mass) * (kinetic + mass);
        float den = 1 / c - 1 / tm - beta / tm * std::log(tm / c);
        float one = std::log(tm / c) - beta * (tm - c) / tm;
        float two = (tm - c) - beta * .5f * (tm * tm - c * c) / tm;
        if (record->spin > 0) {
            den += .5f * (tm - c) / total2;
            one += .25f * (tm * tm - c * c) / total2;
            two += (tm * tm * tm - c * c * c) / (6 * total2);
        }
        // Native finite-size/spin veto correction integrated with fixed log
        // Gauss-Legendre quadrature. Small correction is skipped only when the
        // native predicate (form_factor*E<=1e-6) is inactive for the full domain.
        const float max_ff = record->form_factor * tm;
        if (record->spin == 0.f && max_ff > 1e-6f && max_ff <= .001f) {
            // Integrate (1+F*E)^-2 through F^2 only on the native veto's
            // active interval. Uniform remainder <=4.01e-9 for F*Tmax<=1e-3.
            // Positive Born weights give the same relative moment bound.
            const float ff = record->form_factor;
            const float low = std::max(c, 1e-6f / ff), dt = tm - low;
            const float sq = dt * (tm + low), cube = dt * (tm * tm + tm * low + low * low);
            const float fourth = sq * (tm * tm + low * low);
            const float a1 = dt - beta * .5f * sq / tm;
            const float a2 = .5f * sq - beta * cube / (3 * tm);
            const float a3 = cube / 3 - beta * fourth / (4 * tm);
            one += -2 * ff * a1 + 3 * ff * ff * a2;
            two += -2 * ff * a2 + 3 * ff * ff * a3;
        } else if (max_ff > 1e-6f) {
            constexpr float x[8] = {-.960289856f, -.796666477f, -.525532410f, -.183434642f,
                                    .183434642f,  .525532410f,  .796666477f,  .960289856f};
            constexpr float w[8] = {.101228536f, .222381034f, .313706646f, .362683783f,
                                    .362683783f, .313706646f, .222381034f, .101228536f};
            const float half = .5f * std::log(tm / c), mid = std::log(c) + half;
            float a = 0, b = 0;
            for (unsigned k = 0; k < 8; ++k) {
                const float e = std::exp(mid + half * x[k]);
                const float f1 = record->spin > 0 ? .5f * e * e / total2 : 0.f,
                            shape = 1 - beta * e / tm + f1, ff = record->form_factor * e;
                float accept = 1;
                if (ff > 1e-6f) {
                    accept = 1 / ((1 + ff) * (1 + ff));
                    if (record->spin > 0) {
                        const float x2 = .5f * .51099891f * e / (mass * mass);
                        accept *= 1 + record->magnetic_moment2 * (x2 - f1 / shape) / (1 + x2);
                    }
                    accept = std::clamp(accept, 0.f, 1.f);
                }
                a += w[k] * shape * (accept - 1) * half;
                b += w[k] * shape * (accept - 1) * half * e;
            }
            one += a;
            two += b;
        }
        mean = std::max(0.f, rate * one / den);
        variance = std::max(0.f, rate * two / den);
    }
    void bounds(unsigned kind, float x, unsigned &lo, unsigned &hi) const {
        {
            if (index && x > 0 && std::isfinite(x)) {
                unsigned bucket = (std::bit_cast<unsigned>(x) >> 23) & 255;
                const auto *v = index + kind * 257;
                const unsigned begin = v[bucket];
                const unsigned end = v[bucket + 1];
                lo = std::min(hi, begin > 0 ? begin - 1 : 0);
                hi = std::min(hi, end);
            }
        }
    }
    bool covers(float kinetic) const {
        float e = kinetic / record->a;
        return e >= nodes[0].energy && e <= nodes[record->node_count - 1].energy;
    }
    UnifiedEmNode at(float e) const {
        unsigned lo = 0, hi = record->node_count - 2;
        bounds(0, e, lo, hi);
        while (lo < hi) {
            auto m = (lo + hi + 1) / 2;
            if (nodes[m].energy <= e)
                lo = m;
            else
                hi = m - 1;
        }
        const auto &a = nodes[lo];
        const auto &b = nodes[lo + 1];
        float w = std::clamp((e - a.energy) / (b.energy - a.energy), 0.f, 1.f);
        auto mix = [&](float x, float y) { return x + w * (y - x); };
        return {e,
                mix(a.full, b.full),
                mix(a.restricted, b.restricted),
                mix(a.stopping, b.stopping),
                mix(a.range, b.range),
                mix(a.lambda, b.lambda),
                mix(a.factor, b.factor),
                mix(a.correction, b.correction),
                mix(a.dispersion, b.dispersion),
                mix(a.universal_dispersion, b.universal_dispersion),
                mix(a.q2, b.q2),
                a.model,
                a.fluctuation};
    }
    float raw(int kind, float x, float *slope = nullptr) const {
        const auto *v = segments + record->offsets[kind];
        unsigned lo = 0, hi = record->counts[kind] - 1;
        if (x < v[0].lower) {
            float w = std::max(0.f, x / v[0].lower);
            if (slope)
                *slope = x > 0 ? v[0].y0 * (kind == 2 ? 2 * w / v[0].lower
                                                      : 1 / (2 * std::sqrt(w) * v[0].lower))
                               : 0.f;
            return v[0].y0 * (kind == 2 ? w * w : std::sqrt(w));
        }
        bounds(kind + 1, x, lo, hi);
        while (lo < hi) {
            auto m = (lo + hi + 1) / 2;
            if (v[m].lower <= x)
                lo = m;
            else
                hi = m - 1;
        }
        if (slope)
            *slope = v[lo].derivative(x);
        return v[lo].value(x);
    }
    float rate(float kinetic, float pre_factor) const {
        return continuous_rate(kinetic, pre_factor);
    }
    template <bool ZeroDensityGuard = false, bool FirstOrderDelta = false,
              bool FirstOrderRestricted = false>
    UnifiedEmPointStep prepare(float kinetic) const {
        const auto &r = *record;
        float delta = 0, variance = 0, delta_slope = 0;
        const auto n = at(kinetic / r.a);
        // Eligible local-electron CT paths may use step-start restricted
        // stopping. Range inversion and charge-dependent native corrections
        // remain unchanged; only the within-step linear slope is omitted.
        float slope = 0;
        const float f = n.factor,
                    stop = f * corrected_raw_stopping<ZeroDensityGuard>(
                                   kinetic * r.ratio, FirstOrderRestricted ? nullptr : &slope);
        const float rate = continuous_rate(kinetic, f);
        continuous_moments(kinetic, rate, delta, variance);
        // Eligible local-electron CT ion paths may use the step-start
        // delta mean. This first-order approximation retains the current
        // rate, variance and fluctuations, but omits this slope correction.
        delta_slope = 0.f;
        if constexpr (!FirstOrderDelta) {
            const float eps = .001f * kinetic;
            float dl = 0, dh = 0, v = 0;
            continuous_moments(kinetic - eps, continuous_rate(kinetic - eps, f), dl, v);
            continuous_moments(kinetic + eps, continuous_rate(kinetic + eps, f), dh, v);
            delta_slope = (dh - dl) / (2 * eps);
        }
        const float u = kinetic / r.mass, re = .51099891f / r.mass, bg2 = u * (u + 2),
                    beta = bg2 / ((u + 1) * (u + 1));
        const float tm = 2 * .51099891f * bg2 / (1 + 2 * (u + 1) * re + re * re);
        const float ratio = (tm / beta - .5f * std::min(tm, material.cut)) /
                            (tm / beta - .5f * std::min(tm, continuous->base_cut));
        return {f,
                corrected_raw_range(kinetic * r.ratio) / (f * r.ratio),
                stop,
                n.q2,
                n.dispersion * density_scale * ratio,
                n.universal_dispersion * density_scale * ratio,
                n.fluctuation == 1,
                delta,
                variance,
                f * r.ratio * slope,
                delta_slope,
                rate};
    }
    template <bool NonIon = false>
    float mean(float kinetic, float length, const UnifiedEmPointStep &pre) const {
        const auto &r = *record;
        const float f = pre.factor;
        // Correct only the native linear restricted branch. Range inversion
        // already integrates energy dependence and must not be corrected twice.
        float stopping = pre.stopping;
        if (length < pre.range && stopping * length <= r.linear_limit * kinetic) {
            const float fraction = delta_partition_fraction(pre.delta_partition_rate * length);
            const float corrected = stopping - .5f * length *
                                                   (stopping * fraction + pre.delta_stopping) *
                                                   pre.stopping_slope;
            if (corrected > 0 && corrected * length <= r.linear_limit * kinetic)
                stopping = corrected;
        }
        return primary_restricted_mean_candidate(
                   kinetic, length, stopping, pre.range,
                   [&](float x) { return continuous_inverse(x * f * r.ratio) / r.ratio; },
                   [&](float mid, float loss, float h) {
                       if constexpr (NonIon)
                           return loss;
                       else {
                           if (!r.is_ion)
                               return loss;
                           mid = std::max(.5f * kinetic, mid - .5f * pre.delta_stopping * h);
                           const auto m = at(mid / r.a);
                           if (mid * (938.272013f / r.mass) <= 2.f) {
                               if (r.z > 2)
                                   return (m.stopping + m.correction) * density_scale * h;
                               return loss * m.q2 / std::max(pre.q2, 1e-30f);
                           }
                           return loss + m.correction * density_scale * h;
                       }
                   },
                   r.linear_limit)
            .energy_loss;
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
    template <bool NonIon = false>
    float mean(float t, float h, const UnifiedEmPointStep &pre) const {
        return point().template mean<NonIon>(t, h, pre);
    }
    float cut() const {
        return point().material.cut;
    }
    float peak() const {
        const auto *a = record();
        return a->peak * a->a;
    }
    float step_from_range(float r) const {
        const auto *rec = record();
        float f = rec->step_fraction, x = rec->final_range;
        return r > x ? f * r + x * (1 - f) * (2 - x / r) : r;
    }
    float step(const UnifiedEmPointStep &pre) const {
        return step_from_range(pre.range);
    }
};
struct UnifiedEmDevice {
    const UnifiedEmMaterial *materials{};
    const UnifiedEmSpecies *species{};
    const UnifiedEmRecord *records{};
    const UnifiedEmNode *nodes{};
    const EmCubicSegmentCandidate<float> *segments{};
    const unsigned *energy_index{};
    const EmContinuousDescriptor *continuous_descriptors{};
    const float *continuous_scalars{};
    const float *continuous_ranges{};
    unsigned continuous_degree{}, continuous_scalar_count{};
    const unsigned *continuous_lengths{};
    const UnifiedEmCtFields *ct_fields{};
    const std::uint16_t *ct_fields_voxels{};
    unsigned ct_fields_count{};
    const float *ct_range_values{};
    const unsigned *ct_range_offsets{};
    int ct_range_ion{-1};
    int species_index(unsigned z, unsigned a) const {
        for (int i = 0; i < 18; ++i)
            if (species[i].z == z && species[i].a == a)
                return i;
        return -1;
    }
    UnifiedEmPoint point(unsigned material, int ion, float density_scale,
                         unsigned ct_index = ~0u) const {
        auto *r = records + material * 18 + ion;
        UnifiedEmPoint p{r, nodes + r->node_offset, segments, density_scale,
                         energy_index
                             ? energy_index + (material * 18 + ion) * unified_em_index_stride
                             : nullptr};

        p.continuous = continuous_descriptors + material * 18 + ion;
        const auto &d = *p.continuous;
        if (ct_index != ~0u && ct_fields) {
            const auto &cached = ct_fields[ct_index * 18 + ion];
            p.continuous_coordinate = cached.coordinate;
            p.material = cached.material;
            if (ct_range_values && ion == ct_range_ion)
                p.cached_range_values = ct_range_values + ct_range_offsets[ct_index];
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
        p.continuous_lengths = continuous_lengths ? continuous_lengths + d.range_offset : nullptr;
        p.continuous_degree = continuous_degree;
        p.continuous_range = continuous_ranges + d.range_offset * (continuous_degree + 1);
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
struct UnifiedEmClock {
    DiscreteDeltaClockCandidate<float> clock;
    float threshold = std::numeric_limits<float>::max(), rate{}, last_density{};
    int last_section = -2;
    float update(const UnifiedEmState &s, float kinetic, float f0) {
        if (last_section != s.section || last_density != s.density) {
            threshold = std::numeric_limits<float>::max();
            last_section = s.section;
            last_density = s.density;
        }
        float peak = s.peak();
        if (kinetic <= peak) {
            if (kinetic * 1.25f < threshold) {
                rate = s.point().rate(kinetic, f0);
                threshold = (rate + rate) > 0 ? kinetic : 0;
            }
        } else if (kinetic < threshold) {
            threshold = std::max(peak, kinetic * .8f);
            rate = s.point().rate(threshold, f0);
        }
        return rate;
    }
};
struct UnifiedEmLoss {
    float loss{}, continuous{}, delta{};
    bool valid{false};
    unsigned proposed{}, accepted{};
    std::uint8_t failure_stage{0};
    float mean{}, delta_mean{}, delta_variance{};
};
// PreparedPoint reuses continuous26 views in dedicated CT broadbeam steps.
// The supplied view belongs to the same step-start state as pre; it must not
// survive a density/material change. Other paths keep their original evaluator.
template <bool NonIon = false, bool PreparedPoint = false,
          bool NativeNormal = (NonIon && PreparedPoint), bool NativeMath = (NativeNormal && NonIon),
          bool InverseNormal = NativeNormal, class Uniform>
UnifiedEmLoss unified_em_explicit_loss(const UnifiedEmState &s, float t, float h, bool fluctuations,
                                       const UnifiedEmPointStep &pre, Uniform &uniform,
                                       const UnifiedEmPoint *prepared_point = nullptr) {
    UnifiedEmLoss out;
    const auto lo = [&]() {
        if constexpr (PreparedPoint)
            return *prepared_point;
        else
            return s.point();
    }();
    const auto &r = *lo.record;
    const float mean = [&]() {
        if constexpr (PreparedPoint)
            return lo.template mean<NonIon>(t, h, pre);
        else
            return s.template mean<NonIon>(t, h, pre);
    }();
    float tau = t / r.mass, ratio = .51099891f / r.mass;
    out.mean = mean;
    float tmax = 2 * .51099891f * tau * (tau + 2) / (1 + 2 * (tau + 1) * ratio + ratio * ratio);
    const float material_cut = [&]() {
        if constexpr (PreparedPoint)
            return lo.material.cut;
        else
            return s.cut();
    }();
    float cut = std::min(material_cut, tmax);
    float loss = mean;
    if (fluctuations && mean < t) {
        const auto &fluct = pre;
        RestrictedFluctuationInput<float> input{
            t,    r.mass,       mean, fluct.dispersion * h, fluct.universal_dispersion * h, cut,
            tmax, r.excitation, r.e0};
        // Dense explicit-tail Gamma is scoped to the measured local-electron
        // proton CT broadbeam primary. Carbon and other paths retain Poisson marks.
        constexpr bool dense_tail_gamma = NonIon && PreparedPoint && NativeNormal;
        RestrictedFluctuationSampler<float, Uniform, NativeNormal, NativeMath, InverseNormal,
                                     dense_tail_gamma>
            sampler(uniform);
        auto draw = sampler.sample(input, NonIon ? false : fluct.ion_fluctuation, float(r.z));
        if (!draw.valid) {
            out.failure_stage = 1;
            return out;
        }
        loss = std::min(t, draw.loss);
    }
    if (t - loss <= r.lowest_kinetic)
        loss = t;
    out.continuous = loss;
    // Aggregate delta draw; native restricted fluctuation remains independent.
    const float delta0 = pre.delta_stopping;
    const float delta_slope = pre.delta_slope;
    const float delta_mean = std::max(0.f, delta0 - .5f * (mean + delta0 * h) * delta_slope) * h;
    out.delta_mean = delta_mean;
    if (!(delta_mean >= 0 && std::isfinite(delta_mean))) {
        out.failure_stage = 2;
        return out;
    }
    // Keep the existing non-negative variance guard before sampling.
    const float delta_variance = std::max(0.f, pre.delta_variance * h);
    out.delta_variance = delta_variance;
    const float delta = fluctuations ? delta_moment_draw<NativeNormal, NativeMath, InverseNormal>(
                                           delta_mean, delta_variance, uniform)
                                     : delta_mean;
    if (!(delta >= 0 && std::isfinite(delta))) {
        out.failure_stage = 3;
        return out;
    }
    out.delta = std::min(t - loss, delta);
    loss += out.delta;
    out.loss = loss;
    out.valid = std::isfinite(loss) && loss >= 0 && loss <= t;
    if (!out.valid)
        out.failure_stage = 4;
    return out;
}
template <bool NonIon = false, class Uniform>
UnifiedEmLoss unified_em_loss(const UnifiedEmState &s, float t, float h, bool fluctuations,
                              const UnifiedEmPointStep &pre, Uniform &uniform) {
    return unified_em_explicit_loss<NonIon>(s, t, h, fluctuations, pre, uniform);
}
} // namespace carbon

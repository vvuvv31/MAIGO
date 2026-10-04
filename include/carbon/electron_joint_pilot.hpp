#pragma once
#include "carbon/electron_joint_sampling.hpp"
#include "carbon/electron_packet_transport.hpp"

namespace carbon {
struct ElectronJointPilotView {
    electron_joint_research::View model{};
    const electron_joint_research::V* pre{};
    int section{1}; double density{},radius{.05};
    const ElectronJointPilotView* sections{};
    unsigned section_count{}; bool density_scaling{};
};
// Check the whole possible charged displacement support BEFORE drawing an
// outcome. Outcome-dependent rejection would bias the retained population.
inline bool joint_homogeneous(const ElectronCtGeometry& g,const std::array<double,3>& p,
                              int section,double density,double radius) {
    std::int64_t lo[3],hi[3];
    for(int k=0;k<3;++k) {
        lo[k]=static_cast<std::int64_t>(std::floor((p[k]-radius-g.origin[k])/g.spacing[k]));
        hi[k]=static_cast<std::int64_t>(std::floor((p[k]+radius-g.origin[k])/g.spacing[k]));
        if(lo[k]<0 || hi[k]>=g.bins[k])return false;
    }
    for(auto z=lo[2];z<=hi[2];++z)for(auto y=lo[1];y<=hi[1];++y)for(auto x=lo[0];x<=hi[0];++x) {
        const auto i=(z*g.bins[1]+y)*g.bins[0]+x;
        if(g.section[i]!=section || g.density[i]!=static_cast<float>(density))return false;
    }
    return true;
}
// Counters: eligible, started, advances, fallback, below-table terminal draws,
// deposited, escaped, photon gap, invalid. W stays with the original packet.
template<class Uniform,class Tally,class Score=NoExpectedScore>
inline ElectronEnergyPacket transport_joint_pilot(ElectronEnergyPacket packet,
    ElectronJointPilotView pilot,const MaterialElectronResponseView& legacy,
    const ElectronCtGeometry& geometry,Uniform& uniform,Tally& tally,bool independent_photons=false,bool independent_charged=false,
    Score&& score=Score{},double expected_density=-1.,double expected_min_energy=0.) {
    using namespace electron_joint_research;
    if(pilot.sections) {
        if(legacy.section<0 || unsigned(legacy.section)>=pilot.section_count)return packet;
        pilot=pilot.sections[legacy.section];
    }
    const double scale=pilot.density_scaling?pilot.density/legacy.density_g_cm3:1.;
    // The bank uses the lowest indexed density per composition. Unsupported
    // nodes never enlarge the source support or silently extrapolate.
    if(!(scale>0 && scale<=1.0000001))return packet;
    const double radius=pilot.radius*scale;
    const auto m=pilot.model;
    if(!m.steps || packet.status!=ElectronPacketStatus::active ||
       legacy.section!=pilot.section || (!pilot.density_scaling && legacy.density_g_cm3!=pilot.density) ||
       packet.cursor.energy_MeV>m.edges[m.bins]*(1+1e-7))return packet;
    tally(0,1);
    if(!joint_homogeneous(geometry,packet.cursor.position,pilot.section,
                          packet.cursor.physical_density_g_cm3,radius))return packet;
    tally(1,1);
    auto p=V{packet.cursor.position[0],packet.cursor.position[1],packet.cursor.position[2]};
    auto d=V{packet.cursor.direction[0],packet.cursor.direction[1],packet.cursor.direction[2]};
    double energy=packet.cursor.energy_MeV;
    int cursor=-1,pdg=11;
    Frame original{{1,0,0},{0,1,0},{0,0,1}},target=original;
    auto state=[&]() {
        packet.cursor.position={p.x,p.y,p.z};packet.cursor.direction={d.x,d.y,d.z};
        packet.cursor.energy_MeV=energy;packet.cursor.pdg=pdg;
    };
    auto invalid=[&]() {state();packet.status=ElectronPacketStatus::invalid;tally(8,1);return packet;};
    const unsigned iteration_cap=pilot.density_scaling?65536:4096;
    for(unsigned iteration=0;iteration<iteration_cap;++iteration) {
        const bool exact=cursor>=0;
        const GraphStep* g=nullptr;const SampledStep* s=nullptr;Frame sample_frame{};
        if(exact) {
            if(std::uint32_t(cursor)>=m.graph_count)return invalid();
            g=m.graph+cursor;pdg=g->pdg;
            d=rotate(target,project(original,pilot.pre[cursor]));
            if(pdg==22 && independent_photons) {
                state();packet.cursor.row=kElectronContinuationEnd;
                return packet; // physical photon state, no native source flight replay
            }
            if(!g->inside) {
                if(pdg!=22)return invalid();
                // The native source sphere ended. This is missing photon
                // coverage, not evidence that it escaped the actual patient.
                state();packet.status=ElectronPacketStatus::coverage_missing;
                packet.gap=ElectronPacketGap::photon_continuation;tally(7,1);return packet;
            }
        }else pdg=11;
        if(pdg==11 && (!joint_homogeneous(geometry,{p.x,p.y,p.z},pilot.section,
              packet.cursor.physical_density_g_cm3,radius) || energy>m.edges[m.bins]*(1+1e-7))) {
            state();tally(3,1);
            if(independent_charged){packet.cursor.row=kElectronContinuationEnd;return packet;}
            return resume_electron_packet(packet,legacy,uniform,packet.cursor.physical_density_g_cm3);
        }
        V delta;const double* f;
        if(exact) {delta=rotate(target,project(original,g->delta));f=g->fraction;}
        else {
            if(!(energy>0))return invalid();
            std::uint32_t a=0,b=m.bins;
            while(a<b){const auto mid=a+(b-a)/2;if(m.edges[mid+1]<=energy)a=mid+1;else b=mid;}
            const auto bin=m.nearest[a<m.bins?a:m.bins-1];
            auto lo=m.offsets[bin],hi=m.offsets[bin+1];
            const auto coin=uniform();
            while(lo+1<hi){const auto mid=lo+(hi-lo)/2;if(coin<m.steps[mid-1].cdf)hi=mid;else lo=mid;}
            s=m.steps+lo;
            if(energy<m.edges[0]) {tally(4,1);if(s->fraction[0]<1-1e-7)return invalid();}
            sample_frame=azimuth(d,6.2831853071795864769*uniform());
            delta=rotate(sample_frame,s->delta);f=s->fraction;
        }
        delta=scale*delta;
        if(pdg==11 && ((exact?g->length:s->length)*scale>.250001 || dot(delta,delta)>radius*radius))return invalid();
        const auto endpoint=p+delta;
        if(pdg==22) {
            const auto boundary=first_electron_ct_boundary(geometry,{p.x,p.y,p.z},{endpoint.x,endpoint.y,endpoint.z});
            if(boundary.status==ElectronCtBoundaryStatus::invalid)return invalid();
            if(boundary.status!=ElectronCtBoundaryStatus::contained) {
                p={boundary.position[0],boundary.position[1],boundary.position[2]};state();
                if(boundary.status==ElectronCtBoundaryStatus::escaped){packet.status=ElectronPacketStatus::escaped;tally(6,1);}
                else {packet.status=ElectronPacketStatus::coverage_missing;packet.gap=ElectronPacketGap::photon_continuation;tally(7,1);}
                return packet;
            }
        }
        ++packet.advances;tally(2,1);
        const auto coin=uniform();
        if(pdg==11 && f[0]<1 && packet.cursor.physical_density_g_cm3<=expected_density && energy>=expected_min_energy) {
            // Expected-value estimator of the energy-packet branch: score the
            // conditional mean W*f0 at a uniform chord point, keep W(1-f0) and
            // select a surviving branch with probability f[c]/(1-f0).
            if(f[0]>0) {
                const double along=uniform();const auto at=p+along*delta;
                const double w=packet.weight_MeV,rest=w-w*f[0];
                score(std::array<double,3>{at.x,at.y,at.z},w-rest);packet.weight_MeV=rest;
            }
            const double x=(f[1]+f[2]+f[3])*coin;
            unsigned channel=1;double cumulative=f[1];
            while(channel<3 && (x>=cumulative || !(f[channel]>0))){++channel;cumulative+=f[channel];}
            while(channel>1 && !(f[channel]>0))--channel;
            p=endpoint;
            if(!(f[channel]>0))return invalid();
            energy*=f[channel];
            if(channel==1) {
                if(exact) {
                    cursor=g->next;
                    if(cursor==-2)d=rotate(target,project(original,g->postdir));
                    else if(cursor<0)return invalid();
                }else d=rotate(sample_frame,s->post);
            }else {
                cursor=exact?g->child[channel-2]:s->child[channel-2];
                if(cursor<0 || std::uint32_t(cursor)>=m.graph_count)return invalid();
                if(m.graph[cursor].pdg==22)++packet.photon_children;else ++packet.electron_children;
                if(!exact){original=basis(s->source);target=sample_frame;}
            }
            continue;
        }
        if(coin<f[0]) {
            p=p+(pdg==11?uniform():1.)*delta;
            packet.deposit_position={p.x,p.y,p.z};packet.status=ElectronPacketStatus::deposited;
            state();tally(5,1);return packet;
        }
        p=endpoint;
        unsigned channel=1;double cumulative=f[0]+f[1];
        while(channel<3 && coin>=cumulative)cumulative+=f[++channel];
        if(!(f[channel]>0))return invalid();
        energy*=f[channel];
        if(channel==1) {
            if(exact) {
                cursor=g->next;
                if(cursor==-2){if(pdg!=11)return invalid();d=rotate(target,project(original,g->postdir));}
                else if(cursor<0)return invalid();
            }else d=rotate(sample_frame,s->post);
        }else {
            cursor=exact?g->child[channel-2]:s->child[channel-2];
            if(cursor<0 || std::uint32_t(cursor)>=m.graph_count)return invalid();
            if(m.graph[cursor].pdg==22)++packet.photon_children;else ++packet.electron_children;
            if(!exact){original=basis(s->source);target=sample_frame;}
        }
    }
    return invalid(); // iteration cap is a failure, never local deposition
}
// This overload needs only the physical material identity; it never binds a
// legacy continuation record on leaving homogeneous joint support.
template<class Uniform,class Tally,class Score=NoExpectedScore>
inline ElectronEnergyPacket transport_joint_independent(ElectronEnergyPacket packet,
    ElectronJointPilotView pilot,const ElectronCtGeometry& geometry,Uniform& uniform,Tally& tally,
    Score&& score=Score{},double expected_density=-1.,double expected_min_energy=0.){
    MaterialElectronResponseView identity{};identity.section=packet.cursor.section;
    identity.density_g_cm3=packet.cursor.physical_density_g_cm3;
    return transport_joint_pilot(packet,pilot,identity,geometry,uniform,tally,true,true,score,expected_density,expected_min_energy);
}
} // namespace carbon

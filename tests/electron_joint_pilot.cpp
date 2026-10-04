#include "carbon/electron_joint_pilot.hpp"
#include <cassert>
#include <iostream>
using namespace carbon;
int main(){
    float rho[125];std::uint8_t section[125];
    for(int i=0;i<125;++i){rho[i]=.3f;section[i]=1;}
    ElectronCtGeometry geom{{5,5,5},{0,0,0},{.1,.1,.1},rho,section};
    assert(joint_homogeneous(geom,{.25,.25,.25},1,.3,.05));
    rho[(2*5+2)*5+3]=.4f;
    assert(!joint_homogeneous(geom,{.275,.25,.25},1,.3,.05));
    rho[(2*5+2)*5+3]=.3f;
    assert(!joint_homogeneous(geom,{.01,.25,.25},1,.3,.05));
    using namespace electron_joint_research;
    SampledStep step{};step.cdf=1;step.delta={0,0,.04};step.length=.04;step.fraction[0]=1;step.post={0,0,1};step.source={0,0,1};
    double edges[]={.0001,.65};std::uint32_t offsets[]={0,1},nearest[]={0};
    GraphStep graph{};V pre{0,0,1};
    ElectronJointPilotView pilot{{&step,&graph,edges,offsets,nearest,1,0,1,1,.4},&pre,1,.3,.05};
    MaterialElectronResponseView legacy{};legacy.section=1;legacy.density_g_cm3=.3;
    ElectronEnergyPacket packet;packet.status=ElectronPacketStatus::active;packet.weight_MeV=7;
    packet.cursor.valid=true;packet.cursor.position={.25,.25,.25};packet.cursor.direction={0,0,1};
    packet.cursor.energy_MeV=.4;packet.cursor.physical_density_g_cm3=.3;packet.cursor.pdg=11;
    std::uint64_t counts[9]{};auto tally=[&](int i,std::uint64_t n){counts[i]+=n;};
    unsigned draws=0;auto uniform=[&](){++draws;return .5;};
    auto r=transport_joint_pilot(packet,pilot,legacy,geom,uniform,tally);
    assert(r.status==ElectronPacketStatus::deposited && r.weight_MeV==7 && r.advances==1);
    assert(std::abs(r.deposit_position[2]-.27)<1e-12 && counts[1]==1);
    // Two densities of the same composition share one source model. Dispatch
    // must preserve W and physical density, and scale the complete geometry.
    ElectronJointPilotView models[25]{};models[1]=pilot;models[1].density_scaling=true;
    ElectronJointPilotView bank{};bank.sections=models;bank.section_count=25;
    legacy.density_g_cm3=.6;
    r=transport_joint_pilot(packet,bank,legacy,geom,uniform,tally);
    assert(r.status==ElectronPacketStatus::deposited && r.weight_MeV==7);
    assert(std::abs(r.deposit_position[2]-.26)<1e-12 && r.cursor.physical_density_g_cm3==.3);
    legacy.section=2;draws=0;
    r=transport_joint_pilot(packet,bank,legacy,geom,uniform,tally);
    assert(r.status==ElectronPacketStatus::active && draws==0);
    legacy.section=1;legacy.density_g_cm3=.1;draws=0;
    r=transport_joint_pilot(packet,bank,legacy,geom,uniform,tally);
    assert(r.status==ElectronPacketStatus::active && draws==0); // no unsupported enlargement
    legacy.density_g_cm3=.3;
    draws=0;auto outside=packet;outside.cursor.position[0]=.01;
    r=transport_joint_pilot(outside,pilot,legacy,geom,uniform,tally);
    assert(r.status==ElectronPacketStatus::active && draws==0 && r.cursor.position==outside.cursor.position);
    // Photon is an exact native graph: patient exit is escape, never dose.
    step.fraction[0]=0;step.fraction[2]=1;step.child[0]=0;
    graph.pdg=22;graph.inside=1;graph.delta={0,0,1};graph.length=1;graph.fraction[0]=1;
    r=transport_joint_pilot(packet,pilot,legacy,geom,uniform,tally);
    assert(r.status==ElectronPacketStatus::escaped && r.weight_MeV==7 && counts[6]==1);
    // Scaling applies to photon flights as well as charged displacement.
    legacy.density_g_cm3=3.;
    r=transport_joint_pilot(packet,bank,legacy,geom,uniform,tally);
    assert(r.status==ElectronPacketStatus::deposited);
    assert(std::abs(r.deposit_position[2]-.354)<1e-12);
    legacy.density_g_cm3=.3;
    graph.inside=0;
    r=transport_joint_pilot(packet,pilot,legacy,geom,uniform,tally,true);
    assert(r.status==ElectronPacketStatus::active && r.cursor.pdg==22 && r.weight_MeV==7);
    assert(std::abs(r.cursor.position[2]-.29)<1e-12 && r.cursor.row==kElectronContinuationEnd);
    assert(r.advances==1); // source exhaustion never affects independent photons
    r=transport_joint_pilot(packet,pilot,legacy,geom,uniform,tally);
    assert(r.status==ElectronPacketStatus::coverage_missing && r.weight_MeV==7);
    graph.inside=1;
    section[(4*5+2)*5+2]=2;
    r=transport_joint_pilot(packet,pilot,legacy,geom,uniform,tally);
    assert(r.status==ElectronPacketStatus::coverage_missing && r.gap==ElectronPacketGap::photon_continuation);
    // Force repeated continuation until the geometry gate switches paths.
    // Legacy binding must preserve the current physical state, not the birth.
    section[(4*5+2)*5+2]=1;
    step.fraction[0]=.5;step.fraction[1]=.5;step.fraction[2]=0;
    ElectronContinuationStep native{};native.valid=true;native.pdg=11;
    native.pre_energy_MeV=.5;native.deposited_MeV=.5;
    native.pre_direction=native.post_direction={0,0,1};native.post_position={0,0,.1};native.length_mm=.1;
    std::uint64_t next[]={kElectronContinuationEnd},child_offsets[]={0,0},rows[]={0},offset[]={0,1};
    double crossing_edges[]={.000001,.5},child_energy[]={0};
    MaterialElectronStateSource source{};source.bytes=kElectronStateV4RecordBytes;source.decoded=&native;
    source.next_rows=next;source.child_offsets=child_offsets;source.certified_child_energy=child_energy;
    source.crossings={crossing_edges,offset,rows,1,1};
    legacy.state_sources=&source;legacy.state_source_count=1;
    auto continuation_coin=[](){return .75;};
    r=transport_joint_pilot(packet,pilot,legacy,geom,continuation_coin,tally);
    assert(r.status==ElectronPacketStatus::active && counts[3]==1 && r.weight_MeV==7);
    assert(r.cursor.position[2]>.4 && r.cursor.energy_MeV<.1);
    assert(r.cursor.direction[2]==1 && r.cursor.physical_density_g_cm3==.3);
    assert(std::abs(r.cursor.energy_MeV-.4*std::pow(.5,double(r.advances)))<1e-14);
    auto advanced=advance_electron_continuation(r.cursor,legacy,geom);
    assert(advanced.status!=ElectronAdvanceStatus::invalid);
    // Independent handoff has no legacy source/reference arrays at all.
    packet.cursor.section=1;
    auto independent=transport_joint_independent(packet,pilot,geom,continuation_coin,tally);
    assert(independent.status==ElectronPacketStatus::active && independent.weight_MeV==7);
    assert(independent.cursor.row==kElectronContinuationEnd);
    assert(independent.cursor.position==r.cursor.position && independent.cursor.energy_MeV==r.cursor.energy_MeV);
    assert(independent.cursor.direction==r.cursor.direction);
    std::cout<<"joint pilot geometry, unchanged unsupported birth, FP32 packet weight, deposition, photon escape/gap PASS\n";
}

#include "carbon/electron_short_range.hpp"
#include <cassert>
#include <cstring>

int main() {
    unsigned char raw[312]{};
    auto integer=[&](int offset,std::int32_t value){std::memcpy(raw+offset,&value,4);};
    auto real=[&](int offset,double value){std::memcpy(raw+offset,&value,8);};
    integer(8,1);integer(16,11);integer(20,1);integer(144,4);integer(308,2);
    real(48,.01);real(80,.05);real(104,.01);real(112,.01);
    real(136,1);real(232,1);real(256,1);real(280,.05);real(288,1);
    std::uint64_t next[]={carbon::kElectronContinuationEnd},offsets[]={0,0};
    carbon::MaterialElectronStateSource source{raw,312,next,offsets,nullptr,0};
    carbon::MaterialElectronResponseView table;table.section=8;table.density_g_cm3=1;
    table.state_sources=&source;table.state_source_count=1;
    float rho[]={1};std::uint8_t section[]={8};
    carbon::ElectronCtGeometry geometry{{1,1,1},{0,0,0},{1,1,1},rho,section};
    carbon::ElectronEnergyPacket packet;packet.status=carbon::ElectronPacketStatus::active;
    packet.weight_MeV=100; // Deliberately not the electron KE; must not define range.
    auto& c=packet.cursor;c.valid=true;c.row=0;c.section=8;c.density_g_cm3=1;
    c.energy_MeV=.01;c.position={.5,.5,.5};c.direction={1,0,0};
    assert(carbon::electron_short_range_contained(packet,table,geometry,.1));
    assert(!carbon::electron_short_range_contained(packet,table,geometry,0));
    assert(!carbon::electron_short_range_contained(packet,table,geometry,.025));
    c.position[0]=.99;
    assert(!carbon::electron_short_range_contained(packet,table,geometry,.1));
    c.position[0]=.5;
    const double cached=carbon::electron_short_range_length(source.continuation(),0);
    assert(cached==.05);
    // Sweep voxel-face clearance and the strict range boundary; the cache
    // must not turn a scanning rejection into an acceptance.
    for(double threshold : {0.0,.025,.05,.05000001,.1,.25}) {
        for(double x : {0.0,.001,.04999999,.05,.05000001,.5,.95,.999,1.0}) {
            c.position[0]=x;
            source.short_range_lengths=nullptr;
            const bool scanned=carbon::electron_short_range_contained(packet,table,geometry,threshold);
            source.short_range_lengths=&cached;
            assert(scanned==carbon::electron_short_range_contained(packet,table,geometry,threshold));
        }
    }
    c.position[0]=.5;
    source.short_range_lengths=&cached;
    assert(carbon::electron_short_range_contained(packet,table,geometry,.1));
    assert(!carbon::electron_short_range_contained(packet,table,geometry,.025));
    c.position[0]=.99;
    assert(!carbon::electron_short_range_contained(packet,table,geometry,.1));
    c.position[0]=.5;
    source.short_range_lengths=nullptr;
    real(120,.001);real(104,.009); // Finite source, not a stopped electron.
    assert(carbon::electron_short_range_length(source.continuation(),0)<0);
    assert(!carbon::electron_short_range_contained(packet,table,geometry,.1));
    real(120,0);real(104,.01);integer(300,-1);
    assert(!carbon::electron_short_range_contained(packet,table,geometry,.1));
}

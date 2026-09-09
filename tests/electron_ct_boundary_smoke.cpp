#include "carbon/electron_ct_boundary.hpp"
#include "carbon/material_electron_replay.hpp"
#include "carbon/electron_continuation_transport.hpp"
#include <cassert>
#include <cmath>

int main() {
    carbon::MaterialElectronFrame frame;
    assert(frame.valid());
    frame.origin={1,2,3};frame.axes={{{0,1,0},{-1,0,0},{0,0,1}}};
    assert(frame.valid());
    assert((frame.world({2,3,4})==std::array<double,3>{-2,4,7}));
    frame.axes[2][2]=-1;assert(!frame.valid()); // Reject reflections.
    frame.axes[2][2]=2;assert(!frame.valid()); // Reject hidden path scaling.
    carbon::MaterialElectronFrame from,to;
    assert(carbon::electron_direction_basis({0,0,1},from));
    assert(carbon::electron_direction_basis({1,0,0},to));
    const auto rotation=carbon::electron_frame_rotation(from,to);
    assert(rotation.valid());
    assert((rotation.vector({0,0,1})==std::array<double,3>{1,0,0}));
    float density[]={1,1,0.99F};
    std::uint8_t section[]={2,2,2};
    carbon::ElectronCtGeometry grid{{3,1,1},{0,0,0},{1,1,1},density,section};
    using S=carbon::ElectronCtBoundaryStatus;
    // Patient failure: float subtraction rounded a point BELOW y=3.75 onto
    // the face, selecting the next voxel's density. Double ownership does not.
    std::array<float,607> patient_rho{};patient_rho.fill(1);
    std::array<std::uint8_t,607> patient_sec{};patient_sec.fill(1);
    patient_rho[310]=0.39880782F;patient_rho[311]=0.37407282F;
    carbon::ElectronCtGeometry patient{{1,607,1},{-0.5,-151.75,0},{1,0.5,100},patient_rho.data(),patient_sec.data()};
    const std::array<double,3> birth_point{0,3.7499963928537983,63.381995350502947};
    assert((static_cast<float>(birth_point[1])+151.75F)/0.5F==311);
    const auto actual=carbon::electron_ct_point_material(patient,birth_point,{0,1,0});
    assert(actual.valid && actual.density==patient_rho[310]);
    const auto ownership=carbon::first_electron_ct_boundary(patient,birth_point,birth_point);
    assert(ownership.status==S::contained && ownership.from_density==actual.density);
    assert(carbon::electron_ct_point_material(patient,{0,3.75,63},{0,1,0}).density==patient_rho[311]);
    assert(carbon::electron_ct_point_material(patient,{0,3.75,63},{0,-1,0}).density==patient_rho[310]);
    assert(!carbon::electron_ct_point_material(patient,{0,-152,63},{0,1,0}).valid);
    auto hit=carbon::first_electron_ct_boundary(grid,{0.5,0.5,0.5},{2.5,0.5,0.5});
    assert(hit.status==S::material_change && hit.position[0]==2 && hit.fraction==0.75);
    hit=carbon::first_electron_ct_boundary(grid,{2.5,0.5,0.5},{0.5,0.5,0.5});
    assert(hit.status==S::material_change && hit.position[0]==2 && hit.fraction==0.25);
    hit=carbon::first_electron_ct_boundary(grid,{2,0.5,0.5},{0.5,0.5,0.5});
    assert(hit.status==S::contained && hit.from_density==1);
    hit=carbon::first_electron_ct_boundary(grid,{0.5,0.5,0.5},{2,0.5,0.5});
    assert(hit.status==S::material_change && hit.fraction==1);
    density[2]=1;section[2]=3;
    hit=carbon::first_electron_ct_boundary(grid,{0.5,0.5,0.5},{2.5,0.5,0.5});
    assert(hit.status==S::material_change && hit.to_section==3);
    section[2]=2;
    hit=carbon::first_electron_ct_boundary(grid,{0.5,0.5,0.5},{3.5,0.5,0.5});
    assert(hit.status==S::escaped && hit.position[0]==3);
    float square_density[]={1,2,2,1};std::uint8_t square_section[]={2,3,3,2};
    carbon::ElectronCtGeometry square{{2,2,1},{0,0,0},{1,1,1},square_density,square_section};
    hit=carbon::first_electron_ct_boundary(square,{0.5,0.5,0.5},{1.5,1.5,0.5});
    assert(hit.status==S::contained); // Corner crossing does not visit zero-length side cells.
    const auto rejected=carbon::locate_material_electron_boundary({}, {},square,{});
    assert(rejected.path_status==carbon::MaterialElectronPathVisit::invalid);
    // One synthetic physical step checks clipping and finite-source exhaustion.
    std::array<unsigned char,312> raw{};
    const auto integer=[&](unsigned offset,std::int32_t value){std::memcpy(raw.data()+offset,&value,4);};
    const auto real=[&](unsigned offset,double value){std::memcpy(raw.data()+offset,&value,8);};
    integer(8,1);integer(16,11);integer(20,1);integer(144,4);
    real(48,10);real(56,0.5);real(64,0.5);real(72,0.5);
    real(80,2.5);real(88,0.5);real(96,0.5);
    real(104,4);real(112,10);real(120,6);real(136,1);
    real(232,1);real(256,1);real(280,2);real(288,1);
    const std::uint64_t next[]={carbon::kElectronContinuationEnd},offsets[]={0,0};
    carbon::MaterialElectronStateSource source{raw.data(),raw.size(),next,offsets,nullptr,0};
    carbon::MaterialElectronResponseView table;table.section=2;table.density_g_cm3=1;
    table.state_sources=&source;table.state_source_count=1;
    carbon::ElectronContinuationCursor cursor;cursor.valid=true;cursor.row=0;cursor.energy_MeV=10;
    cursor.position={0.5,0.5,0.5};cursor.direction={1,0,0};cursor.section=2;cursor.density_g_cm3=1;
    density[2]=0.99F;
    const auto cut=carbon::advance_electron_continuation(cursor,table,grid);
    assert(cut.status==carbon::ElectronAdvanceStatus::material_boundary);
    assert(cut.deposited_MeV==3 && cut.next.energy_MeV==7 && cut.energy_residual_MeV==0);
    assert(cut.children_begin==cut.children_end);
    density[2]=1;
    const auto exhausted=carbon::advance_electron_continuation(cursor,table,grid);
    assert(exhausted.status==carbon::ElectronAdvanceStatus::source_exhausted);
    assert(exhausted.deposited_MeV==4 && exhausted.next.energy_MeV==6); // Never absorb leftover KE.
    density[2]=0.99F;real(80,2);real(104,10);real(120,0);real(280,1.5);
    const auto endpoint=carbon::advance_electron_continuation(cursor,table,grid);
    assert(endpoint.status==carbon::ElectronAdvanceStatus::stopped);
    assert(endpoint.deposited_MeV==10 && endpoint.next.energy_MeV==0);
}

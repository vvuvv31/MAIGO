// Synthetic host check for electron birth binding + energy-weighted lineage.
// No raw packages, no GPU, no production kernel. Uses only hand-filled V4
// records following the existing smoke-test encoding pattern.
#include "carbon/electron_packet_transport.hpp"
#include "carbon/electron_continuation_crossings.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {
constexpr std::size_t kRec = 312;
void set_int(unsigned char* base, unsigned off, std::int32_t v) { std::memcpy(base + off, &v, 4); }
void set_real(unsigned char* base, unsigned off, double v) { std::memcpy(base + off, &v, 8); }
std::array<unsigned char, kRec> blank_row() {
    std::array<unsigned char, kRec> r{};
    set_int(r.data(), 144, 4);
    set_real(r.data(), 136, 1.0);
    set_real(r.data(), 288, 1.0);
    return r;
}
double dot3(const std::array<double,3>& a, const std::array<double,3>& b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
double norm3(const std::array<double,3>& a) { return std::sqrt(dot3(a,a)); }
} // namespace

int main() {
    using namespace carbon;
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    // Continuous density: same-section brackets, fraction-weighted birth law.
    ElectronBirthChannel density_channels[]={{0,450,0.2,0,1},{0,450,0.6,0,1}};
    ElectronBirthSample density_samples[]={{1,7,0},{1,9,0}};
    MaterialElectronResponseView density_tables[2]{};
    for(int i=0;i<2;++i) {
        auto& t=density_tables[i];t.section=8;t.density_g_cm3=1+i;
        t.birth_channels=&density_channels[i];t.channel_count=1;
        t.birth_samples=&density_samples[i];t.birth_sample_count=1;
    }
    const auto bracket=material_electron_density_bracket(8,1.5,density_tables,2);
    assert(bracket.valid && bracket.upper_weight==0.5);
    auto mixed=sample_material_electron_birth(8,1.5,200,0.24,0.5,density_tables,2);
    assert(mixed.birth.valid && mixed.table==0 && std::abs(mixed.birth.fraction-0.4)<1e-14);
    mixed=sample_material_electron_birth(8,1.5,200,0.26,0.5,density_tables,2);
    assert(mixed.birth.valid && mixed.table==1 && mixed.birth.reference.row==9);
    assert(!material_electron_density_bracket(7,1.5,density_tables,2).valid);
    assert(!material_electron_density_bracket(8,0.99,density_tables,2).valid);
    assert(!material_electron_density_bracket(8,2.01,density_tables,2).valid);
    assert(!material_electron_density_bracket(8,qnan,density_tables,2).valid);
    mixed=sample_material_electron_birth(8,1,200,0.99,0.5,density_tables,2);
    assert(mixed.birth.valid && mixed.table==0 && mixed.birth.fraction==0.2);

    // ---- 1. birth binding: cone preserved, anchor, energy ----
    std::array<unsigned char,kRec> birth_raw = blank_row();
    set_int(birth_raw.data(),0,1);set_int(birth_raw.data(),4,1);
    set_int(birth_raw.data(),8,10);set_int(birth_raw.data(),12,5);
    set_int(birth_raw.data(),16,11);set_int(birth_raw.data(),20,1);
    set_int(birth_raw.data(),148,1);set_int(birth_raw.data(),196,3);
    set_real(birth_raw.data(),48,2.5);
    set_real(birth_raw.data(),56,1.0);set_real(birth_raw.data(),64,2.0);set_real(birth_raw.data(),72,3.0);
    set_real(birth_raw.data(),80,1.5);set_real(birth_raw.data(),88,2.0);set_real(birth_raw.data(),96,3.0);
    set_real(birth_raw.data(),104,0.5);set_real(birth_raw.data(),112,2.5);set_real(birth_raw.data(),120,2.0);
    set_real(birth_raw.data(),160,0.0);set_real(birth_raw.data(),168,0.0);set_real(birth_raw.data(),176,1.0);
    const double inv = 0.7071067811865475;
    set_real(birth_raw.data(),232,inv);set_real(birth_raw.data(),240,0.0);set_real(birth_raw.data(),248,inv);
    set_real(birth_raw.data(),256,1.0);set_real(birth_raw.data(),264,0.0);set_real(birth_raw.data(),272,0.0);
    set_real(birth_raw.data(),280,0.5);
    const std::uint64_t bnext[]={kElectronContinuationEnd}, boff[]={0,0};
    MaterialElectronStateSource bsrc{birth_raw.data(),birth_raw.size(),bnext,boff,nullptr,0};
    MaterialElectronResponseView btable{};btable.section=2;btable.density_g_cm3=1.0;
    btable.state_sources=&bsrc;btable.state_source_count=1;
    ElectronBirthDraw draw{};draw.valid=true;draw.fraction=0.4;
    draw.reference.valid=true;draw.reference.source=0;draw.reference.row=0;
    const std::array<double,3> world{5.0,6.0,7.0};
    auto cursor = bind_electron_birth(draw,btable,world,{0,1,0},0.25);
    assert(cursor.valid && cursor.pdg==11 && cursor.energy_MeV==2.5);
    assert(cursor.position==world && cursor.section==2 && cursor.density_g_cm3==1.0);
    assert(cursor.source==0 && cursor.row==0 && cursor.fraction==0);
    // Recorded cone: parent(0,0,1) vs electron(inv,0,inv) -> cos = inv.
    const std::array<double,3> carbon{0,1,0};
    assert(std::abs(dot3(carbon,cursor.direction)-inv)<1e-9);
    // Parent maps exactly to carbon; anchor maps pre position to world.
    const std::array<double,3> mapped = cursor.raw_to_world.vector({0,0,1});
    assert(std::abs(mapped[0])<1e-9 && std::abs(mapped[1]-1)<1e-9 && std::abs(mapped[2])<1e-9);
    const auto anchored = cursor.raw_to_world.world({1.0,2.0,3.0});
    for(unsigned k=0;k<3;++k)assert(std::abs(anchored[k]-world[k])<1e-9);
    assert(std::abs(norm3(cursor.direction)-1)<1e-12);
    // Zero-transfer births never launch.
    { ElectronBirthDraw z=draw;z.fraction=0;assert(!bind_electron_birth(z,btable,world,{0,1,0},0.25).valid); }
    { auto zero=birth_raw;set_real(zero.data(),48,0.0);set_real(zero.data(),112,0.0);
      MaterialElectronStateSource zs{zero.data(),zero.size(),bnext,boff,nullptr,0};
      MaterialElectronResponseView zt=btable;zt.state_sources=&zs;
      assert(!bind_electron_birth(draw,zt,world,{0,1,0},0.25).valid); }
    // Invalid draws / uniforms / references rejected.
    { ElectronBirthDraw bad=draw;bad.valid=false;assert(!bind_electron_birth(bad,btable,world,{0,1,0},0.25).valid); }
    { assert(!bind_electron_birth(draw,btable,world,{0,1,0},1.0).valid);
      assert(!bind_electron_birth(draw,btable,world,{0,1,0},-0.1).valid);
      assert(!bind_electron_birth(draw,btable,world,{0,1,0},qnan).valid);
      assert(!bind_electron_birth(draw,btable,world,{0,0,0},0.25).valid);
      ElectronBirthDraw oob=draw;oob.reference.source=7;
      assert(!bind_electron_birth(oob,btable,world,{0,1,0},0.25).valid);
      ElectronBirthDraw post=draw;post.reference.post=true;
      assert(!bind_electron_birth(post,btable,world,{0,1,0},0.25).valid); }
    { auto wrong=birth_raw;set_int(wrong.data(),16,22);
      MaterialElectronStateSource ws{wrong.data(),wrong.size(),bnext,boff,nullptr,0};
      MaterialElectronResponseView wt=btable;wt.state_sources=&ws;
      assert(!bind_electron_birth(draw,wt,world,{0,1,0},0.25).valid); }
    { auto mismatch=birth_raw;set_real(mismatch.data(),48,1.0); // birth != pre
      MaterialElectronStateSource ms{mismatch.data(),mismatch.size(),bnext,boff,nullptr,0};
      MaterialElectronResponseView mt=btable;mt.state_sources=&ms;
      assert(!bind_electron_birth(draw,mt,world,{0,1,0},0.25).valid); }

    // ---- 2. contained lineage: deposit / parent / e-child / photon ----
    std::vector<unsigned char> payload(4*kRec);
    auto row = [&](unsigned i){ return payload.data()+i*kRec; };
    for(unsigned i=0;i<4;++i){ auto b=blank_row();std::memcpy(row(i),b.data(),kRec); }
    // Row0 parent electron: E=10 = 2 (dep) + 5 (next) + 2 + 1 (children).
    set_int(row(0),0,1);set_int(row(0),4,1);set_int(row(0),8,7);set_int(row(0),12,6);
    set_int(row(0),16,11);set_int(row(0),20,2);set_int(row(0),148,1);set_int(row(0),196,1);
    set_real(row(0),48,10.0);set_real(row(0),104,2.0);set_real(row(0),112,10.0);set_real(row(0),120,5.0);
    set_real(row(0),56,0.0);set_real(row(0),64,0.0);set_real(row(0),72,0.0);
    set_real(row(0),80,1.0);set_real(row(0),88,0.0);set_real(row(0),96,0.0);
    set_real(row(0),232,1.0);set_real(row(0),256,1.0);set_real(row(0),280,1.0);
    // Row1 next: pre 5 == post of row0, same track step+1.
    set_int(row(1),0,1);set_int(row(1),4,1);set_int(row(1),8,7);set_int(row(1),12,6);
    set_int(row(1),16,11);set_int(row(1),20,3);set_int(row(1),148,1);set_int(row(1),196,1);
    set_real(row(1),48,5.0);set_real(row(1),104,1.0);set_real(row(1),112,5.0);set_real(row(1),120,4.0);
    set_real(row(1),56,1.0);set_real(row(1),80,2.0);set_real(row(1),232,1.0);set_real(row(1),256,1.0);
    set_real(row(1),280,1.0);
    // Row2 electron child birth 2 at parent post (1,0,0).
    set_int(row(2),0,1);set_int(row(2),4,1);set_int(row(2),8,8);set_int(row(2),12,7);
    set_int(row(2),16,11);set_int(row(2),20,1);set_int(row(2),148,1);set_int(row(2),196,2);
    set_real(row(2),48,2.0);set_real(row(2),104,0.5);set_real(row(2),112,2.0);set_real(row(2),120,1.5);
    set_real(row(2),56,1.0);set_real(row(2),80,1.5);
    set_real(row(2),232,0.0);set_real(row(2),240,1.0);set_real(row(2),248,0.0);
    set_real(row(2),256,0.0);set_real(row(2),264,1.0);set_real(row(2),272,0.0);
    set_real(row(2),280,0.5);
    // Row3 photon child birth 1 at parent post.
    set_int(row(3),0,1);set_int(row(3),4,1);set_int(row(3),8,9);set_int(row(3),12,7);
    set_int(row(3),16,22);set_int(row(3),20,1);set_int(row(3),148,1);set_int(row(3),196,2);
    set_real(row(3),48,1.0);set_real(row(3),104,0.0);set_real(row(3),112,1.0);set_real(row(3),120,1.0);
    set_real(row(3),56,1.0);set_real(row(3),80,2.0);
    set_real(row(3),232,0.0);set_real(row(3),240,0.0);set_real(row(3),248,1.0);
    set_real(row(3),256,0.0);set_real(row(3),264,0.0);set_real(row(3),272,1.0);
    const std::uint64_t next4[]={1,kElectronContinuationEnd,kElectronContinuationEnd,kElectronContinuationEnd};
    const std::uint64_t off4[]={0,2,2,2,2};
    const std::uint64_t kids[]={2,3};
    MaterialElectronStateSource src4{payload.data(),payload.size(),next4,off4,kids,2};
    MaterialElectronResponseView tab4{};tab4.section=2;tab4.density_g_cm3=1.0;
    tab4.state_sources=&src4;tab4.state_source_count=1;
    ElectronContinuationCursor in{};in.valid=true;in.source=0;in.row=0;
    in.fraction=0;in.energy_MeV=10.0;in.position={0,0,0};in.direction={1,0,0};
    in.section=2;in.density_g_cm3=1.0;in.pdg=11;
    float rho1[]={1.0F};std::uint8_t sec1[]={2};
    ElectronCtGeometry box{{1,1,1},{-5,-5,-5},{10,10,10},rho1,sec1};
    const auto adv = advance_electron_continuation(in,tab4,box);
    assert(adv.status==ElectronAdvanceStatus::step_complete);
    assert(adv.deposited_MeV==2.0 && adv.child_energy_MeV==3.0 && adv.next.energy_MeV==5.0);
    assert(adv.energy_residual_MeV==0);
    // Deposit: threshold 0.5 in [0,2), uniform point on clipped segment.
    { auto l=choose_electron_energy_lineage(in,adv,tab4,0.05,0.3);
      assert(l.status==ElectronLineageStatus::deposited && l.physical_selected_deposit_MeV==2.0);
      assert(std::abs(l.deposit_position[0]-0.3)<1e-12 && l.deposit_position[1]==0 && l.deposit_position[2]==0); }
    // Parent: threshold 5 in [2,7).
    { auto l=choose_electron_energy_lineage(in,adv,tab4,0.5,0.3);
      assert(l.status==ElectronLineageStatus::parent_continuation);
      assert(l.cursor.valid && l.cursor.energy_MeV==5.0 && l.cursor.row==1); }
    // Electron child: threshold 8 in [7,9).
    { auto l=choose_electron_energy_lineage(in,adv,tab4,0.8,0.3);
      assert(l.status==ElectronLineageStatus::electron_child && l.cursor.pdg==11);
      assert(l.cursor.energy_MeV==2.0 && l.child_index==0);
      for(unsigned k=0;k<3;++k)assert(std::abs(l.cursor.position[k]-adv.end[k])<1e-9); }
    // Photon: threshold 9.5 in [9,10) -> explicit pending, never deposit/escape.
    { auto l=choose_electron_energy_lineage(in,adv,tab4,0.95,0.3);
      assert(l.status==ElectronLineageStatus::photon_pending && l.cursor.pdg==22);
      assert(l.cursor.energy_MeV==1.0 && l.child_index==1); }
    // Invalid uniforms / closure / malformed children rejected.
    assert(choose_electron_energy_lineage(in,adv,tab4,qnan,0.3).status==ElectronLineageStatus::invalid);
    assert(choose_electron_energy_lineage(in,adv,tab4,1.0,0.3).status==ElectronLineageStatus::invalid);
    assert(choose_electron_energy_lineage(in,adv,tab4,0.5,qnan).status==ElectronLineageStatus::invalid);
    assert(choose_electron_energy_lineage(in,adv,tab4,0.5,1.0).status==ElectronLineageStatus::invalid);
    { auto bad=adv;bad.deposited_MeV+=1e-3;
      assert(choose_electron_energy_lineage(in,bad,tab4,0.5,0.3).status==ElectronLineageStatus::invalid); }
    { auto bad=adv;bad.child_energy_MeV=2.5;
      assert(choose_electron_energy_lineage(in,bad,tab4,0.8,0.3).status==ElectronLineageStatus::invalid); }
    { auto bad=adv;bad.children_end=5;
      assert(choose_electron_energy_lineage(in,bad,tab4,0.8,0.3).status==ElectronLineageStatus::invalid); }
    { ElectronContinuationCursor wrong=in;wrong.energy_MeV=9.5;
      assert(choose_electron_energy_lineage(wrong,adv,tab4,0.5,0.3).status==ElectronLineageStatus::invalid); }
    // Invalid parent is rejected even when the draw falls in the deposit band.
    { ElectronContinuationCursor wrong=in;wrong.energy_MeV=9.5;
      assert(choose_electron_energy_lineage(wrong,adv,tab4,0.05,0.3).status==ElectronLineageStatus::invalid); }
    { ElectronContinuationCursor wrong=in;wrong.row=999;
      assert(choose_electron_energy_lineage(wrong,adv,tab4,0.05,0.3).status==ElectronLineageStatus::invalid); }
    // Malformed empty CSR is rejected even for a deposit draw.
    { auto bad=adv;bad.children_begin=999;bad.children_end=999;bad.child_energy_MeV=0;
      bad.deposited_MeV=5.0; // forge closure 10=5+5+0 to isolate the CSR check
      assert(choose_electron_energy_lineage(in,bad,tab4,0.05,0.3).status==ElectronLineageStatus::invalid); }
    // Omitted children with forged matching energies rejected (range mismatch).
    { auto bad=adv;bad.children_begin=0;bad.children_end=0;bad.child_energy_MeV=0;
      bad.deposited_MeV=5.0; // 10=5+5+0 closes, but raw row owns 2 children
      assert(choose_electron_energy_lineage(in,bad,tab4,0.05,0.3).status==ElectronLineageStatus::invalid); }
    // Positive-energy next cursor required regardless of selected branch.
    { auto bad=adv;bad.next.valid=false;
      assert(choose_electron_energy_lineage(in,bad,tab4,0.05,0.3).status==ElectronLineageStatus::invalid); }
    // Deterministic stratified uniforms on 10=2+5+2+1: 20/50/20/10, W unchanged.
    { unsigned n_dep=0,n_par=0,n_e=0,n_ph=0;
      for(unsigned i=0;i<100;++i) {
          double packet_W=1.0; // dose packet weight lives with caller, never scaled
          const double u=(double(i)+0.5)/100.0;
          auto l=choose_electron_energy_lineage(in,adv,tab4,u,0.3);
          assert(packet_W==1.0);
          if(l.status==ElectronLineageStatus::deposited) {
              ++n_dep;
              assert(l.physical_selected_deposit_MeV==2.0); // placement only, not packet score
          }
          else if(l.status==ElectronLineageStatus::parent_continuation)++n_par;
          else if(l.status==ElectronLineageStatus::electron_child)++n_e;
          else if(l.status==ElectronLineageStatus::photon_pending)++n_ph;
          else assert(false);
      }
      assert(n_dep==20 && n_par==50 && n_e==20 && n_ph==10); }

    // ---- 3. boundary / escape / exhausted keep distinct statuses ----
    std::array<unsigned char,kRec> clip = blank_row();
    set_int(clip.data(),8,1);set_int(clip.data(),16,11);set_int(clip.data(),20,1);
    set_real(clip.data(),48,10.0);
    set_real(clip.data(),56,0.5);set_real(clip.data(),64,0.5);set_real(clip.data(),72,0.5);
    set_real(clip.data(),80,2.5);set_real(clip.data(),88,0.5);set_real(clip.data(),96,0.5);
    set_real(clip.data(),104,4.0);set_real(clip.data(),112,10.0);set_real(clip.data(),120,6.0);
    set_real(clip.data(),232,1.0);set_real(clip.data(),256,1.0);set_real(clip.data(),280,2.0);
    const std::uint64_t cn[]={kElectronContinuationEnd}, coff[]={0,0};
    MaterialElectronStateSource cs{clip.data(),clip.size(),cn,coff,nullptr,0};
    MaterialElectronResponseView ct{};ct.section=2;ct.density_g_cm3=1.0;
    ct.state_sources=&cs;ct.state_source_count=1;
    ElectronContinuationCursor cc{};cc.valid=true;cc.row=0;cc.energy_MeV=10.0;
    cc.position={0.5,0.5,0.5};cc.direction={1,0,0};cc.section=2;cc.density_g_cm3=1.0;
    float rho3[]={1.0F,1.0F,0.99F};std::uint8_t sec3[]={2,2,2};
    ElectronCtGeometry grid{{3,1,1},{0,0,0},{1,1,1},rho3,sec3};
    const auto cut = advance_electron_continuation(cc,ct,grid);
    assert(cut.status==ElectronAdvanceStatus::material_boundary);
    assert(cut.deposited_MeV==3.0 && cut.next.energy_MeV==7.0);
    { auto l=choose_electron_energy_lineage(cc,cut,ct,0.1,0.5); // threshold 1 < 3
      assert(l.status==ElectronLineageStatus::deposited && l.physical_selected_deposit_MeV==3.0);
      assert(std::abs(l.deposit_position[0]-1.25)<1e-12); // exact clipped midpoint
      assert(l.deposit_position[1]==0.5 && l.deposit_position[2]==0.5); }
    { auto l=choose_electron_energy_lineage(cc,cut,ct,0.5,0.5); // threshold 5 >= 3
      assert(l.status==ElectronLineageStatus::material_boundary && l.cursor.energy_MeV==7.0); }
    float rhoe[]={1.0F,1.0F};std::uint8_t sece[]={2,2};
    ElectronCtGeometry egrid{{2,1,1},{0,0,0},{1,1,1},rhoe,sece};
    const auto esc = advance_electron_continuation(cc,ct,egrid);
    assert(esc.status==ElectronAdvanceStatus::escaped);
    { auto l=choose_electron_energy_lineage(cc,esc,ct,0.9,0.0);
      assert(l.status==ElectronLineageStatus::escaped && l.cursor.energy_MeV==7.0); }
    // Zero KE at a face must be stopped, never escape/boundary (production guard kept).
    { auto bad=esc;bad.next.energy_MeV=0;
      assert(choose_electron_energy_lineage(cc,bad,ct,0.05,0.0).status==ElectronLineageStatus::invalid); }
    // Finite-source exhaustion stays distinct.
    std::array<unsigned char,kRec> fin = blank_row();
    set_int(fin.data(),8,1);set_int(fin.data(),16,11);set_int(fin.data(),20,1);
    set_real(fin.data(),48,10.0);
    set_real(fin.data(),104,4.0);set_real(fin.data(),112,10.0);set_real(fin.data(),120,6.0);
    set_real(fin.data(),232,1.0);set_real(fin.data(),256,1.0);set_real(fin.data(),280,1.0);
    set_real(fin.data(),56,0.0);set_real(fin.data(),80,1.0);
    MaterialElectronStateSource fs{fin.data(),fin.size(),cn,coff,nullptr,0};
    MaterialElectronResponseView ft{};ft.section=2;ft.density_g_cm3=1.0;
    ft.state_sources=&fs;ft.state_source_count=1;
    ElectronContinuationCursor fc{};fc.valid=true;fc.row=0;fc.energy_MeV=10.0;
    fc.position={0,0,0};fc.direction={1,0,0};fc.section=2;fc.density_g_cm3=1.0;
    const auto exh = advance_electron_continuation(fc,ft,box);
    assert(exh.status==ElectronAdvanceStatus::source_exhausted);
    { auto l=choose_electron_energy_lineage(fc,exh,ft,0.5,0.0);
      assert(l.status==ElectronLineageStatus::source_exhausted && l.cursor.energy_MeV==6.0); }
    // Childless exhausted row: out-of-bounds empty window rejected even on deposit draw.
    { auto bad=exh;bad.children_begin=7;bad.children_end=7;
      assert(choose_electron_energy_lineage(fc,bad,ft,0.05,0.0).status==ElectronLineageStatus::invalid); }
    // Clipped transaction must expose no children, even on a deposit draw.
    { auto bad=cut;bad.children_begin=0;bad.children_end=0;bad.child_energy_MeV=0;
      // cut already has empty window; forge a nonempty claim with matching closure is impossible
      // for clipped (child sum must be 0), so check out-of-bounds empty instead:
      bad.children_begin=4;bad.children_end=4;
      assert(choose_electron_energy_lineage(cc,bad,ct,0.05,0.5).status==ElectronLineageStatus::invalid); }

    // Photon flight: no loss before interaction, deposit only at endpoint,
    // no interpolation of pre/post photon momenta before the scattering event.
    set_int(clip.data(),16,22);
    set_real(clip.data(),256,0);set_real(clip.data(),264,1);
    auto pc=cc;pc.pdg=22;
    const auto photon_cut=advance_electron_continuation(pc,ct,grid);
    assert(photon_cut.status==ElectronAdvanceStatus::material_boundary);
    assert(photon_cut.deposited_MeV==0 && photon_cut.child_energy_MeV==0);
    assert(photon_cut.next.energy_MeV==10 && photon_cut.next.direction==pc.direction);
    const auto photon_escape=advance_electron_continuation(pc,ct,egrid);
    assert(photon_escape.status==ElectronAdvanceStatus::escaped && photon_escape.next.energy_MeV==10);
    const auto photon_whole=advance_electron_continuation(pc,ct,box);
    assert(photon_whole.status==ElectronAdvanceStatus::source_exhausted);
    assert(photon_whole.deposited_MeV==4 && photon_whole.next.energy_MeV==6);
    const auto endpoint=choose_electron_energy_lineage(pc,photon_whole,ct,0.1,0.25);
    assert(endpoint.status==ElectronLineageStatus::deposited && endpoint.deposit_position==photon_whole.end);
    auto uniform=[](){return 0.5;};
    ElectronEnergyPacket pp;pp.status=ElectronPacketStatus::active;pp.cursor=pc;pp.weight_MeV=7.25;
    auto remaining=transport_electron_packet_step(pp,&ct,1,box,uniform);
    assert(remaining.status==ElectronPacketStatus::active && remaining.cursor.energy_MeV==6);
    remaining=transport_electron_packet_step(remaining,&ct,1,box,uniform);
    assert(remaining.status==ElectronPacketStatus::coverage_missing && remaining.weight_MeV==7.25);
    assert(remaining.gap==ElectronPacketGap::photon_continuation && remaining.cursor.energy_MeV==6);
    const auto escaped_packet=transport_electron_packet_step(pp,&ct,1,egrid,uniform);
    assert(escaped_packet.status==ElectronPacketStatus::escaped && escaped_packet.weight_MeV==7.25);

    // A photon may create an electron; its exact child is then transported,
    // not relabelled/dumped. Restore source mutations after this local check.
    set_int(row(0),16,22);set_int(row(1),16,22);
    auto photon_parent=in;photon_parent.pdg=22;
    const auto phadv=advance_electron_continuation(photon_parent,tab4,box);
    const auto phchild=choose_electron_energy_lineage(photon_parent,phadv,tab4,0.8,0.2);
    assert(phchild.status==ElectronLineageStatus::electron_child && phchild.cursor.pdg==11);
    assert(advance_electron_continuation(phchild.cursor,tab4,box).status!=ElectronAdvanceStatus::invalid);
    set_int(row(0),16,11);set_int(row(1),16,11);

    // Finite-source electron KE=6 restarts in the SAME material at an exact
    // energy crossing of an independent 8->0 step, without modifying W/KE.
    set_real(fin.data(),48,8);set_real(fin.data(),112,8);
    set_real(fin.data(),104,8);set_real(fin.data(),120,0);
    const auto crossing=ElectronEnergyCrossings::build(fs.continuation(),4096*1024);
    fs.crossings={crossing.edges.data(),crossing.offsets.data(),crossing.rows.data(),
                  crossing.edges.size()-1,crossing.rows.size()};
    ElectronEnergyPacket ep;ep.status=ElectronPacketStatus::active;ep.cursor=exh.next;ep.weight_MeV=7.25;
    auto restarted=transport_electron_packet_step(ep,&ft,1,box,uniform);
    assert(restarted.material_table_index==0);
    // Cached material lookup must preserve sampling, state and RNG consumption.
    for(unsigned seed=1;seed<=32;++seed) {
        std::uint64_t ra=seed,rb=seed;
        auto ua=[&](){ra=ra*6364136223846793005ULL+1442695040888963407ULL;return double(ra>>11)*0x1.0p-53;};
        auto ub=[&](){rb=rb*6364136223846793005ULL+1442695040888963407ULL;return double(rb>>11)*0x1.0p-53;};
        auto a=ep,b=ep;b.material_table_index=0;
        for(unsigned step=0;step<8 && a.status==ElectronPacketStatus::active;++step) {
            a.material_table_index=std::numeric_limits<std::size_t>::max();
            a=transport_electron_packet_step(a,&ft,1,box,ua);
            b=transport_electron_packet_step(b,&ft,1,box,ub);
            assert(ra==rb && a.status==b.status && a.gap==b.gap);
            assert(a.weight_MeV==b.weight_MeV && a.deposit_position==b.deposit_position);
            assert(a.cursor.position==b.cursor.position && a.cursor.direction==b.cursor.direction);
            assert(a.cursor.source==b.cursor.source && a.cursor.row==b.cursor.row);
            assert(a.cursor.energy_MeV==b.cursor.energy_MeV && a.cursor.fraction==b.cursor.fraction);
            assert(a.advances==b.advances && a.source_restarts==b.source_restarts);
        }
    }
    auto wrong_index=ep;wrong_index.material_table_index=1;
    assert(transport_electron_packet_step(wrong_index,&ft,1,box,uniform).status==ElectronPacketStatus::invalid);
    assert(restarted.status==ElectronPacketStatus::active && restarted.source_restarts==1);
    assert(restarted.cursor.energy_MeV==6 && restarted.weight_MeV==7.25);
    assert(restarted.cursor.position==ep.cursor.position && restarted.cursor.direction==ep.cursor.direction);
    auto deposited=transport_electron_packet_step(restarted,&ft,1,box,uniform);
    assert(deposited.status==ElectronPacketStatus::deposited && deposited.weight_MeV==7.25);
    assert(transport_electron_packet_step(deposited,&ft,1,box,uniform).advances==deposited.advances);
    auto missing=ep;missing.cursor.energy_MeV=40;
    missing=transport_electron_packet_step(missing,&ft,1,box,uniform);
    assert(missing.status==ElectronPacketStatus::coverage_missing && missing.weight_MeV==7.25 && missing.cursor.energy_MeV==40);
    auto bad_uniform=[](){return std::numeric_limits<double>::quiet_NaN();};
    const auto invalid_restart=transport_electron_packet_step(ep,&ft,1,box,bad_uniform);
    assert(invalid_restart.status==ElectronPacketStatus::invalid && invalid_restart.weight_MeV==7.25);
    std::cout << "electron_energy_lineage_smoke: birth_cone=1 photon_endpoint=1 photon_child=1 source_restart=1 retained_gap=1 failed=0\n";
    return 0;
}

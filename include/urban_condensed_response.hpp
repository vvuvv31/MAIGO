#pragma once
// Joint angle/displacement/path response for C12 Urban transport in Water_75eV.
// Binary identity/grids/shapes are checked; no checksum or sidecar is used.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
namespace carbon {
struct CondensedHeader {
 char magic[8];
 std::uint32_t z,a,ne,nh,nt,na,nb,ntq,naq,nbq,nsq,branches,floats,probability_energies;
 float density,excitation,zeff,radlen,mass,coeff1,coeff2,reserved;
};
struct CondensedNode {
 std::uint32_t first_branch,branch_count;
 float p0,max_theta,max_a,max_b,max_u;
 std::uint32_t reserved;
};
struct CondensedBranch {
 std::uint32_t theta,a,b,slack,tail,tail_count;
 std::int32_t sign;
 float weight,tail_weight;
 std::uint32_t reserved;
};
static_assert(sizeof(CondensedHeader)==96&&sizeof(CondensedNode)==32&&sizeof(CondensedBranch)==40);
struct CondensedView {
 CondensedHeader header{};
 const float *energies{},*lengths{},*data{},*probability_energies{},*no_subdivision{};
 const CondensedNode* nodes{};
 const CondensedBranch* branches{};
};
struct UrbanCondensedResponsePackage {
 CondensedHeader header{};
 std::vector<float> energies,lengths,data,probability_energies,no_subdivision;
 std::vector<CondensedNode> nodes;
 std::vector<CondensedBranch> branches;
 static UrbanCondensedResponsePackage load(const std::filesystem::path& path) {
  UrbanCondensedResponsePackage out;std::ifstream in(path,std::ios::binary);
  in.read(reinterpret_cast<char*>(&out.header),sizeof(CondensedHeader));auto& h=out.header;
  if(!in||std::memcmp(h.magic,"URBCND02",8)||h.z!=6||h.a!=12||h.ne<2||h.ne>2048||h.nh<2||h.nh>2048||
     !h.nt||h.nt>1024||!h.na||h.na>1024||!h.nb||h.nb>1024||!h.ntq||h.ntq>65536||
     !h.naq||h.naq>1024||!h.nbq||h.nbq>1024||!h.nsq||h.nsq>1024||
     h.branches<h.ne*h.nh||h.branches>2*h.ne*h.nh||h.floats>500000000||h.probability_energies<2||h.probability_energies>4096||h.reserved!=0||
     !std::isfinite(h.density+h.excitation+h.zeff+h.radlen+h.mass+h.coeff1+h.coeff2)||h.density!=1.f||h.excitation!=75e-6f||!(h.zeff>0&&h.radlen>0&&h.mass>0&&h.coeff1>0&&h.coeff2>0))
    throw std::runtime_error("Invalid condensed water/C12 package header");
  const std::uint64_t cells=std::uint64_t(h.ne)*h.nh;
  const std::uint64_t bytes=sizeof(h)+4ULL*(h.ne+h.nh)+4ULL*h.probability_energies*(1ULL+h.nh)+sizeof(CondensedNode)*cells+
      sizeof(CondensedBranch)*h.branches+4ULL*h.floats;
  if(std::filesystem::file_size(path)!=bytes)throw std::runtime_error("Condensed package size mismatch");
  auto read=[&]<class T>(std::vector<T>& v,std::size_t n){v.resize(n);in.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));if(!in)throw std::runtime_error("Truncated condensed package");};
  read(out.energies,h.ne);read(out.lengths,h.nh);read(out.probability_energies,h.probability_energies);read(out.no_subdivision,std::size_t(h.probability_energies)*h.nh);read(out.nodes,cells);read(out.branches,h.branches);read(out.data,h.floats);
  auto grid=[](const std::vector<float>& g){for(std::size_t i=0;i<g.size();++i)if(!(g[i]>0&&std::isfinite(g[i]))||(i&&!(g[i]>g[i-1])))throw std::runtime_error("Invalid condensed response grid");};
  grid(out.energies);grid(out.lengths);grid(out.probability_energies);
  if(out.probability_energies.front()!=out.energies.front()||out.probability_energies.back()!=out.energies.back())throw std::runtime_error("Condensed probability/shape coverage mismatch");
  for(float p:out.no_subdivision)if(!(p>0&&p<=1&&std::isfinite(p)))throw std::runtime_error("Invalid condensed no-subdivision probability");
  for(float x:out.data)if(!std::isfinite(x))throw std::runtime_error("Nonfinite condensed response");
  const std::uint64_t knots=std::uint64_t(h.naq)+h.nbq+h.nsq+3;
  if(knots>h.floats)throw std::runtime_error("Missing condensed probability grids");
  auto probability_grid=[&](std::size_t off,unsigned n){if(out.data[off]!=0||out.data[off+n]!=1)throw std::runtime_error("Bad condensed probability endpoints");for(unsigned j=0;j<n;++j)if(!(out.data[off+j+1]>out.data[off+j]))throw std::runtime_error("Nonmonotone condensed probability knots");};
  probability_grid(0,h.naq);probability_grid(h.naq+1,h.nbq);probability_grid(h.naq+h.nbq+2,h.nsq);
  auto check_array=[&](unsigned off,std::uint64_t rows,unsigned columns,float minimum,float maximum){
   if(std::uint64_t(off)+rows*columns>h.floats)throw std::runtime_error("Condensed response offset out of bounds");
   for(std::uint64_t i=0;i<rows;++i)for(unsigned j=0;j<columns;++j){float x=out.data[off+i*columns+j];if(x<minimum||x>maximum||(j&&x<out.data[off+i*columns+j-1]))throw std::runtime_error("Invalid condensed inverse response");}
  };
  unsigned next_branch=0;
  for(const auto& n:out.nodes) {
   if(!std::isfinite(n.p0+n.max_theta+n.max_a+n.max_b+n.max_u)||n.first_branch!=next_branch||n.branch_count<1||n.branch_count>2||n.first_branch+n.branch_count>h.branches||n.reserved||
      !(n.p0>0&&n.p0<=1&&n.max_theta>=0&&n.max_theta<=1&&n.max_a>=0&&n.max_b>=0&&n.max_u>=0&&n.max_u<1))
     throw std::runtime_error("Invalid condensed node identity or bounds");
   float sum=0;int signs=0;
   for(unsigned k=0;k<n.branch_count;++k) {
    const auto& b=out.branches[n.first_branch+k];
    if((b.sign!=-1&&b.sign!=1)||!b.tail_count||b.tail_count>10000000||b.reserved||!(b.weight>0&&b.weight<=1&&b.tail_weight>0&&b.tail_weight<1))throw std::runtime_error("Invalid condensed signed branch");
    int bit=b.sign>0?1:2;if(signs&bit)throw std::runtime_error("Repeated condensed direction sign");signs|=bit;sum+=b.weight;
    check_array(b.theta,1,h.ntq+1,0,n.max_theta);check_array(b.a,h.nt,h.naq+1,-n.max_a,n.max_a);
    check_array(b.b,std::uint64_t(h.nt)*h.na,h.nbq+1,-n.max_b,n.max_b);
    check_array(b.slack,std::uint64_t(h.nt)*h.na*h.nb,h.nsq+1,0,n.max_u);
    if(std::uint64_t(b.tail)+4ULL*b.tail_count>h.floats)throw std::runtime_error("Condensed tail out of bounds");
    for(unsigned j=0;j<b.tail_count;++j){auto* p=out.data.data()+b.tail+4*j;if(!(p[0]>=0&&p[0]<=n.max_theta&&std::abs(p[1])<=n.max_a&&std::abs(p[2])<=n.max_b&&p[3]>=0&&p[3]<=n.max_u))throw std::runtime_error("Invalid condensed direct tail");}
   }
   if(std::abs(sum-1)>1e-6f)throw std::runtime_error("Unnormalized condensed branches");next_branch+=n.branch_count;
  }
  if(next_branch!=h.branches)throw std::runtime_error("Unreferenced condensed response branches");
  return out;
 }
};
}

// Included in namespace carbon after the response view, SYCL and rng.hpp.
// Probabilities describe a condensed response; no event clock is consumed.
struct CondensedQuery {
 unsigned cell[4]{};float weight[4]{},angle_scale[4]{},position_scale[4]{};
 float energy{},length{},p0{},max_theta{},max_radius{},max_u{};
 bool valid{},has_backward{};
};
struct CondensedCanonical {float theta{},a{},b{},u{},path_excess{},dz{};};
inline unsigned condensed_lower(const float* p,unsigned n,float value) {
 unsigned lo=0,hi=n-1;
 while(hi-lo>1){unsigned m=(lo+hi)/2;if(p[m]<=value)lo=m;else hi=m;}
 return lo;
}
inline float condensed_core_scale(CondensedHeader h,float energy_mevu,float length) {
 float e=h.a*energy_mevu,inv=(e+h.mass)/(e*(e+2*h.mass)),y=length/h.radlen;
 return 13.6f*h.z*sycl::sqrt(y)*inv*(h.coeff1+h.coeff2*sycl::log(y));
}
inline CondensedQuery condensed_query(CondensedView v,float energy_mevu,float length) {
 CondensedQuery q{};auto h=v.header;
 if(!v.nodes||!(energy_mevu>=v.energies[0]&&energy_mevu<=v.energies[h.ne-1]&&length>=v.lengths[0]&&length<=v.lengths[h.nh-1]))return q;
 unsigned ie=condensed_lower(v.energies,h.ne,energy_mevu),ih=condensed_lower(v.lengths,h.nh,length);
 float e0=v.energies[ie],e1=v.energies[ie+1],h0=v.lengths[ih],h1=v.lengths[ih+1];
 float we=sycl::log(energy_mevu/e0)/sycl::log(e1/e0),wh=sycl::log(length/h0)/sycl::log(h1/h0);
 // log(p0) is affine in path length at each fixed energy. Its interpolation
 // retains that probability identity without introducing a physical clock.
 unsigned ip=condensed_lower(v.probability_energies,h.probability_energies,energy_mevu);
 float pe=(energy_mevu-v.probability_energies[ip])/(v.probability_energies[ip+1]-v.probability_energies[ip]);
 float ph=(length-h0)/(h1-h0),logp0=0;
 for(unsigned i=0;i<4;++i) {
  float wp=(i/2?pe:1-pe)*(i%2?ph:1-ph);
  logp0+=wp*sycl::log(v.no_subdivision[(ip+i/2)*h.nh+ih+i%2]);
 }
 float scale=condensed_core_scale(h,energy_mevu,length);
 for(unsigned i=0;i<4;++i) {
  unsigned ei=ie+i/2,hi=ih+i%2;q.cell[i]=ei*h.nh+hi;
  q.weight[i]=(i/2?we:1-we)*(i%2?wh:1-wh);
  const auto n=v.nodes[q.cell[i]];
  q.angle_scale[i]=scale/condensed_core_scale(h,v.energies[ei],v.lengths[hi]);
  q.position_scale[i]=q.angle_scale[i]*length/v.lengths[hi];
  if(q.weight[i]>0) {
   q.max_theta=sycl::fmax(q.max_theta,n.max_theta*q.angle_scale[i]);
   q.max_radius=sycl::fmax(q.max_radius,sycl::sqrt(n.max_a*n.max_a+n.max_b*n.max_b)*q.position_scale[i]);
   q.max_u=sycl::fmax(q.max_u,n.max_u);
   for(unsigned b=0;b<n.branch_count;++b)q.has_backward|=v.branches[n.first_branch+b].sign<0;
  }
 }
 q.energy=energy_mevu;q.length=length;q.p0=sycl::exp(logp0);
 q.valid=sycl::isfinite(q.p0+q.max_theta+q.max_radius+q.max_u)&&q.p0>=0&&q.p0<=1&&q.max_theta<1&&q.max_u<1;
 return q;
}
inline float condensed_inverse(float u,const float* probabilities,unsigned n,const float* row) {
 unsigned lo=0,hi=n;
 while(hi-lo>1){unsigned m=(lo+hi)/2;if(probabilities[m]<=u)lo=m;else hi=m;}
 float f=(u-probabilities[lo])/(probabilities[lo+1]-probabilities[lo]);
 return row[lo]+f*(row[lo+1]-row[lo]);
}
inline CondensedCanonical condensed_draw(CondensedView v,CondensedQuery q,
 std::uint64_t seed,std::uint64_t history,std::uint64_t step) {
 auto uniform=[&](unsigned d){return rng::msc_unit_strict(seed,history,step,d);};
 unsigned corner=3;float u=uniform(199),edge=0;
 for(unsigned i=0;i<3;++i){edge+=q.weight[i];if(u<edge){corner=i;break;}}
 const auto n=v.nodes[q.cell[corner]];unsigned k=0;
 if(n.branch_count==2&&uniform(192)>=v.branches[n.first_branch].weight)k=1;
 auto b=v.branches[n.first_branch+k];auto h=v.header;CondensedCanonical r{};
 if(uniform(193)<b.tail_weight) {
  unsigned i=sycl::min(unsigned(uniform(194)*b.tail_count),b.tail_count-1);auto p=v.data+b.tail+4*i;
  r.theta=p[0];r.a=p[1];r.b=p[2];r.u=p[3];
 } else {
  // Couple the core radius quantile to the original Urban inverse draw.
  // Complementing the 24 random bits is a permutation, so this retains the
  // same strict-unit marginal; it improves matched-history precision only.
  const unsigned bits=(rng::msc_random_u32(seed,history,step,162)>>8U)^0xffffffU;
  const float reverse=(static_cast<float>(bits)+.5f)*0x1p-24f;
  float ut=sycl::fmin(reverse,0x1.fffffep-1f),ua=uniform(195),ub=uniform(196),us=uniform(197);
  unsigned it=sycl::min(unsigned(ut*h.nt),h.nt-1),ia=sycl::min(unsigned(ua*h.na),h.na-1),ib=sycl::min(unsigned(ub*h.nb),h.nb-1);
  float x=ut*h.ntq;unsigned j=sycl::min(unsigned(x),h.ntq-1);float f=x-j;
  r.theta=v.data[b.theta+j]+f*(v.data[b.theta+j+1]-v.data[b.theta+j]);
  const float* pa=v.data;const float* pb=pa+h.naq+1;const float* ps=pb+h.nbq+1;
  r.a=condensed_inverse(ua,pa,h.naq,v.data+b.a+it*(h.naq+1));
  r.b=condensed_inverse(ub,pb,h.nbq,v.data+b.b+(it*h.na+ia)*(h.nbq+1));
  r.u=condensed_inverse(us,ps,h.nsq,v.data+b.slack+((it*h.na+ia)*h.nb+ib)*(h.nsq+1));
 }
 r.theta*=q.angle_scale[corner];r.a*=q.position_scale[corner];r.b*=q.position_scale[corner];
 float r2=r.a*r.a+r.b*r.b,chord=sycl::sqrt(q.length*q.length+r2);
 float minimum=r2/(q.length+chord),scale=minimum+q.length*r.theta*r.theta;
 r.path_excess=minimum+scale*r.u/(1-r.u);
 r.dz=b.sign*sycl::sqrt(1-r.theta*r.theta);return r;
}

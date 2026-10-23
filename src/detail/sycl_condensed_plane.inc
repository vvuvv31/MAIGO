// Condensed Urban plane response, prepared before loss/nuclear limits.
// Unsupported initial states keep the original Urban update; sampled tails
// are never discarded or resampled to fit a voxel.
struct UrbanPlaneProposal {
 Direction3F endpoint{},direction{};
 float true_path{},stable_delta{};
 bool active{},used{},failed{};
};
inline UrbanPlaneProposal condensed_plane_prepare(CondensedView view,UrbanGlobalProposal& p,
 std::uint64_t seed,std::uint64_t history,std::uint64_t step,
 float nuclear_macro=0.f,float nuclear_tau=0.f) {
 UrbanPlaneProposal out{};if(!view.nodes||!p.valid)return out;
 const auto& cell=p.cell;auto d=p.direction;
 float dn=sycl::sqrt(d.x*d.x+d.y*d.y+d.z*d.z);d={d.x/dn,d.y/dn,d.z/dn};
 // Eligibility depends only on the initial state, before any random draw.
 const unsigned mask=(cell.safety.kind!=1&&cell.safety.kind!=2?1U:0U)|(cell.face_mask!=4U?2U:0U)|
     (d.z<.99875026f?4U:0U)|(p.limit.no_displacement?8U:0U)|
     (!urban_reaches_voxel_face(p.geom_path,cell.face_distance)?16U:0U);
 if(mask) {return out;}
 float h=(cell.face_position.z-p.position.z)/d.z;
 auto q=condensed_query(view,p.energy/12.f,h);
 if(!q.valid||q.has_backward){return out;}
 // All bounds below depend on the initial state and the complete neighboring
 // distributions, rather than on the sampled event or tail.
 float sine=sycl::sqrt(d.x*d.x+d.y*d.y);
 float minz=d.z*sycl::sqrt(1-q.max_theta*q.max_theta)-sine*q.max_theta;
 if(!(minz>0)){return out;}
 float radius=q.max_radius*(1+8*1.1920928955078125e-7f),r2=radius*radius;
 float minimum=r2/(h+sycl::sqrt(h*h+r2)),scale=minimum+h*q.max_theta*q.max_theta;
 float excess=minimum+scale*q.max_u/(1-q.max_u);
 float eta=sine*radius/(h*d.z);
 if(!(eta<1)){return out;}
 float max_factor=1/(1-eta);
 float max_length=max_factor*(h+excess)*(1+16*1.1920928955078125e-7f);
 // Use the existing nuclear clock before drawing any response. If any
 // supported true path could reach that event, keep the original Urban step.
 // This avoids conditioning the angle draw on an observed shortening, and
 // requires no approximate endpoint at an interior nuclear collision.
 if(nuclear_macro>0.f && nuclear_macro*sycl::fmax(max_length,p.true_path)>=nuclear_tau) {
  return out;
 }
 // Bound the complete scaled support before sampling, including world-coordinate
 // rounding. No sampled endpoint is clipped, rejected or resampled to fit.
 float world_scale=sycl::fmax(1.f,sycl::fmax(h,sycl::fmax(sycl::fabs(p.position.x),sycl::fabs(p.position.y))));
 float endpoint_x=p.position.x+h*d.x,endpoint_y=p.position.y+h*d.y;
 float margin=max_factor*(radius+eta*h*sine)+16*1.1920928955078125e-7f*world_scale;
 if(max_length>p.limit.t_msc_mm||endpoint_x-margin<=cell.safety.box_x0||
    endpoint_x+margin>=cell.safety.box_x1||endpoint_y-margin<=cell.safety.box_y0||
    endpoint_y+margin>=cell.safety.box_y1){return out;}
 if(cell.safety.kind==2) {
  // The surrounding water shares the same material law. Bound the entire
  // normal response path, not only its exit point, before using it outside
  // the score ROI. A path of length L joining (0,0) to (r,h) lies inside
  // its focal ellipsoid: its transverse radius is <=(r+sqrt(L^2-h^2))/2.
  // The pre-draw union bound retains all neighboring response tails.
  const float path_radius=.5f*(radius+sycl::sqrt(excess*(2*h+excess)));
  const float path_margin=max_factor*(path_radius+eta*h*sine)+
      16*1.1920928955078125e-7f*world_scale;
  const float x0=sycl::fmin(p.position.x,endpoint_x)-path_margin;
  const float x1=sycl::fmax(p.position.x,endpoint_x)+path_margin;
  const float y0=sycl::fmin(p.position.y,endpoint_y)-path_margin;
  const float y1=sycl::fmax(p.position.y,endpoint_y)+path_margin;
  const bool separated=x1<cell.safety.inner_x0||x0>cell.safety.inner_x1||
                       y1<cell.safety.inner_y0||y0>cell.safety.inner_y1;
  if(!separated||x0<=cell.safety.box_x0||x1>=cell.safety.box_x1||
     y0<=cell.safety.box_y0||y1>=cell.safety.box_y1) {
   return out;
  }
 }

 if(rng::msc_unit_strict(seed,history,step*1024U,200)<q.p0){return out;}
 auto c=condensed_draw(view,q,seed,history,step*1024U);
 // Alternative MSC branches share the azimuth draw; all response quantiles
 // remain independent of it. This is common-random-number coupling, not a
 // change in the scattering law or an alias with another physical process.
 float phi=6.2831853071795864769f*rng::msc_unit_strict(seed,history,step*1024U,163);
 float cp=sycl::cos(phi),sp=sycl::sin(phi),x=c.a*cp-c.b*sp,y=c.a*sp+c.b*cp;
 auto u=rotate_local_direction(1,0,0,d),v=rotate_local_direction(0,1,0,d);
 auto next=rotate_local_direction(c.theta*cp,c.theta*sp,c.dz,d);
 // Map the entire rotated path by one positive similarity factor. Extending
 // backwards along the final post-boundary direction is not valid for a late
 // large-angle kick. Whole-path scaling preserves the path/chord inequality.
 float projection_z=h*d.z+u.z*x+v.z*y;
 float factor=(h*d.z)/projection_z;
 float transverse2=x*x+y*y;
 float sampled_minimum=transverse2/(h+sycl::sqrt(h*h+transverse2));
 float slack=(sampled_minimum+h*c.theta*c.theta)*c.u/(1-c.u);
 float path=factor*(h+sampled_minimum+slack);
 // Reconstruct the nonnegative gap directly; subtracting two nearly equal
 // path lengths would destroy the sub-ULP longitudinal information.
 float stable=factor*slack;
 Direction3F endpoint{p.position.x+factor*(h*d.x+u.x*x+v.x*y),
                     p.position.y+factor*(h*d.y+u.y*x+v.y*y),cell.face_position.z};
 if(!(path>0&&stable>=0&&stable<=path&&path<=p.limit.t_msc_mm&&
      endpoint.x>=cell.safety.box_x0&&endpoint.x<=cell.safety.box_x1&&
      endpoint.y>=cell.safety.box_y0&&endpoint.y<=cell.safety.box_y1&&
      sycl::isfinite(path+stable+endpoint.x+endpoint.y+endpoint.z))) {
  out.failed=true;p.valid=false;return out;
 }
 out.endpoint=endpoint;out.direction=next;out.true_path=path;
 out.stable_delta=stable;out.active=true;p.true_path=path;
 return out;
}
inline CorrelatedScatteringStep condensed_plane_finalize(const UrbanGlobalProposal& p,
 UrbanPlaneProposal& c,float accepted,std::uint64_t seed,std::uint64_t history,
 std::uint64_t step) {
 if(!c.active)return urban_global_finalize(p,accepted,seed,history,step);
 // The pre-draw full-support clock guard guarantees no nuclear truncation
 // of an active response. A violation rejects the batch rather than sampling
 // or shortening a conditional plane response after seeing its outcome.
 if(accepted!=c.true_path) {
  c.failed=true;
  return CorrelatedScatteringStep{};
 }
 const float delta=c.stable_delta,geom=accepted-delta;
 CorrelatedScatteringStep r{c.direction,{c.endpoint.x-p.position.x-geom*p.direction.x,
   c.endpoint.y-p.position.y-geom*p.direction.y,c.endpoint.z-p.position.z-geom*p.direction.z}};
 r.proposed_true_path_mm=p.limit.t_msc_mm;r.final_true_path_mm=accepted;r.final_geom_path_mm=geom;
 r.stable_delta_mm=delta;r.current_range_mm=p.limit.range_mm;r.pre_step_energy_mev=p.energy;
 r.predicted_internal_energy_mev=p.energy;r.proposal_valid=true;r.boundary_crossed=true;
 r.range_terminal=accepted>=p.limit.range_mm;c.used=true;
 return r;
}

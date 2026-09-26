inline bool urban_global_outside_water(Direction3F p,Direction3F d,float h,float length) {
    return p.z<0 || p.z>=length || (p.z==0 && d.z<0) ||
        p.x< -h || p.x>h || p.y< -h || p.y>h ||
        (p.x== -h && d.x<0) || (p.x==h && d.x>=0) ||
        (p.y== -h && d.y<0) || (p.y==h && d.y>=0);
}
// TOPAS water includes a physical replica ROI inside a larger water box.
// Exterior tracks remain in water and may re-enter the ROI; no artificial
// replica faces are introduced outside the scored daughter volume.
inline UrbanVoxelCell urban_global_water_cell(Direction3F p,Direction3F d,
    Direction3F lo,Direction3F hi,Direction3F pitch,int nx,int ny,int nz,float half_width) {
    auto cell=urban_voxel_cell(p,d,lo,hi,pitch,nx,ny,nz);
    if(cell.valid)return cell;
    cell=urban_voxel_cell(p,d,{-half_width,-half_width,lo.z},
        {half_width,half_width,hi.z},{2*half_width,2*half_width,hi.z-lo.z},1,1,1);
    if(!cell.valid)return cell;
    auto& safety=cell.safety;safety.kind=2;
    safety.inner_x0=lo.x;safety.inner_x1=hi.x;safety.inner_y0=lo.y;safety.inner_y1=hi.y;
    float enter=-std::numeric_limits<float>::infinity(),leave=std::numeric_limits<float>::infinity();
    float tx=enter,ty=enter;
    auto axis=[&](float x,float dx,float l,float h,float& entry) {
        if(dx==0)return x>=l && x<=h;
        float a=(l-x)/dx,b=(h-x)/dx;if(a>b){float t=a;a=b;b=t;}
        entry=a;enter=sycl::fmax(enter,a);leave=sycl::fmin(leave,b);return true;
    };
    if(axis(p.x,d.x,lo.x,hi.x,tx) && axis(p.y,d.y,lo.y,hi.y,ty) &&
       enter>0 && leave>=enter && enter<cell.face_distance) {
        cell.face_distance=enter;cell.face_mask=0;
        cell.face_position={d.x>0?lo.x:hi.x,d.y>0?lo.y:hi.y,0};
        if(tx==enter)cell.face_mask|=1;if(ty==enter)cell.face_mask|=2;
        safety.box_boundary_path_mm=enter;
    }
    return cell;
}
// Included after sycl_device_math.inc, inside the same anonymous namespace.
// Proposal and scattering are separate: nuclear competition shortens the
// accepted TRUE path before a single angular/lateral sample is committed.
struct UrbanGlobalProposal {
    UrbanV2Material material{};
    UrbanV2LimitProposal limit{};
    UrbanV2TrackState state{};
    UrbanVoxelCell cell{};
    Direction3F position{},direction{};
    float energy{},true_path{},geom_path{},delta{};
    bool valid{};
};
inline UrbanGlobalProposal urban_global_prepare(
    UrbanMcsView view,int material,int ion,float energy,float true_ceiling,
    Direction3F position,Direction3F direction,UrbanVoxelCell cell,
    bool boundary,UrbanV2TrackState state,std::uint64_t seed,std::uint64_t history,
    std::uint64_t step) {
    UrbanGlobalProposal p{};p.state=state;p.cell=cell;p.position=position;p.direction=direction;p.energy=energy;
    const auto context=view.at(material,ion,energy);
    if(!context.valid() || !cell.valid || !(true_ceiling>0))return p;
    const auto& m=view.materials[material];const auto& s=view.species[ion];
    p.material.table.active_context=context;p.material.zeff=m.zeff;
    p.material.radlen_mm=m.radiation_length_mm;p.material.projectile_z=s.z;
    p.material.projectile_a=s.a;p.material.mass_mev=s.mass_mev;
    p.material.density_g_per_cm3=m.density;p.material.mfp_kind=2;
    p.limit=copper_urban_v2_limit_step_with_range(true_ceiling,energy,p.material,
        urban_v2_safety_at(position.x,position.y,position.z,cell.safety),boundary,
        p.state,context.range(energy),seed,history,step,0);
    if(!p.limit.valid)return p;
    p.geom_path=sycl::fmin(p.limit.g_msc_mm,cell.face_distance);
    bool guard=false;
    p.true_path=copper_urban_v2_finalize_true(p.geom_path,p.limit.g_msc_mm,
        p.limit.t_msc_mm,p.limit.delta_msc_mm,p.limit.lambda0_mm,p.limit.range_mm,
        p.limit.par1,p.limit.par3,p.delta,guard,p.limit.range_power_branch);
    p.valid=!guard && p.true_path>0 && p.geom_path>0 &&
        sycl::isfinite(p.true_path+p.geom_path+p.delta);
    return p;
}
inline CorrelatedScatteringStep urban_global_finalize(
    const UrbanGlobalProposal& p,float accepted_true,std::uint64_t seed,
    std::uint64_t history,std::uint64_t step) {
    CorrelatedScatteringStep result{p.direction,{0,0,0}};
    if(!p.valid || !(accepted_true>0) || accepted_true>p.true_path)return result;
    float geom=p.geom_path,delta=p.delta;
    if(accepted_true<p.true_path) {
        const auto converted=copper_urban_v2_true_to_geom(accepted_true,p.limit.lambda0_mm,
            p.limit.range_mm,p.energy,p.material);
        geom=converted.g_mm;delta=converted.delta_mm;
        if(!(geom>0) || geom>p.geom_path+4*urban_ulp_above(p.geom_path) ||
           !sycl::isfinite(geom+delta))return result;
    }
    if(accepted_true<p.limit.range_mm && accepted_true>kUrbanV2GeomMinMm) {
        const float safety=urban_reaches_voxel_face(geom,p.cell.face_distance)?0.f:
            urban_v2_safety_at(p.position.x+geom*p.direction.x,
                p.position.y+geom*p.direction.y,p.position.z+geom*p.direction.z,p.cell.safety);
        result=copper_urban_v2_sample_full(p.direction,p.energy,accepted_true,delta,
            p.material,p.limit.range_mm,p.limit.lambda0_mm,p.limit.tlimitmin_mm,
            safety,1.f,!p.limit.no_displacement,seed,history,step*1024U,160U);
    }
    result.proposed_true_path_mm=p.limit.t_msc_mm;
    result.final_true_path_mm=accepted_true;result.final_geom_path_mm=geom;
    result.stable_delta_mm=delta;result.current_range_mm=p.limit.range_mm;
    result.pre_step_energy_mev=p.energy;result.range_terminal=accepted_true>=p.limit.range_mm;
    result.boundary_crossed=urban_reaches_voxel_face(geom,p.cell.face_distance);
    result.proposal_valid=true;
    return result;
}
inline void urban_global_count(std::uint64_t* counts,int ion,bool secondary) {
    sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
        sycl::access::address_space::global_space>(counts[2*ion+(secondary?1:0)]).fetch_add(1);
}

// Dedicated ordered joint-path worker. Atomics score FP32 packets; double
// in/out totals are energy bookkeeping, not dose scoring.
const auto task=fe_joint_tasks[slot];
const auto channel=electron_joint_channels_device[task.channel];
const auto sample=select_fe_joint_sample(channel,static_cast<double>(task.cdf_u),electron_joint_samples_device);
const auto range=electron_path_ranges_device[sample];
const float path_cos=sycl::cos(task.phi),path_sin=sycl::sin(task.phi);
const auto ex=rotate_local_direction(path_cos,path_sin,0,
    Direction3F{task.direction[0],task.direction[1],task.direction[2]});
const auto ey=rotate_local_direction(-path_sin,path_cos,0,
    Direction3F{task.direction[0],task.direction[1],task.direction[2]});
const std::array<float,3> origin_f{ct_origin_x,ct_origin_y,ct_origin_z};
const std::array<float,3> spacing_f{ct_spacing_x,ct_spacing_y,ct_spacing_z};
const std::array<int,3> dims_i{static_cast<int>(ct_nx),static_cast<int>(ct_ny),static_cast<int>(ct_nz)};
auto density_f=[&](const std::array<int,3>& c) {
    return ct_density_device[(static_cast<std::size_t>(c[2])*ct_ny+c[1])*ct_nx+c[0]];
};
MassPathResult path;
path.endpoint={task.birth[0],task.birth[1],task.birth[2]};
const auto stored_bounds=electron_path_bounds_device[sample];
MassPathBoundsT<float> bounds_f;
for(int bi=0;bi<3;++bi) {
    bounds_f.low[bi]=static_cast<float>(stored_bounds.low[bi]);
    bounds_f.high[bi]=static_cast<float>(stored_bounds.high[bi]);
    bounds_f.net[bi]=static_cast<float>(stored_bounds.net[bi]);
}
const std::array<std::array<float,3>,3> basis_f{
    {{ex.x,ex.y,ex.z},{ey.x,ey.y,ey.z},{task.direction[0],task.direction[1],task.direction[2]}}};
std::array<float,3> probe_f=task.birth;
const bool same_cell=try_same_voxel_mass_path<float>(probe_f,
    bounds_f,basis_f,origin_f,spacing_f,dims_i,density_f);
if(same_cell) {
    path.endpoint={probe_f[0],probe_f[1],probe_f[2]};path.completed_segments=range.count;
} else path=replay_mass_polyline_indexed<float>(task.birth,range.count,
    [&](std::size_t j) {
        const auto v=electron_path_vectors_device[range.offset+j];
        return std::array<float,3>{
            static_cast<float>(v[0])*ex.x+static_cast<float>(v[1])*ey.x+static_cast<float>(v[2])*task.direction[0],
            static_cast<float>(v[0])*ex.y+static_cast<float>(v[1])*ey.y+static_cast<float>(v[2])*task.direction[1],
            static_cast<float>(v[0])*ex.z+static_cast<float>(v[1])*ey.z+static_cast<float>(v[2])*task.direction[2]};
    },origin_f,spacing_f,dims_i,density_f);
std::size_t target=task.source_voxel;int target_z=static_cast<int>(task.source_bin);
if(!path.invalid && !path.escaped) {
    const int x=static_cast<int>(sycl::floor((path.endpoint[0]-ct_origin_x)/ct_spacing_x));
    const int y=static_cast<int>(sycl::floor((path.endpoint[1]-ct_origin_y)/ct_spacing_y));
    const int z=static_cast<int>(sycl::floor((path.endpoint[2]-ct_origin_z)/ct_spacing_z));
    if(x<0 || y<0 || z<0 || x>=static_cast<int>(ct_nx) || y>=static_cast<int>(ct_ny) || z>=static_cast<int>(ct_nz))path.escaped=true;
    else {target=(static_cast<std::size_t>(z)*ct_ny+y)*ct_nx+x;target_z=z;}
}
auto counter=[&](int index,std::uint64_t amount) {
    if(!enable_electron_joint_diagnostics && index!=1 && index!=2)return;
    sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
        sycl::access::address_space::global_space> a(electron_joint_diag_device[index]);a.fetch_add(amount);
};
auto add=[](auto* address,float value) {
    using T=std::remove_pointer_t<decltype(address)>;
    sycl::atomic_ref<T,sycl::memory_order::relaxed,sycl::memory_scope::device,
        sycl::access::address_space::global_space> a(*address);a.fetch_add(static_cast<T>(value));
};
if(path.invalid) {
    counter(2,1);
    // Native invalid-march behavior retains the packet at its birth scorer.
    target=task.source_voxel;target_z=static_cast<int>(task.source_bin);
} else if(path.escaped) {
    counter(4,static_cast<std::uint64_t>(static_cast<double>(task.packet_MeV)*1e6));
    sycl::atomic_ref<double,sycl::memory_order::relaxed,sycl::memory_scope::device,
        sycl::access::address_space::global_space> in_grid(*grid_deposited_in_device);
    in_grid.fetch_add(-static_cast<double>(task.packet_MeV));
    grid_deposit_split_device(grid_deposited_in_device,grid_deposited_out_device,false,task.packet_MeV);
} else counter(3,static_cast<std::uint64_t>(static_cast<double>(task.packet_MeV)*1e6));
if(path.invalid || !path.escaped) {
    add(voxel_dose_device+target,task.packet_MeV);
    add(dose_device+target_z,task.packet_MeV);
    if(in_fov_dose_device)add(in_fov_dose_device+target_z,task.packet_MeV);
}

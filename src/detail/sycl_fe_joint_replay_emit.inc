// Compile-time separate from inline replay. Parent RNG domains and rounding
// of deposited-packet are retained. The independent stage adds the endpoint
// packet and corrects only the scorer in/out ledger on escape.
if(use_electron_joint && in_ct && enable_voxel_scoring &&
   voxel_index<number_of_voxels && deposited_MeV>0) {
    auto counter=[&](int slot,std::uint64_t amount) {
        if(!enable_electron_joint_diagnostics && slot!=1 && slot!=2)return;
        sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,
            sycl::memory_scope::device,sycl::access::address_space::global_space>
            a(electron_joint_diag_device[slot]);a.fetch_add(amount);
    };
    counter(0,1);
    const float cdf_u=rng::uniform01(spot_seed,rng_history,steps,21);
    const auto probe=probe_fe_joint_channel(ct_material,energy_MeVu,cdf_u,
        electron_joint_channels_device,electron_joint_samples_device,
        electron_joint_minimum,electron_joint_maximum,electron_joint_ceiling);
    if(probe.status!=ElectronJointStatus::hit) {
        counter(probe.status==ElectronJointStatus::energy_domain ? 1 : 2,1);
        counter(5,static_cast<std::uint64_t>(static_cast<double>(deposited_MeV)*1e6));
    } else if(probe.fraction>0) {
        const float fraction=static_cast<float>(probe.fraction);
        const float probability=sycl::fmax(electron_joint_sample_probability,fraction);
        if(probability>=1.f || rng::uniform01(spot_seed,rng_history,steps,903U)<probability) {
            const float packet=deposited_MeV*(electron_joint_sample_probability==1.f ? fraction : fraction/probability);
            if(packet>0) {
                const double phi=6.2831853071795864769*rng::uniform01(spot_seed,rng_history,steps,22);
                const double birth=rng::uniform01(spot_seed,rng_history,steps,20);
                FeJointReplayTask task;
                task.birth={position_x_mm+static_cast<float>(birth)*step_mm*direction_x,
                            position_y_mm+static_cast<float>(birth)*step_mm*direction_y,
                            position_z_mm+static_cast<float>(birth)*step_mm*direction_z};
                task.direction={direction_x,direction_y,direction_z};
                task.phi=static_cast<float>(phi);task.cdf_u=cdf_u;task.packet_MeV=packet;
                task.channel=probe.channel;task.source_voxel=static_cast<std::uint32_t>(voxel_index);
                task.source_bin=static_cast<std::uint32_t>(bin);
                const auto slot=sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,
                    sycl::memory_scope::device,sycl::access::address_space::global_space>
                    (fe_joint_queue_counts[0]).fetch_add(1);
                if(slot<fe_joint_queue_capacity)fe_joint_tasks[slot]=task;
                // Overflow rejects the entire batch on the host before replay.
                counter(6,1);
                local_voxel_deposit_MeV-=packet;forward_shifted_MeV+=packet;
            }
        }
    }
}

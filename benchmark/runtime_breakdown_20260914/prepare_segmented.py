from pathlib import Path
import shutil,json,hashlib,difflib
repo=Path(__file__).resolve().parents[2];root=repo/'scratch/runtime_breakdown_20260914';out=root/'segmented_source'
assert not out.exists();shutil.copytree(root/'mean_guard_source',out)
p=out/'src/transport_sycl.cpp';s=p.read_text()
def one(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
fields=[('bool','sec_terminal_recorded'),('bool','unified_secondary_escaped_ct'),('ContinuousSpeciesTrackTally','continuous_species_tally')]+[('float',n) for n in ['sec_e','sec_x','sec_y','sec_z','sec_dx','sec_dy','sec_dz','pending_sec_depth_MeV','pending_sec_voxel_MeV']]+[('int',n) for n in ['pending_sec_bin','pending_sec_voxel']]+[('UnifiedEmClock','unified_secondary_clock'),('std::uint64_t','unified_secondary_counter'),('UnifiedEmState','unified_secondary_state'),('std::array<std::uint64_t,8>','unified_secondary_audit'),('uint32_t','sec_steps'),('std::uint64_t','local_sec_rate_queries'),('std::uint64_t','local_sec_steps')]
struct='struct SecondaryResumeState {\n'+''.join(f'    {t} {n}{{}};\n' for t,n in fields)+'    std::array<double,6> he4_audit{};\n    std::array<std::uint64_t,CARBON_SECONDARY_STEP_PROFILE?60:0> sec_prof{};\n};\n'
one('struct UnifiedEmFailureRecord {',struct+'struct UnifiedEmFailureRecord {')
anchor='                auto sec_event = queue.submit([&](sycl::handler& cgh) {'
one(anchor,'''                const unsigned resume_capacity=generation_end-generation_begin;
                auto* resume_states=mem_tracker.allocate<SecondaryResumeState>(resume_capacity);
                auto* resume_ready=mem_tracker.allocate<unsigned>(resume_capacity);
                auto* active_order=mem_tracker.allocate<unsigned>(resume_capacity);
                auto* next_order=mem_tracker.allocate<unsigned>(resume_capacity);
                auto* keep=mem_tracker.allocate<unsigned>(resume_capacity);
                auto* ranks=mem_tracker.allocate<unsigned>(resume_capacity);
                auto* block_counts=mem_tracker.allocate<unsigned>((resume_capacity+255)/256);
                auto* block_offsets=mem_tracker.allocate<unsigned>((resume_capacity+255)/256);
                auto* active_count=mem_tracker.allocate<unsigned>(1);
                if(!resume_states||!resume_ready||!active_order||!next_order||!keep||!ranks||!block_counts||!block_offsets||!active_count)throw std::bad_alloc();
                queue.fill(resume_ready,0u,resume_capacity).wait_and_throw();
                queue.parallel_for(sycl::range<1>(resume_capacity),[=](sycl::id<1> i){active_order[i[0]]=i[0];}).wait_and_throw();
                unsigned resume_active=resume_capacity,segment_rounds=0;
                double compact_seconds=0;
                std::cout<<"[segmented-secondary] state_bytes="<<sizeof(SecondaryResumeState)<<" capacity="<<resume_capacity<<"\\n";
                while(resume_active){
                const bool finish_tail=resume_active<8192;
'''+anchor)
one('sycl::range<1>(generation_end - generation_begin),\n                    [=](sycl::id<1> item_id)', 'sycl::range<1>(resume_active),\n                    [=](sycl::id<1> item_id)')
one('''                        const auto sec_idx = group_secondaries ? secondary_order[item_id[0]]
                            : generation_begin + item_id[0];''','''                        const unsigned state_idx=active_order[item_id[0]];
                        const bool resumed=resume_ready[state_idx]!=0;
                        keep[item_id[0]]=0;
                        const auto sec_idx = group_secondaries ? secondary_order[state_idx]
                            : generation_begin + state_idx;''')
# Birth accounting is once per original track; registry lookup remains per launch.
a='''                        const int schneider_reg_idx =
                            secondary_projectile_lut_index_device(
                                      schneider_ct_device_ctx.sec_proj_keys,
                                      schneider_ct_device_ctx.sec_num_projectiles,
                                      frag.z, frag.a);'''
assert s.count(a)==1;s=s.replace(a,'')
one('''                        // A track that reaches the stepping loop has actually''',a+'''\n                        if(!resumed){
                        // A track that reaches the stepping loop has actually''')
restore='''                        }
                        if(resumed){
                            const auto saved=resume_states[state_idx];
'''+''.join(f'                            {n}=saved.{n};\n' for t,n in fields)+'''                            for(int j=0;j<6;++j)he4_audit[j]=saved.he4_audit[j];
                            if constexpr(CARBON_SECONDARY_STEP_PROFILE)for(int j=0;j<60;++j)sec_prof[j]=saved.sec_prof[j];
                        }
                        unsigned segment_steps=0;bool segment_paused=false;
'''
one('                        while (sec_e > energy_cutoff_MeV',restore+'                        while (sec_e > energy_cutoff_MeV')
one('''                               sec_steps < kSecondaryMaxSteps) {
                            const auto bin_z''','''                               sec_steps < kSecondaryMaxSteps) {
                            if(!finish_tail && segment_steps>=64){segment_paused=true;break;}
                            ++segment_steps;
                            const auto bin_z''')
save='''                        if(segment_paused){
                            SecondaryResumeState saved;
'''+''.join(f'                            saved.{n}={n};\n' for t,n in fields)+'''                            for(int j=0;j<6;++j)saved.he4_audit[j]=he4_audit[j];
                            if constexpr(CARBON_SECONDARY_STEP_PROFILE)for(int j=0;j<60;++j)saved.sec_prof[j]=sec_prof[j];
                            resume_states[state_idx]=saved;resume_ready[state_idx]=1;keep[item_id[0]]=1;
                            return; // Suspend: no terminal scoring or audit flush.
                        }
'''
one('                        if(unified_secondary && CARBON_EM_LOCAL_AUDIT)flush_unified_em_audit',save+'                        if(unified_secondary && CARBON_EM_LOCAL_AUDIT)flush_unified_em_audit')
one('''                secondary_kernel_seconds += event_duration_seconds(sec_event);
                const auto batch_end''','''                secondary_kernel_seconds += event_duration_seconds(sec_event);
                ++segment_rounds;
                const auto compact_start=std::chrono::steady_clock::now();
                const unsigned resume_groups=(resume_active+255)/256;
                queue.parallel_for(sycl::nd_range<1>(resume_groups*256,256),[=](sycl::nd_item<1> it){
                    const auto i=it.get_global_linear_id();const unsigned flag=i<resume_active?keep[i]:0;
                    const auto rank=sycl::exclusive_scan_over_group(it.get_group(),flag,sycl::plus<unsigned>());
                    const auto count=sycl::reduce_over_group(it.get_group(),flag,sycl::plus<unsigned>());
                    if(i<resume_active)ranks[i]=rank;
                    if(it.get_local_linear_id()==0)block_counts[it.get_group_linear_id()]=count;
                }).wait_and_throw();
                queue.single_task([=](){unsigned sum=0;for(unsigned i=0;i<resume_groups;++i){block_offsets[i]=sum;sum+=block_counts[i];}*active_count=sum;}).wait_and_throw();
                queue.parallel_for(sycl::range<1>(resume_active),[=](sycl::id<1> id){const auto i=id[0];if(keep[i])next_order[block_offsets[i/256]+ranks[i]]=active_order[i];}).wait_and_throw();
                queue.copy(active_count,&resume_active,1).wait_and_throw();std::swap(active_order,next_order);
                compact_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-compact_start).count();
                }
                std::cout<<"[segmented-secondary] rounds="<<segment_rounds<<" compaction_s="<<compact_seconds<<"\\n";
                mem_tracker.free(resume_states);
                for(auto* ptr:{resume_ready,active_order,next_order,keep,ranks,block_counts,block_offsets,active_count})mem_tracker.free(ptr);
                const auto batch_end''')
p.write_text(s)
dest=Path(__file__).resolve().parent
(dest/'segmented.patch').write_text(''.join(difflib.unified_diff((root/'mean_guard_source/src/transport_sycl.cpp').read_text().splitlines(True),s.splitlines(True),fromfile='a/src/transport_sycl.cpp',tofile='b/src/transport_sycl.cpp')))
(dest/'segmented_source_manifest.json').write_text(json.dumps({str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob('*')) if f.is_file()},indent=2)+'\n')

from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[2]/'scratch/runtime_breakdown_20260914'
for source,dest in [('mean_guard_source','terminal_base_source'),('segmented_source','terminal_segment_source')]:
 out=root/dest;assert not out.exists();shutil.copytree(root/source,out);p=out/'src/transport_sycl.cpp';s=p.read_text()
 def one(a,b):
  global s
  assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
 one('struct UnifiedEmFailureRecord {','''struct SecondaryTerminalRecord {
    std::uint64_t stream,rng;
    std::uint32_t parent,generation,z,a,steps,flags,bin,voxel;
    float values[16];
};
static_assert(sizeof(SecondaryTerminalRecord)==112);
struct UnifiedEmFailureRecord {''')
 one('    double secondary_kernel_seconds = 0.0;','    std::ofstream terminal_file("secondary_terminal.bin",std::ios::binary);\n    if(!terminal_file)throw std::runtime_error("terminal output open");\n    double secondary_kernel_seconds = 0.0;')
 anchor='                const unsigned resume_capacity=' if 'segmented' in source else '                auto sec_event ='
 i=s.index(anchor);s=s[:i]+'''                auto* terminal_records=mem_tracker.allocate<SecondaryTerminalRecord>(generation_end-generation_begin);
                if(!terminal_records)throw std::bad_alloc();
                queue.fill(terminal_records,SecondaryTerminalRecord{},generation_end-generation_begin).wait_and_throw();
'''+s[i:]
 one('                        if(unified_secondary && CARBON_EM_LOCAL_AUDIT)flush_unified_em_audit','''                        terminal_records[sec_idx-generation_begin]={frag.rng_stream,unified_secondary_counter,
                            frag.parent_history,unsigned(frag.generation),unsigned(frag.z),unsigned(frag.a),sec_steps,
                            1u|(unsigned(unified_secondary_clock.clock.active)<<1)|(unsigned(sec_terminal_recorded)<<2)|(unsigned(unified_secondary_escaped_ct)<<3),unsigned(pending_sec_bin),unsigned(pending_sec_voxel),
                            {sec_e,sec_x,sec_y,sec_z,sec_dx,sec_dy,sec_dz,pending_sec_depth_MeV,pending_sec_voxel_MeV,
                             unified_secondary_clock.clock.remaining,unified_secondary_clock.threshold,unified_secondary_clock.rate0,unified_secondary_clock.rate1,unified_secondary_clock.last_density,
                             continuous_species_tally.all_MeV,continuous_species_tally.fov_MeV}};
                        if(unified_secondary && CARBON_EM_LOCAL_AUDIT)flush_unified_em_audit''')
 one('                const auto batch_end = generation_end;','''                { std::vector<SecondaryTerminalRecord> records(generation_end-generation_begin);
                  queue.copy(terminal_records,records.data(),records.size()).wait_and_throw();
                  terminal_file.write(reinterpret_cast<const char*>(records.data()),records.size()*sizeof(SecondaryTerminalRecord));
                  if(!terminal_file)throw std::runtime_error("terminal output write"); }
                mem_tracker.free(terminal_records);
                const auto batch_end = generation_end;''')
 p.write_text(s)

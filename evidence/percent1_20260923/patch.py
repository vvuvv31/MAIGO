from pathlib import Path
import hashlib,shutil,json,subprocess
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/percent1_20260923';p=r/'src/transport_sycl.cpp';s=p.read_text();binary=r/'build/oneapi-nvidia-minibeam/carbon_mc'
expected=json.loads((o/'gpu_quality.json').read_text())['binary_sha256'];assert hashlib.sha256(binary.read_bytes()).hexdigest()==expected
shutil.copy2(binary,o/'carbon_mc_before_delta_fix');(o/'transport_sycl.before_delta_fix.cpp').write_text(s)
a='''    const auto voxel_bins_z = config.voxel_bins_z;
    const auto voxel_size_x_mm = static_cast<float>(config.voxel_size_x_mm);
    const auto voxel_size_y_mm = static_cast<float>(config.voxel_size_y_mm);
    const auto voxel_size_z_mm = static_cast<float>(config.voxel_size_z_mm);'''
b='''    // Resolve the legacy water grid exactly as its allocation/header do.
    // Raw voxel_bins_z/voxel_size_z_mm can both be zero when the grid is
    // defined by phantom_length_mm and depth_bin_width_mm.
    const auto voxel_bins_z = number_of_bins;
    const auto voxel_size_x_mm = static_cast<float>(config.voxel_size_x_mm);
    const auto voxel_size_y_mm = static_cast<float>(config.voxel_size_y_mm);
    const auto voxel_size_z_mm = static_cast<float>(config.scorer_spacing_z_mm());'''
assert s.count(a)==1;s=s.replace(a,b)
start=s.index('                    if (minibeam_water_delta_v1 && unified_em &&')
end=s.index('                    if(use_material_ct && in_ct && deposited_MeV>0)',start)
chunk=s[start:end]
# Existing minibeam delta semantics classify out-of-volume packets as physical
# escape. Do not also classify that same packet as energy deposited outside
# the scorer: the generic grid split would subtract it a second time.
chunk=chunk.replace('''                                        delta_tail_escaped_scorer_MeV +=
                                            relocated;
                                        water_physical_escape_MeV += relocated;''','''                                        water_physical_escape_MeV += relocated;''')
chunk=chunk.replace('''                                    delta_tail_escaped_scorer_MeV += relocated;
                                    water_physical_escape_MeV += relocated;''','''                                    water_physical_escape_MeV += relocated;''')
assert 'delta_tail_escaped_scorer_MeV' not in chunk
needle='''                                        add_delta(voxel_dose_device + target);
                                        if (dz >= 0 &&'''
replacement='''                                        add_delta(voxel_dose_device + target);
                                        if ((!kProductionPrimaryPath && enable_charged_origin_voxel_scoring))
                                            add_delta(charged_origin_voxel_dose_device + target);
                                        if ((!kProductionPrimaryPath && enable_minibeam_component_voxel_scoring))
                                            add_delta(minibeam_component_voxel_dose_device + target);
                                        if (dz >= 0 &&'''
assert chunk.count(needle)==1;chunk=chunk.replace(needle,replacement)
assert chunk.endswith('''                        }
                    }
''')
chunk=chunk[:-len('''                    }
''')]+'''                        // Escaping packet/photon energy is absent from deposition
                        // and must be returned through the history escape ledger.
                        // Count it once; physical escape is not an outside-grid deposit.
                        history_water_electron_escaped_MeV += water_physical_escape_MeV;
                    }
'''
s=s[:start]+chunk+s[end:];p.write_text(s)
subprocess.run(['git','diff','--check'],cwd=r,check=True)
with (o/'delta_scoring_fix.patch').open('w') as f:subprocess.run(['diff','-u',str(o/'transport_sycl.before_delta_fix.cpp'),str(p)],stdout=f)
print('PATCHED',hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)

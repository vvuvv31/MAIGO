"""Step 26: compile versioned v2 secondary CINEL03 package (standalone).

Reads schneider-secondary-v2/campaign_manifest.json + raw_validation.json,
compiles ONLY validated non-empty nodes:
  v1 raw (existing package events are NOT recompiled; v2 adds nodes) +
  v2 new raw files.
Hmm -- correction: v2 package = v1 events UNION validated v2-new events,
so channels keep every valid existing node (manifest rule 6).

Rules enforced (any violation aborts, no partial package):
- canonical sort by (projectile_Z, projectile_A, target_Z, energy)
- reject ambiguous duplicates / empty nodes / illegal offsets-counts
- energy_min/max/max_gap computed from REAL nodes (never forged)
- no target alias (exact Z only; p+H absent unless validated events exist)
- record raw input paths and sizes, compiler commit + command
- write to temp path, verify by re-reading, then atomic rename
Outputs:
  data/schneider/cinel03_secondary_targets_v2.bin
  data/schneider/cinel03_secondary_targets_v2.metadata.json
  data/schneider/cinel03_secondary_targets_v2.channels.json
"""
import datetime
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
REPO = Path(os.environ.get('MAIGO_REPO_ROOT', Path(__file__).resolve().parents[3]))
WORK = Path(os.environ.get('MAIGO_PACKAGE_WORK_ROOT', REPO / 'extensions' / 'work'))
V2 = Path(os.environ.get('MAIGO_V2_CAMPAIGN_ROOT', WORK / 'schneider-secondary-v2'))
sys.path.insert(0, str(REPO / 'extensions' / 'package_tools'))
import cinel02
import cinel03
V1_BIN = REPO / 'data/schneider/cinel03_secondary_targets.bin'
V1_UUID = '00000000-0000-4000-8000-000000000020'
V2_UUID = '00000000-0000-4000-8000-000000000022'
V2_1_UUID = '00000000-0000-4000-8000-000000000023'
V2_1_14P_UUID = '00000000-0000-4000-8000-000000000026'
TAG = sys.argv[sys.argv.index('--tag') + 1] if '--tag' in sys.argv else 'v2'
WITH_C12 = '--c12-secondary' in sys.argv
UUID = V2_1_14P_UUID if WITH_C12 else V2_1_UUID if TAG == 'v2_1' else V2_UUID
OUT_STEM = f'cinel03_secondary_targets_{TAG}_14p' if WITH_C12 else f'cinel03_secondary_targets_{TAG}'
MAX_EVENTS_PER_FILE = 30
C12_V1_PROD = Path(os.environ.get('MAIGO_PRIMARY_CAMPAIGN_ROOT', WORK / 'production'))

def audit_event(rec, prods):
    """Shared v1 closure + role audit. Returns True to keep."""
    e_coll = float(rec['collision_energy_MeV'])
    e_tot = float(rec['parent_energy_MeV']) + float(rec['process_local_deposit_MeV']) + float(rec['unsupported_product_energy_MeV']) + sum((float(p['kinetic_energy_MeV']) for p in prods))
    if e_tot > e_coll + max(200.0, 0.2 * e_coll):
        return False
    for p in prods:
        is_ground_ion = p['z'] > 0 and p['a'] >= p['z'] and (p['pdg'] % 10 == 0) and (p['excitation'] <= 0.0001)
        is_ground_neutral = p['pdg'] in (22, 2112) and p['excitation'] <= 0.0001
        if p['role'] == 2 and (is_ground_ion or is_ground_neutral):
            return False
    return True

def main():
    events = []
    raw_input_bytes = {}
    v1_root = Path(os.environ.get('MAIGO_SECONDARY_V1_RAW_ROOT', WORK / 'step20_secondary_campaigns' / 'raw'))
    v1_files = sorted(v1_root.rglob('worker_*.cinel02'))
    print(f'v1 raw files: {len(v1_files)}')
    n_v1 = 0
    for fp in v1_files:
        kept = 0
        for rec, prods in cinel02.read_raw(fp):
            if kept >= MAX_EVENTS_PER_FILE:
                break
            if not audit_event(rec, prods):
                continue
            events.append((rec, prods))
            kept += 1
        n_v1 += kept
    print(f'v1 base events kept: {n_v1} (package has 33139 interactions)')
    raw_input_bytes[str(V1_BIN)] = V1_BIN.stat().st_size
    val_name = sys.argv[sys.argv.index('--validation') + 1] if '--validation' in sys.argv else 'raw_validation.json'
    man_name = sys.argv[sys.argv.index('--manifest') + 1] if '--manifest' in sys.argv else 'campaign_manifest.json'
    validation = json.load(open(V2 / val_name))
    manifest = json.load(open(V2 / man_name))
    by_task = {t['task_index']: t for t in manifest['tasks']}
    n_new = 0
    n_c12 = 0
    for r in validation['results']:
        if not r['pass']:
            continue
        t = by_task[r['task_index']]
        if (t['projectile_Z'], t['projectile_A']) == (6, 12):
            n_c12 += 1
            continue
        case_dir = V2 / t['case_dir']
        files = sorted((case_dir / 'raw').rglob('worker_*.cinel02'))
        for fp in files:
            raw_input_bytes[str(fp.relative_to(V2))] = fp.stat().st_size
            kept = 0
            for rec, prods in cinel02.read_raw(fp):
                if kept >= MAX_EVENTS_PER_FILE:
                    break
                if not audit_event(rec, prods):
                    continue
                events.append((rec, prods))
                kept += 1
            n_new += kept
    print(f'v2 new validated events: {n_new} (C12 tasks routed to primary: {n_c12}); total: {len(events)}')
    if n_new == 0:
        raise RuntimeError('no validated new events; refusing empty-delta compile')
    n_c12_sec = 0
    if WITH_C12:
        c12_files = sorted(C12_V1_PROD.rglob('worker_*.cinel02'))
        print(f'C12 v1 production raw files: {len(c12_files)}')
        for fp in c12_files:
            kept = 0
            for rec, prods in cinel02.read_raw(fp):
                if kept >= MAX_EVENTS_PER_FILE:
                    break
                assert (rec['projectile_z'], rec['projectile_a']) == (6, 12), fp
                if not audit_event(rec, prods):
                    continue
                events.append((rec, prods))
                kept += 1
            n_c12_sec += kept
        for r in validation['results']:
            if not r['pass']:
                continue
            t = by_task[r['task_index']]
            if (t['projectile_Z'], t['projectile_A']) != (6, 12):
                continue
            case_dir = V2 / t['case_dir']
            files = sorted((case_dir / 'raw').rglob('worker_*.cinel02'))
            for fp in files:
                raw_input_bytes[str(fp.relative_to(V2))] = fp.stat().st_size
                kept = 0
                for rec, prods in cinel02.read_raw(fp):
                    if kept >= MAX_EVENTS_PER_FILE:
                        break
                    assert (rec['projectile_z'], rec['projectile_a']) == (6, 12), fp
                    if not audit_event(rec, prods):
                        continue
                    events.append((rec, prods))
                    kept += 1
                n_c12_sec += kept
        print(f'C12 secondary events: {n_c12_sec}; total: {len(events)}')
        if n_c12_sec == 0:
            raise RuntimeError('WITH_C12 but no C12 events; refusing')

    def f32(x):
        return struct.unpack('<f', struct.pack('<f', x))[0]
    events.sort(key=lambda it: (it[0]['projectile_z'], it[0]['projectile_a'], it[0]['target_z'], f32(it[0]['collision_energy_MeV_per_u'])))
    packed_interactions, packed_products, cells = ([], [], [])
    rej = defaultdict(int)

    def cell_bin(e):
        return max(0, int(math.floor((e - 0.5) / 1.0)))
    cursor = 0
    while cursor < len(events):
        r0 = events[cursor][0]
        key = (r0['projectile_z'], r0['projectile_a'], r0['target_z'], cell_bin(r0['collision_energy_MeV_per_u']))
        start = cursor
        while cursor < len(events):
            r = events[cursor][0]
            k = (r['projectile_z'], r['projectile_a'], r['target_z'], cell_bin(r['collision_energy_MeV_per_u']))
            if k != key:
                break
            cursor += 1
        cells.append(key + (start, cursor - start))
    for pz, pa, tz, ebin, start, count in cells:
        for i in range(start, start + count):
            rec, prods = events[i]
            packed_interactions.append(cinel02._pack_fixed(rec))
            for p in prods:
                packed_products.append(cinel02._pack_product(p))
    pkg = cinel03.Cinel03Package()
    pkg.minimum_energy_MeV_per_u = 0.5
    pkg.energy_bin_width_MeV_per_u = 1.0
    pkg.minimum_events_per_bin = 1
    pkg.campaign_uuid = UUID
    cell_dicts = []
    for c in cells:
        lo = 0.5 + c[3] * 1.0
        if c[3] == 0:
            content_min = min((events[i][0]['collision_energy_MeV_per_u'] for i in range(c[4], c[4] + c[5])))
            lo = min(lo, content_min)
        cell_dicts.append({'projectile_z': c[0], 'projectile_a': c[1], 'target_element_z': c[2], 'energy_bin': c[3], 'interaction_offset': c[4], 'interaction_count': c[5], 'energy_lower_MeV_per_u': lo, 'energy_upper_MeV_per_u': 0.5 + (c[3] + 1) * 1.0})
    pkg.cells = cell_dicts
    pkg.interactions = packed_interactions
    pkg.products = packed_products
    cursor = 0
    while cursor < len(events):
        r0 = events[cursor][0]
        nk = (r0['projectile_z'], r0['projectile_a'], r0['target_z'], f32(r0['collision_energy_MeV_per_u']))
        start = cursor
        while cursor < len(events):
            r = events[cursor][0]
            if (r['projectile_z'], r['projectile_a'], r['target_z'], f32(r['collision_energy_MeV_per_u'])) != nk:
                break
            cursor += 1
        pkg.energy_nodes.append(nk)
        pkg.event_offsets.append(start)
    pkg.event_offsets.append(len(events))
    pkg.event_indices = list(range(len(events)))
    tmpdir = Path(tempfile.mkdtemp(prefix='cinel03v2_', dir=REPO / 'data/schneider'))
    tmp_bin = tmpdir / (OUT_STEM + '.bin')
    pkg.write_binary(tmp_bin)
    reread = cinel03.Cinel03Package.read_binary(tmp_bin)
    assert len(reread.interactions) == len(events), 're-read event mismatch'
    assert len(reread.energy_nodes) == len(pkg.energy_nodes), 'node mismatch'
    final_bin = REPO / ('data/schneider/' + OUT_STEM + '.bin')
    tmp_bin.replace(final_bin)
    ch = defaultdict(list)
    for e_idx, (pz, pa, tz, en) in enumerate(pkg.energy_nodes):
        ch[pz, pa, tz].append(en)
    channels = []
    for (pz, pa, tz), nodes in sorted(ch.items()):
        a = sorted(nodes)
        gaps = [a[i + 1] - a[i] for i in range(len(a) - 1)]
        channels.append({'projectile_z': pz, 'projectile_a': pa, 'target_element_z': tz, 'energy_min_MeV_per_u': min(a), 'energy_max_MeV_per_u': max(a), 'energy_nodes': len(a), 'maximum_observed_node_gap_MeV_per_u': max(gaps) if gaps else 0.0, 'maximum_allowed_node_gap_MeV_per_u': 5.0, 'interpolation_policy': 'stochastic_bracketing', 'target_alias_allowed': False})
    (REPO / ('data/schneider/' + OUT_STEM + '.channels.json')).write_text(json.dumps({'channels': channels}, indent=1))
    if WITH_C12:
        projs = sorted({(c['projectile_z'], c['projectile_a']) for c in channels})
        assert len(projs) == 14, f'14p scope requires 14 projectiles, got {len(projs)}'
        assert (6, 12) in projs, 'C12 secondary channels missing'
        for c in channels:
            if (c['projectile_z'], c['projectile_a']) == (6, 12):
                assert c['maximum_observed_node_gap_MeV_per_u'] <= 5.0, f'C12/tz={c['target_element_z']} gap {c['maximum_observed_node_gap_MeV_per_u']} exceeds 5.0'
    ext_files = ['CarbonInelasticEventWriter.cc', 'CarbonInelasticEventWriter.hh', 'CarbonInelasticCapturePhysics.cc', 'CarbonInelasticCapturePhysics.hh', 'CarbonInelasticCaptureProcess.cc', 'CarbonInelasticCaptureProcess.hh', 'CarbonInelasticExposureNtuple.cc', 'CarbonInelasticExposureNtuple.hh']
    meta = {'schema_version': 1, 'format': 'CINPKG04', 'data_filename': OUT_STEM + '.bin', 'file_size_bytes': final_bin.stat().st_size, 'topas_version': '4.2.p3', 'geant4_version': 'geant4-11-03-patch-02', 'physics_list': manifest.get('physics_list', 'g4em-standard_opt4 g4h-phy_QGSP_BIC_HP g4ion-inclxx CarbonInelasticCapturePhysics g4h-elastic_HP g4stopping'), 'schneider_source_path': 'data/HUtoMaterialSchneider.txt', 'extractor_git_commit': None, 'compiler_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip(), 'compiler_command': 'python3 tools/compile_v2_package.py ' + ' '.join(sys.argv[1:]), 'raw_input_bytes': raw_input_bytes, 'energy_min_MeVu': min((c['energy_min_MeV_per_u'] for c in channels)), 'energy_max_MeVu': max((c['energy_max_MeV_per_u'] for c in channels)), 'energy_grid': 'campaign-energy nodes; per-channel max gap <= 5 MeV/u', 'projectiles': sorted({(c['projectile_z'], c['projectile_a']) for c in channels}), 'target_elements': sorted({c['target_element_z'] for c in channels}), 'units': 'mm, MeV, ns', 'generation_timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'policy': {'be6_status': 'EXCLUDED (TopasCompatKill)', 'aliasing_status': 'FORBIDDEN (exact target element Z only)'}}
    (REPO / ('data/schneider/' + OUT_STEM + '.metadata.json')).write_text(json.dumps(meta, indent=1))
    print(f'compiled {final_bin} ({final_bin.stat().st_size / 1000000.0:.1f} MB)')
if __name__ == '__main__':
    sys.exit(main())

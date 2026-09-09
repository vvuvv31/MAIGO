#!/usr/bin/env python3
"""Compile independent Water_75eV EM records into a diagnostic 3D response.

No reference dose is an input. Samples retain both endpoints of the depositing
step, in its root electron's parent frame. The CDF is weighted by deposition,
not by track count. This is homogeneous-water data, NOT a Schneider section.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from compare_electron_response_geometries import load_v2_binary
from water_electron_path_graph import build_path_graph, NONE
from audit_schneider_response_scope import schneider_identity


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def audit_records(d, histories, bounds, *, expected_density_g_cm3=1.0):
    """Strict embedded-source, EM-only, unit-weight, no-reentry audit."""
    if d.ndim != 2 or d.shape[1] not in (34,46) or not len(d) or not np.isfinite(d).all():
        raise ValueError('Expected finite V3 or V4 records')
    if not np.isfinite(expected_density_g_cm3) or expected_density_g_cm3<=0:
        raise ValueError('Invalid expected homogeneous density')
    if (np.any(d[:, 21] != (4 if d.shape[1]==46 else 3)) or np.any(d[:, 19] != 1)
            or not np.allclose(d[:, 20], expected_density_g_cm3, rtol=1e-6, atol=1e-8)
            or np.any(~np.isin(d[:, 4], [1000060120, 11, 22]))
            or np.any(d[:, [9, 16, 17, 18]] < 0)):
        raise ValueError('Not unit-weight homogeneous C12 EM records at expected density')
    order = np.lexsort((d[:, 5], d[:, 2], d[:, 1], d[:, 0]))
    d = d[order]
    changed = np.any(d[1:, :3] != d[:-1, :3], axis=1)
    starts = np.r_[0, np.flatnonzero(changed) + 1]
    ends = np.r_[starts[1:] - 1, len(d) - 1]
    if np.any(d[starts, 5] != 1) or np.any((np.diff(d[:, 5]) != 1) & ~changed):
        raise ValueError('Missing/duplicate steps or reentry')
    first, last = d[starts], d[ends]
    if not np.allclose(first[:, 6:9], first[:, 10:13], rtol=0, atol=1e-8):
        raise ValueError('Unrecorded birth-to-first-step gap')
    keys = [tuple(x.astype(int)) for x in first[:, :3]]
    lookup = {k: i for i, k in enumerate(keys)}
    root = np.full(len(first), -2, dtype=int)
    for i in range(len(first)):
        chain, cur = [], i
        while root[cur] == -2:
            if cur in chain:
                raise ValueError('Cyclic ancestry')
            chain.append(cur)
            if first[cur, 3] == 0:
                if first[cur, 4] != 1000060120:
                    raise ValueError('Non-C12 primary')
                root[cur] = -1
                break
            pk = (*keys[cur][:2], int(first[cur, 3]))
            if pk not in lookup:
                raise ValueError('Missing ancestor')
            parent = lookup[pk]
            if first[parent, 3] == 0:
                r = first[cur]
                if r[4] != 11 or r[22] != 1 or r[27] != 1:
                    raise ValueError('Unbound primary electron')
                step = int(r[29])
                index = starts[parent] + step - 1
                if (step < 1 or index > ends[parent]
                        or abs(d[index, 17] - r[23]) > 1e-8
                        or abs(d[index, 18] - r[30]) > 1e-8):
                    raise ValueError('Generating parent-step energy mismatch')
                if not np.allclose(d[index, 13:16], r[6:9], rtol=0, atol=1e-8):
                    raise ValueError('Generating parent-step birth position mismatch')
                root[cur] = cur
                break
            cur = parent
        root[chain] = root[cur]
    primary_tracks = root == -1
    if np.count_nonzero(primary_tracks) != histories:
        raise ValueError('Primary history count mismatch')
    bounds = np.asarray(bounds, dtype=float).reshape(3, 2)
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError('Invalid bounds')
    escaped = last[:, 18] > 0
    pos = last[escaped, 13:16]
    direction = pos - last[escaped, 10:13]
    low = np.abs(pos - bounds[:, 0]) <= 1e-8
    high = np.abs(pos - bounds[:, 1]) <= 1e-8
    if (np.any(pos < bounds[:, 0] - 1e-8) or np.any(pos > bounds[:, 1] + 1e-8)
            or np.any(~np.any((low & (direction < 0)) | (high & (direction > 0)), axis=1))):
        raise ValueError('Unexplained positive-KE terminal or inward escape')
    # A charged escape would truncate precisely the long tail being modeled.
    if np.any(escaped & (last[:, 4] != 22)):
        raise ValueError('Charged escape: enlarge extraction volume, do not renormalize')
    track_dep = np.add.reduceat(d[:, 16], starts)
    dep, esc = np.zeros(len(first)), np.zeros(len(first))
    nonprimary = root >= 0
    np.add.at(dep, root[nonprimary], track_dep[nonprimary])
    np.add.at(esc, root[nonprimary], last[nonprimary, 18])
    families = np.flatnonzero(root == np.arange(len(root)))
    birth = first[families, 9]
    residual = birth - dep[families] - esc[families]
    if np.any(birth <= 0) or np.any(np.abs(residual) > 1e-3 * birth):
        raise ValueError('Individual electron-family closure failed')
    for ev in set(k[:2] for k in keys):
        take = np.array([k[:2] == ev for k in keys])
        incident = first[take & primary_tracks, 17].sum()
        if incident <= 0 or abs(incident - track_dep[take].sum() - last[take, 18].sum()) > 1e-3 * incident:
            raise ValueError('Per-event closure failed')
    row_track = np.repeat(np.arange(len(first)), ends - starts + 1)
    return d, first, root[row_track], families, dep, esc, dict(
        histories=histories, families=len(families),
        max_family_relative_residual=float(np.max(np.abs(residual) / birth)),
        charged_escaped_MeV=0., photon_escaped_MeV=float(esc.sum()))


def frame_segments(rows, roots):
    z = roots[:, 24:27]
    if not np.allclose(np.linalg.norm(z, axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError('Invalid parent frame')
    z = z/np.linalg.norm(z, axis=1)[:, None]
    axis = np.zeros_like(z)
    axis[np.arange(len(z)), np.argmin(np.abs(z), axis=1)] = 1
    x = np.cross(axis, z)
    x /= np.linalg.norm(x, axis=1)[:, None]
    y = np.cross(z, x)
    basis = np.stack([x, y, z], axis=1)
    result = np.column_stack([np.einsum('nij,nj->ni', basis, rows[:, c:c+3] - roots[:, 6:9])
                              for c in (10, 13)])
    # Photons have discrete local deposition at their interaction point,
    # not continuous charged-particle loss along the flight segment.
    photon = rows[:, 4] == 22
    result[photon, :3] = result[photon, 3:]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--energy-max', type=int, choices=(300,450), default=300)
    p.add_argument('--material-response', action='store_true',
                   help='Emit distinct material-keyed schema; never alias as a water package')
    p.add_argument('--transport-state-source', action='store_true',
                   help='Read complete V4 genealogy and write pinned sample/node state references')
    args = p.parse_args()
    if args.transport_state_source and any(args.output.with_suffix(s).exists()
            for s in ('.bin','.paths.bin','.metadata.json','.states.npz')):
        raise ValueError('V4 compilation requires new output paths')
    manifest = json.loads(args.manifest.read_text())
    material_identity = None
    material_name = 'Water_75eV'
    expected_density = 1.0
    component = 'ROI'
    if args.material_response:
        material_identity = manifest['material_identity']
        component = manifest['scored_component']
        material_name = material_identity['material_name']
        expected_density = material_identity['runtime_density_g_cm3']
        if not np.isfinite(expected_density) or expected_density<=0:
            raise ValueError('Invalid material density')
        if material_identity['kind']=='schneider':
            truth=schneider_identity(material_identity['schneider_file'],material_identity['hu'])
            if truth!=material_identity['schneider_identity']:
                raise ValueError('Schneider material identity changed')
            hu=truth['hu']
            expected_name='PatientTissueFromHU'+('Negative'+str(-hu) if hu<0 else str(hu))
            if material_name!=expected_name or expected_density!=float(format(truth['density_g_cm3'],'.6g')):
                raise ValueError('Schneider name/density mismatch')
        elif material_identity['kind']=='water_density':
            if material_name!='ResponseWater75' or expected_density not in (.25,.5,1.,1.5,2.):
                raise ValueError('Unplanned water density identity')
        else:
            raise ValueError('Unknown material response kind')
    pins = {str(args.manifest.resolve()): sha(args.manifest)}
    for path, pin in manifest['pins'].items():
        if sha(path) != pin:
            raise ValueError('Input pin mismatch: ' + path)
        pins[path] = pin
    if manifest['production_cut_mm'] != .05:
        raise ValueError('Wrong production cut')
    channel_count = args.energy_max//5
    totals = np.zeros((channel_count, 6))  # loss, local, birth, family dep, photon escape, families
    payload = [[] for _ in range(channel_count)]
    path_heads = [[] for _ in range(channel_count)]
    sample_states = [[] for _ in range(channel_count)]
    state_graphs, state_sources = [], []
    graphs, node_count = [], 0
    audits, actual_max = [], 0.
    for source_id, job in enumerate(manifest['jobs']):
        folder = Path(job['path'])
        if sha(folder/'topas.txt') != job['config_sha256'] or 'Finalization:' not in (folder/'topas.log').read_text():
            raise ValueError('Unfrozen/incomplete source')
        config = (folder/'topas.txt').read_text()
        if f's:Ge/{component}/Material = "{material_name}"' not in config:
            raise ValueError('Wrong material identity')
        if args.material_response and material_identity['kind']=='water_density':
            for required in ('s:Ma/ResponseWater75/BaseMaterial = "Water_75eV"',
                             'd:Ma/ResponseWater75/MeanExcitationEnergy = 75 eV',
                             f'd:Ma/ResponseWater75/Density = {expected_density:g} g/cm3'):
                if required not in config:raise ValueError('Wrong density-water composition')
        if args.transport_state_source:
            from electron_transport_state import load_transport_states
            raw = load_transport_states(folder/'steps.phsp')
            source_rows = np.lexsort((raw[:,5],raw[:,2],raw[:,1],raw[:,0])).astype('<u8')
        else:
            raw = load_v2_binary(folder/'steps.phsp')
            source_rows = None
        d, first, roots, families, dep, esc, audit = audit_records(
            raw, job['histories'],
            job.get('recording_bounds_mm', [[-80, 80], [-80, 80], [0, 350]]),
            expected_density_g_cm3=expected_density)
        state_links = {} if args.transport_state_source else None
        graph, pre_nodes, path_audit = build_path_graph(d, first, roots,
            source_rows=source_rows,state_links=state_links)
        if args.transport_state_source:
            state_links['source_id'] = np.full(len(graph),source_id,dtype='<u4')
            state_graphs.append(state_links)
            state_sources.append(dict(path=str((folder/'steps.phsp').resolve()),
                sha256=sha(folder/'steps.phsp'),rows=len(raw)))
        if node_count + len(graph) >= int(NONE):
            raise ValueError('Combined graph exceeds uint32 domain')
        linked = graph['previous'] != NONE
        graph['previous'][linked] += np.uint32(node_count)
        valid = pre_nodes != NONE
        pre_nodes[valid] += np.uint32(node_count)
        graphs.append(graph); node_count += len(graph)
        max_energy = float(d[d[:, 3] == 0, 17].max()/12)
        actual_max = max(actual_max, max_energy)
        if max_energy > args.energy_max+0.0001:
            raise ValueError('Pilot exceeds requested nominal source coverage')
        def bins(e):
            if np.any(e < 0) or np.any(e > max_energy):
                raise ValueError('Uncovered energy')
            # Explicit closed nominal-ceiling bin includes measured source
            # float rounding; metadata records the actual upper bound.
            return np.minimum((e/5).astype(int), channel_count-1)
        primary = d[d[:, 3] == 0]
        pb = bins(primary[:, 17]/12)
        np.add.at(totals[:, 0], pb, primary[:, 17]-primary[:, 18])
        np.add.at(totals[:, 1], pb, primary[:, 16])
        fb = bins(first[families, 23]/12)
        for col, val in [(2, first[families, 9]), (3, dep[families]), (4, esc[families]), (5, np.ones(len(families)))]:
            np.add.at(totals[:, col], fb, val)
        take = (roots >= 0) & (d[:, 16] > 0)
        rows, generating = d[take], first[roots[take]]
        eb = bins(generating[:, 23]/12)
        vectors = frame_segments(rows, generating)
        for b in range(channel_count):
            chosen = eb == b
            if chosen.any():
                payload[b].append(np.column_stack([rows[chosen, 16], vectors[chosen]]))
                path_heads[b].append(pre_nodes[take][chosen])
                if args.transport_state_source:
                    ids=source_rows[take][chosen]
                    sample_states[b].append(np.column_stack((np.full(len(ids),source_id,dtype='<u8'),ids)))
        audits.append(dict(energy_MeVu=job['energy_MeVu'], actual_max_MeVu=max_energy, **audit, **path_audit))
        for name in ['steps.phsp', 'steps.header', 'topas.txt', 'topas.log']:
            path = folder/name
            pins[str(path)] = sha(path)
        print('audited', job['energy_MeVu'], audit, flush=True)
    channels, samples, offset = [], [], 0
    for b, (loss, local, birth, deposit, escape, count) in enumerate(totals):
        if loss <= 0:
            raise ValueError('Missing primary energy exposure')
        residual = local + birth - loss
        if (b != 0 and abs(residual) > 1e-7*loss) or abs(residual) > 1e-3*loss:
            raise ValueError(f'Unexplained primary energy closure: bin={b}, loss={loss}, local={local}, birth={birth}, residual={residual}')
        rows = np.concatenate(payload[b]) if payload[b] else np.empty((0, 7))
        if abs(rows[:, 0].sum()-deposit) > 1e-9*loss:
            raise ValueError('Family/sample deposit mismatch')
        if len(rows):
            rows[:, 0] = np.cumsum(rows[:, 0])/deposit
            rows[-1, 0] = 1.
        channels.append(dict(energy_bin=b, energy_min_MeVu=5*b,
            energy_max_MeVu=actual_max if b == channel_count-1 else 5*(b+1),
            parent_loss_MeV=loss, raw_local_MeV=local, birth_MeV=birth,
            deposited_MeV=deposit, photon_escape_MeV=escape,
            terminal_local_excess_MeV=residual, deposited_fraction=deposit/loss,
            birth_fraction=birth/loss, family_count=int(count), offset=offset, count=len(rows)))
        samples.append(rows); offset += len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as f:
        f.write(struct.pack('<8sIII', b'MELSEG01' if args.material_response else b'WELSEG01',
                            1 if args.material_response or args.energy_max==300 else 2, channel_count, offset))
        for c in channels:
            f.write(struct.pack('<4dII', c['energy_min_MeVu'], c['energy_max_MeVu'],
                c['deposited_fraction'], c['photon_escape_MeV']/c['parent_loss_MeV'], c['offset'], c['count']))
        for rows in samples:
            f.write(rows.astype('<f8').tobytes())
    path_file = args.output.with_suffix('.paths.bin')
    with path_file.open('wb') as f:
        f.write(struct.pack('<8sIII', b'WELPTH01', 1, offset, node_count))
        for parts in path_heads:
            if parts:
                f.write(np.concatenate(parts).astype('<u4').tobytes())
        for graph in graphs:
            f.write(graph.tobytes())
    pins[str(Path(__file__).resolve())] = sha(__file__)
    state_file = None
    if args.transport_state_source:
        state_file = args.output.with_suffix('.states.npz')
        refs = np.concatenate([np.concatenate(parts) for parts in sample_states if parts])
        if len(refs)!=offset:
            raise ValueError('Sample/state count mismatch')
        arrays = {key:np.concatenate([g[key] for g in state_graphs]) for key in state_graphs[0]}
        with state_file.open('xb') as stream:
            np.savez(stream,sample_source_id=refs[:,0],sample_source_row=refs[:,1],**arrays)
        for helper in ('electron_transport_state.py','compare_electron_response_geometries.py'):
            helper_path=Path(__file__).with_name(helper).resolve()
            pins[str(helper_path)]=sha(helper_path)
    graph_source = Path(__file__).with_name('water_electron_path_graph.py')
    pins[str(graph_source.resolve())] = sha(graph_source)
    meta = dict(schema_version=2 if args.energy_max==300 else 3, status='UNVALIDATED_WATER_EM_DIAGNOSTIC',
        material='Water_75eV', density_g_cm3=1., production_cut_mm=.05,
        projectile=[6, 12], nominal_energy_max_MeVu=args.energy_max, actual_energy_max_MeVu=actual_max,
        data_filename=args.output.name, data_sha256=sha(args.output), data_size_bytes=args.output.stat().st_size,
        channels=channels, audits=audits, pins=pins,
        path_filename=path_file.name, path_sha256=sha(path_file), path_size_bytes=path_file.stat().st_size,
        path_nodes=node_count,
        sample_semantics='Deposit-weighted full 3D pre/post segment in parent frame; charged deposit uniform within own step, photon deposit at post-step interaction; one common random azimuth for entire ancestry.',
        limitations=['16 histories per source: pilot, not production data.',
            'Photon escape is explicit and must not be renormalized into samples or silently scored locally.',
            'Homogeneous water only; complete shared ancestry graph is not a CT material response.',
            'No GPU match established by compilation; exact energy loss is not adjusted by terminal raw excess.'])
    if args.material_response:
        meta.update(schema_version=4,status='UNVALIDATED_MATERIAL_EM_DIAGNOSTIC',
                    material=material_name,density_g_cm3=expected_density,material_identity=material_identity)
        meta['limitations']=['Homogeneous material response only; boundary continuation requires explicit transport handling.',
                            'Pilot statistics, not production validation; no dose fitting or coordinate rescaling.',
                            'Photon escape remains explicit and must not be silently deposited locally.']
    if state_file is not None:
        meta['transport_states']=dict(schema_version=1,filename=state_file.name,
            sha256=sha(state_file),size_bytes=state_file.stat().st_size,sources=state_sources,
            semantics='Exact raw V4 rows: sample deposition, node pre/post state, parent incoming edge; not a boundary sampling law.')
    args.output.with_suffix('.metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print('compiled', offset, 'segments', args.output, flush=True)


if __name__ == '__main__':
    main()

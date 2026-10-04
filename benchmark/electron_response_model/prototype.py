#!/usr/bin/env python3
"""Diagnostic conditional electron deposition model; NOT a CT boundary model.

All positive electron-family deposits contribute, including rare tails. Fit a
sparse joint histogram in asinh(r/scale), asinh(z/scale), conditional on root
electron birth energy. A draw selects a weighted cell, draws within its two
coordinates, and chooses a uniform azimuth. No raw trajectory is needed for
sampling. Primary-event holdout is separate from representation-error checks.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import struct
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIELDS = dict(run=(0, '<i4'), event=(4, '<i4'), track=(8, '<i4'), parent=(12, '<i4'),
              pdg=(16, '<i4'), step=(20, '<i4'), birth=(48, '<f8'),
              pre=(56, ('<f8', 3)), post=(80, ('<f8', 3)), dep=(104, '<f8'),
              energy=(112, '<f8'), remaining=(120, '<f8'), version=(144, '<i4'),
              direction=(232, ('<f8', 3)))
DTYPE = np.dtype(dict(names=list(FIELDS), formats=[v[1] for v in FIELDS.values()],
                     offsets=[v[0] for v in FIELDS.values()], itemsize=312))
SCALE = 0.01  # mm, coordinate transform only; not a fitted transport parameter
ENERGY_EDGES = np.geomspace(0.001, 1., 97)
TAILS = np.array([0.1, 0.5, 1., 2., 5.])


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def extract(folder, table=None, electron_source=None):
    if electron_source is None:
        meta_path = folder/'response.metadata.json'
        assert sha(meta_path) == table['metadata_sha256']
        meta = json.loads(meta_path.read_text())
        refs_meta_path = folder/'state_refs.metadata.json'
        assert sha(refs_meta_path) == table['state_reference_metadata_sha256']
        ref = json.loads(refs_meta_path.read_text())
        assert len(ref['sources']) == 1
        source = ref['sources'][0]
        projectile = meta['projectile']
        assert projectile in [[1, 1], [6, 12]]
        primary_pdg = 2212 if projectile == [1, 1] else 1000060120
        expected_histories = 16
    else:
        assert electron_source['accepted'] and electron_source['particle'] == 'e-'
        source = dict(electron_source, rows=Path(electron_source['path']).stat().st_size//312)
        meta = source
        table = dict(metadata_sha256=source['input_sha256'], material_identity=dict(section_id=source['section']))
        primary_pdg, expected_histories = 11, source['histories']
    path = Path(source['path'])
    if not path.is_absolute():
        path = folder/path
    assert path.stat().st_size == source['rows']*312 and sha(path) == source['sha256']
    raw = np.memmap(path, DTYPE, mode='r')
    assert np.all(raw['version'] == 4)
    order = np.lexsort((raw['step'], raw['track'], raw['event'], raw['run']))
    changed = np.ones(len(raw), bool)
    changed[1:] = ((raw['track'][order[1:]] != raw['track'][order[:-1]]) |
                  (raw['event'][order[1:]] != raw['event'][order[:-1]]) |
                  (raw['run'][order[1:]] != raw['run'][order[:-1]]))
    starts = np.flatnonzero(changed)
    first = order[starts]
    last = order[np.r_[starts[1:]-1, len(order)-1]]
    assert np.all(raw['step'][first] == 1)
    assert np.all((np.diff(raw['step'][order]) == 1) | changed[1:])
    groups = np.empty(len(raw), np.int32)
    groups[order] = np.cumsum(changed)-1
    lookup = {(int(raw['run'][r]), int(raw['event'][r]), int(raw['track'][r])): i for i, r in enumerate(first)}
    root = np.full(len(first), -2, np.int32)
    for i in range(len(first)):
        chain = []; cur = i
        while root[cur] == -2:
            if cur in chain:
                raise ValueError('Cyclic family')
            chain.append(cur); row = first[cur]
            if raw['parent'][row] == 0:
                assert raw['pdg'][row] == primary_pdg
                root[cur] = cur if primary_pdg == 11 else -1
                break
            parent = lookup[(int(raw['run'][row]), int(raw['event'][row]), int(raw['parent'][row]))]
            if raw['parent'][first[parent]] == 0 and primary_pdg != 11:
                assert raw['pdg'][row] == 11
                root[cur] = cur
                break
            cur = parent
        root[chain] = root[cur]
    roots = np.flatnonzero(root == np.arange(len(root)))
    row_root = root[groups]
    # Charged-only diagnostic: exclude first photon branches explicitly.
    # This is narrower than CT, which can replay recorded same-material
    # photon steps and reports missing continuation at interfaces/exhaustion.
    photon_branch = np.full(len(first), -1, np.int8)
    parent_track = np.full(len(first), -1, np.int32)
    for i, row in enumerate(first):
        if raw['parent'][row]:
            parent_track[i] = lookup[(int(raw['run'][row]), int(raw['event'][row]), int(raw['parent'][row]))]
    for i in range(len(first)):
        chain = []; cur = i
        while cur >= 0 and photon_branch[cur] == -1:
            chain.append(cur); cur = parent_track[cur]
        flag = bool(photon_branch[cur]) if cur >= 0 else False
        for item in reversed(chain):
            flag = flag or raw['pdg'][first[item]] == 22
            photon_branch[item] = flag
    selected = np.flatnonzero((row_root >= 0) & (raw['dep'] > 0) & (photon_branch[groups] == 0))
    rid = row_root[selected]; rfirst = first[rid]
    direction = raw['direction'][rfirst].copy()
    assert np.allclose(np.linalg.norm(direction, axis=1), 1., atol=1e-6, rtol=0)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    axis = np.zeros_like(direction)
    axis[np.arange(len(axis)), np.argmin(abs(direction), axis=1)] = 1
    x = np.cross(axis, direction); x /= np.linalg.norm(x, axis=1)[:, None]
    y = np.cross(direction, x)
    basis = np.stack([x, y, direction], axis=1)
    origin = raw['pre'][rfirst]
    pre = np.einsum('nij,nj->ni', basis, raw['pre'][selected]-origin)
    post = np.einsum('nij,nj->ni', basis, raw['post'][selected]-origin)
    assert np.all(raw['pdg'][selected] == 11)
    weight = raw['dep'][selected].copy()
    energy = raw['birth'][rfirst].copy()
    event = raw['event'][rfirst].copy()
    dep = np.bincount(rid, weights=weight, minlength=len(first))
    escape = np.zeros(len(first))
    tail_tracks = root >= 0
    assert not np.any(tail_tracks & (raw['pdg'][last] != 22) & (raw['remaining'][last] > 0))
    photon_roots = np.flatnonzero((root >= 0) & (raw['pdg'][first] == 22) &
                                 (parent_track >= 0) & (photon_branch[np.maximum(parent_track, 0)] == 0))
    np.add.at(escape, root[photon_roots], raw['birth'][first[photon_roots]])
    birth = raw['birth'][first[roots]]
    residual = dep[roots]+escape[roots]-birth
    assert np.all(abs(residual) <= 1e-3*birth)
    assert np.isfinite(pre).all() and np.isfinite(post).all() and np.all(weight > 0)
    primary_events = np.unique(raw['event'][first[raw['parent'][first] == 0]])
    assert len(primary_events) == expected_histories
    exposure = dict(electron_energy=raw['birth'][first[roots]].copy(),
                    deposited=dep[roots], escaped=escape[roots], event=raw['event'][first[roots]].copy())
    audit = dict(raw_rows=len(raw), depositing_segments=len(selected), electron_families=len(roots),
                 primary_histories=len(primary_events), max_family_relative_closure=float(np.max(abs(residual)/birth)),
                 deposited_MeV=float(weight.sum()), photon_branch_sink_MeV=float(escape.sum()),
                 photon_scope='Charged-only diagnostic: first photon branches and descendants excluded; narrower than recorded same-material CT photon replay',
                 density_g_cm3=meta['density_g_cm3'], section=table['material_identity']['section_id'],
                 raw_sha256=source['sha256'], metadata_sha256=table['metadata_sha256'])
    return pre, post, weight, energy, event, exposure, audit


def nodes(order):
    x, w = np.polynomial.legendre.leggauss(order)
    return (x+1)/2, w/2


def points(pre, post, weight, order):
    for t, w in zip(*nodes(order)):
        p = pre+t*(post-pre)
        yield np.hypot(p[:, 0], p[:, 1]), p[:, 2], weight*w


def primitive(t, r):
    """Integral of CDF of r*cos(phi); exact uniform azimuth."""
    q = np.clip(t/np.maximum(r, 1e-300), -1, 1)
    inner = t*(0.5+np.arcsin(q)/np.pi)+np.sqrt(np.maximum(0, r*r-t*t))/np.pi
    return np.where(t >= r, t, np.where(t <= -r, 0., inner))


def periodic_roi(r, centre, halfwidth=0.2, pitch=3.6, slit_width=0.5):
    # Infinite uniform slit comb, normalized per source slit: diagnostic only.
    # Integrate source x in [-width/2,width/2] analytically, no angular MC noise.
    out = np.zeros_like(r)
    kmax = int(math.ceil((float(r.max(initial=0))+slit_width/2+halfwidth+abs(centre))/pitch))+1
    for k in range(-kmax, kmax+1):
        a, b = k*pitch+centre-halfwidth, k*pitch+centre+halfwidth
        out += (primitive(b+slit_width/2, r)-primitive(b-slit_width/2, r)
                -primitive(a+slit_width/2, r)+primitive(a-slit_width/2, r))/slit_width
    return np.clip(out, 0, 1)


NAMES = ['r2', 'z', 'z2', 'r_z', 'peak', 'valley', *[f'tail_r_gt_{v:g}mm' for v in TAILS]]


def measures(r, z):
    return np.column_stack([r*r, z, z*z, r*z, periodic_roi(r, 0.), periodic_roi(r, 1.8),
                            *[(r > v).astype(float) for v in TAILS]])


def reference(pre, post, weight, order):
    result = np.zeros(len(NAMES))
    for r, z, w in points(pre, post, weight, order):
        result += w @ measures(r, z)
    return result/weight.sum()


def grid_edges(pre, post, step):
    rmax = max(np.linalg.norm(pre[:, :2], axis=1).max(), np.linalg.norm(post[:, :2], axis=1).max())
    zmin = min(pre[:, 2].min(), post[:, 2].min()); zmax = max(pre[:, 2].max(), post[:, 2].max())
    u = np.arange(math.ceil(np.arcsinh(rmax/SCALE)/step)+2)*step
    # Tail cutoffs are exact cell boundaries, not discarded/truncated tails.
    u = np.unique(np.r_[u, np.arcsinh(TAILS[TAILS < rmax]/SCALE)])
    v = np.arange(math.floor(np.arcsinh(zmin/SCALE)/step)-1,
                  math.ceil(np.arcsinh(zmax/SCALE)/step)+2)*step
    return u, v


def fit(pre, post, weight, step, order=8):
    u, v = grid_edges(pre, post, step)
    # Sparse bin IDs avoid a dense r-z grid even for long-range lung electrons.
    ids = []; masses = []
    for r, z, w in points(pre, post, weight, order):
        i = np.searchsorted(u, np.arcsinh(r/SCALE), side='right')-1
        j = np.searchsorted(v, np.arcsinh(z/SCALE), side='right')-1
        assert np.all((i >= 0) & (i < len(u)-1) & (j >= 0) & (j < len(v)-1))
        ids.append(i*(len(v)-1)+j); masses.append(w)
    unique, inverse = np.unique(np.concatenate(ids), return_inverse=True)
    mass = np.bincount(inverse, weights=np.concatenate(masses))
    assert abs(mass.sum()/weight.sum()-1) < 1e-12
    i, j = unique//(len(v)-1), unique%(len(v)-1)
    bounds = np.column_stack([u[i], u[i+1], v[j], v[j+1]]).astype('<f4')
    cdf = np.cumsum(mass, dtype=np.float64)/mass.sum(); cdf[-1] = 1.
    assert np.all(np.diff(np.r_[0., cdf]) > 0)
    return bounds, cdf


def model_metrics(bounds, cdf, order=6):
    b = bounds.astype(float); mass = np.diff(np.r_[0., cdf]); a, c = b[:, 2], b[:, 3]
    mean_z = SCALE*(np.cosh(c)-np.cosh(a))/(c-a)
    mean_z2 = SCALE**2*((np.sinh(2*c)-np.sinh(2*a))/4/(c-a)-0.5)
    result = np.zeros(len(NAMES))
    for t, w in zip(*nodes(order)):
        r = SCALE*np.sinh(b[:, 0]+t*(b[:, 1]-b[:, 0]))
        values = measures(r, mean_z)
        values[:, 2] = mean_z2
        result += (mass*w) @ values
    return result


def error(reference, candidate):
    return {name: dict(reference=float(a), model=float(b), difference=float(b-a),
                       relative_difference=float((b-a)/a) if abs(a) > 1e-12 else None)
            for name, a, b in zip(NAMES, reference, candidate)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tables', nargs='+', default=['s1_mid9', 's5_0', 's19_1'])
    p.add_argument('--projectile', choices=['proton', 'carbon'], default='carbon')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    index_path = ROOT/'data/ct_electron'/args.projectile/'index.json'
    index = json.loads(index_path.read_text())
    report = dict(status='HOMOGENEOUS_ELECTRON_MODEL_DIAGNOSTIC_ONLY', ct_acceptance=False,
                  script_sha256=sha(Path(__file__)), projectile=args.projectile,
                  index_sha256=sha(index_path), energy_interpolation_validated=False,
                  boundary_model_available=False, primary_energy_validation=False,
                  holdout='event modulo 4 == 0; independent primary histories, not random steps', tables=[])
    for name in args.tables:
        start = time.monotonic()
        table = next(t for t in index['tables'] if Path(t['path']).parent.name == name)
        folder = index_path.parent/Path(table['path']).parent
        pre, post, weight, energy, event, exposure, audit = extract(folder, table)
        save(args.out/(name+'_source.json'), audit)
        print('extracted', name, audit['depositing_segments'], flush=True)
        eb = np.searchsorted(ENERGY_EDGES, energy, side='right')-1
        assert np.all((eb >= 0) & (eb < len(ENERGY_EDGES)-1))
        root_bins = np.searchsorted(ENERGY_EDGES, exposure['electron_energy'], side='right')-1
        # Six fixed energy windows shared across materials; unsupported windows
        # remain explicitly empty. No material substitution or extrapolation.
        selected_bins = np.searchsorted(ENERGY_EDGES, [0.025, 0.05, 0.1, 0.2, 0.4, 0.65], side='right')-1
        channels = []; all_bounds = []; all_cdf = []; offset = 0
        for bin_id in selected_bins:
            train = (eb == bin_id) & (event % 4 != 0)
            test = (eb == bin_id) & (event % 4 == 0)
            if not train.any():
                channels.append(dict(energy_bin=int(bin_id), status='NO_TRAINING_COVERAGE')); continue
            a, b, w = pre[train], post[train], weight[train]
            ref = reference(a, b, w, 16); ref8 = reference(a, b, w, 8)
            candidates = []
            for step in [0.04, 0.02, 0.01, 0.005, 0.0025]:
                bounds, cdf = fit(a, b, w, step)
                m = model_metrics(bounds, cdf)
                errors = error(ref, m)
                guarded = ['r2', 'z2', 'peak', 'valley']
                passed = all(abs(errors[k]['relative_difference']) < 0.005 if errors[k]['relative_difference'] is not None
                             else abs(errors[k]['difference']) < 1e-12 for k in guarded)
                candidates.append(dict(asinh_grid_step=step, cells=len(cdf), representation_pass=passed, errors=errors))
                if passed:
                    break
            held = reference(pre[test], post[test], weight[test], 16) if test.any() else None
            event_metrics = []
            for ev in np.unique(event[test]):
                take = test & (event == ev)
                event_metrics.append(reference(pre[take], post[take], weight[take], 16))
            channel = dict(energy_bin=int(bin_id), energy_MeV=ENERGY_EDGES[bin_id:bin_id+2].tolist(),
                           training_segments=int(train.sum()), test_segments=int(test.sum()),
                           candidates=candidates, offset=offset, count=len(cdf),
                           quadrature_8_vs_16=error(ref, ref8),
                           heldout_errors=error(held, m) if held is not None else None,
                           heldout_primary_event_metrics=[dict(zip(NAMES, map(float, x))) for x in event_metrics],
                           min_sampled_E_MeV=float(energy[train].min()), max_sampled_E_MeV=float(energy[train].max()))
            channel['energy_accounting'] = {}
            for label, selector in [('train', exposure['event'] % 4 != 0), ('heldout', exposure['event'] % 4 == 0)]:
                take = (root_bins == bin_id) & selector
                incident = float(exposure['electron_energy'][take].sum())
                deposited = float(exposure['deposited'][take].sum())
                escaped = float(exposure['escaped'][take].sum())
                channel['energy_accounting'][label] = dict(families=int(take.sum()), incident_MeV=incident,
                    deposited_MeV=deposited, photon_branch_sink_MeV=escaped,
                    deposited_fraction=deposited/incident if incident else None,
                    photon_branch_sink_fraction=escaped/incident if incident else None)
            all_bounds.append(bounds); all_cdf.append(cdf); offset += len(cdf); channels.append(channel)
            print(name, int(bin_id), len(cdf), 'representation_pass', passed, flush=True)
        model = args.out/(name+'.npz')
        np.savez_compressed(model, bounds=np.concatenate(all_bounds), cdf=np.concatenate(all_cdf),
                            energy_edges=ENERGY_EDGES, coordinate_scale_mm=np.float64(SCALE))
        save(args.out/(name+'_channels.json'), channels)
        # Portable sampler payload: 4 float32 bounds + one float64 CDF per cell.
        packed = args.out/(name+'.bin')
        cells = np.empty(offset, dtype=[('cdf', '<f8'), ('bounds', '<f4', 4)])
        cells['cdf'] = np.concatenate(all_cdf); cells['bounds'] = np.concatenate(all_bounds)
        packed.write_bytes(cells.tobytes())
        result = dict(name=name, audit=audit, channels=channels, model_sha256=sha(packed),
                      runtime_payload_bytes=packed.stat().st_size, compressed_bytes=model.stat().st_size,
                      raw_file_bytes=(folder/'steps.phsp').stat().st_size, seconds=time.monotonic()-start)
        report['tables'].append(result); save(args.out/'report.json', report)
        print('finished', name, result['runtime_payload_bytes'], result['seconds'], flush=True)


if __name__ == '__main__':
    main()

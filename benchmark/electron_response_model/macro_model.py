#!/usr/bin/env python3
"""Compile audited sphere outcomes into compact, correlated macro responses.

Research only. No CT boundary fallback or continuous-energy interpolation is
implemented here. Every positive outcome contributes its energy; sparse cells
retain tails rather than discarding infrequent events. Compression replaces
each cell with its energy-weighted representative and is explicitly measured.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from prototype import FIELDS, nodes, periodic_roi, sha, save

EXTRA = dict(post_direction=(256, ('<f8', 3)), parent_step=(196, '<i4'),
             pre_material=(296, '<i4'), post_material=(300, '<i4'), status=(304, '<i4'), length=(280, '<f8'))
FIELDS = dict(FIELDS, **EXTRA)
DTYPE = np.dtype(dict(names=list(FIELDS), formats=[v[1] for v in FIELDS.values()],
                     offsets=[v[0] for v in FIELDS.values()], itemsize=312))


def extract(source):
    path = Path(source['path'])
    assert source['accepted'] and source['geometry'] in ['sphere_in_vacuum', 'voxel_box_in_vacuum']
    assert source['records_include_vacuum']
    assert sha(path) == source['sha256'] and path.stat().st_size % 312 == 0
    raw = np.memmap(path, DTYPE, mode='r')
    assert np.all(raw['version'] == 4) and np.all(np.isin(raw['pdg'], [11, 22]))
    order = np.lexsort((raw['step'], raw['track'], raw['event'], raw['run']))
    change = np.r_[True, (raw['track'][order[1:]] != raw['track'][order[:-1]]) |
                   (raw['event'][order[1:]] != raw['event'][order[:-1]]) |
                   (raw['run'][order[1:]] != raw['run'][order[:-1]])]
    starts = np.flatnonzero(change)
    first = order[starts]; last = order[np.r_[starts[1:]-1, len(order)-1]]
    group = np.empty(len(raw), np.int32); group[order] = np.cumsum(change)-1
    assert np.all(raw['step'][first] == 1)
    gaps = np.flatnonzero((np.diff(raw['step'][order]) != 1) & ~change[1:])
    if len(gaps):
        # Geant4 can perform an unscored zero-state navigation step at the
        # sphere/vacuum face. Accept only an exactly unchanged position,
        # direction and energy across exactly one omitted step. No missing
        # deposition, interior gap, or inferred continuation is permitted.
        assert source['geometry'] == 'sphere_in_vacuum'
        before, after = order[gaps], order[gaps+1]
        assert np.all(raw['step'][after]-raw['step'][before] == 2)
        assert np.all(raw['post'][before] == raw['pre'][after])
        assert np.all(raw['post_direction'][before] == raw['direction'][after])
        assert np.all(raw['remaining'][before] == raw['energy'][after])
        assert np.all(raw['post_material'][before] == raw['pre_material'][before])
        assert np.all(raw['post_material'][before] != raw['pre_material'][after])
        assert np.max(abs(np.linalg.norm(raw['post'][before], axis=1)-source['radius_mm'])) < 1e-9
        assert np.all(after == last[group[after]])
    lookup = {(int(raw['run'][r]), int(raw['event'][r]), int(raw['track'][r])): i for i, r in enumerate(first)}
    parent = np.full(len(first), -1, np.int32)
    for i, r in enumerate(first):
        if raw['parent'][r]:
            parent[i] = lookup[(int(raw['run'][r]), int(raw['event'][r]), int(raw['parent'][r]))]
    root = np.full(len(first), -1, np.int32)
    photon = np.zeros(len(first), bool)
    for i in range(len(first)):
        chain = []; j = i
        while root[j] < 0:
            if j in chain:
                raise ValueError('Cyclic genealogy')
            chain.append(j)
            if parent[j] < 0:
                assert raw['pdg'][first[j]] == 11
                root[j] = j
                break
            j = parent[j]
        for k in reversed(chain):
            if parent[k] >= 0:
                root[k] = root[parent[k]]
                photon[k] = photon[parent[k]] or raw['pdg'][first[k]] == 22
    primary = np.flatnonzero(parent < 0)
    assert len(primary) == source['histories']
    incident_energy = raw['birth'][first[primary]].copy()
    assert np.max(abs(incident_energy-incident_energy[0])) < 1e-12
    energy = float(incident_energy[0])
    assert abs(energy-source['energy_MeV']) < 1e-7*source['energy_MeV']
    # Validate generating-step association and physical energy closure before
    # any compression. Positions and energy must be continuous within tracks.
    children = np.flatnonzero(parent >= 0)
    child_first = first[children]
    offsets = starts[parent[children]] + raw['parent_step'][child_first]-1
    assert np.all(offsets >= starts[parent[children]])
    assert np.all(offsets < np.r_[starts[1:], len(order)][parent[children]])
    generating = order[offsets]
    assert np.all(raw['step'][generating] == raw['parent_step'][child_first])
    assert np.max(abs(raw['post'][generating]-raw['pre'][child_first]), initial=0) < 1e-6
    sums = np.bincount(generating, weights=raw['birth'][child_first], minlength=len(raw))
    residual = raw['energy']-raw['remaining']-raw['dep']-sums
    assert np.all(abs(residual) <= 1e-8+1e-10*raw['energy'])
    previous, following = order[:-1][~change[1:]], order[1:][~change[1:]]
    assert np.max(abs(raw['remaining'][previous]-raw['energy'][following]), initial=0) < 1e-8
    assert np.max(abs(raw['post'][previous]-raw['pre'][following]), initial=0) < 1e-6
    assert np.all(raw['dep'] >= 0) and np.all(raw['remaining'] >= 0)
    origin = raw['pre'][first[primary]]
    if source['geometry'] == 'sphere_in_vacuum':
        assert np.max(abs(origin), initial=0) < 1e-10
    else:
        assert np.max(abs(origin[:, 0]), initial=0) <= source['source_slit_width_mm']/2+1e-8
        assert np.max(abs(origin[:, 1]), initial=0) <= source['source_y_halfwidth_mm']+1e-8
        assert np.max(abs(origin[:, 2]-source['source_z_mm']), initial=0) < 1e-8
    axis = raw['direction'][first[primary]][0].copy()
    assert np.max(abs(raw['direction'][first[primary]]-axis), initial=0) < 1e-10
    assert abs(np.linalg.norm(axis)-1) < 1e-10
    helper = np.eye(3)[np.argmin(abs(axis))]
    x = np.cross(helper, axis); x /= np.linalg.norm(x)
    basis = np.array([x, np.cross(axis, x), axis])
    radius = source.get('radius_mm', 0)
    material = raw['pre_material'][first[primary]][0]
    assert np.all(raw['pre_material'][first[primary]] == material)
    inside = raw['pre_material'] == material
    step_limit = source.get('requested_max_step_mm', 0)
    max_step = float(raw['length'][inside & (raw['pdg'] == 11)].max(initial=0))
    if step_limit > 0:
        assert max_step <= step_limit*(1+1e-6)
    if source['geometry'] == 'sphere_in_vacuum':
        assert np.max(np.linalg.norm(raw['pre'][inside], axis=1)) <= radius+1e-6
        assert np.max(np.linalg.norm(raw['post'][inside], axis=1)) <= radius+1e-6
    else:
        for key in ['pre', 'post']:
            assert np.all(abs(raw[key][inside]) <= np.array(source['half_lengths_mm'])+1e-6)
    # TOPAS Vacuum has a tiny nonzero density. Only the pre-vacuum state is
    # included in the macro response; record and bound its later energy loss.
    vacuum_deposit = float(raw['dep'][~inside].sum())
    assert vacuum_deposit < 1e-12
    dep_rows = np.flatnonzero(inside & (~photon[group]) & (raw['dep'] > 0))
    outside_rows = order[~inside[order]]
    escape_tracks, first_outside = np.unique(group[outside_rows], return_index=True)
    escape_rows = outside_rows[first_outside][~photon[escape_tracks]]
    escape_tracks = escape_tracks[~photon[escape_tracks]]
    if source['geometry'] == 'voxel_box_in_vacuum':
        assert len(escape_tracks) == 0, 'Voxel reference is too small to stop charged family'
        radius = 1.  # Only empty exit arrays below; not a transport radius.
    # In vacuum no charged particle returns, loses energy or changes direction.
    assert np.all(raw['remaining'][last[escape_tracks]] == raw['energy'][escape_rows])
    assert np.all(~inside[last[escape_tracks]])
    assert np.all(raw['remaining'][last[inside[last]]] == 0)
    p = raw['pre'][escape_rows] @ basis.T
    d = raw['direction'][escape_rows] @ basis.T
    assert np.max(abs(np.linalg.norm(p, axis=1)-radius), initial=0) < 1e-6
    assert np.max(abs(np.linalg.norm(d, axis=1)-1), initial=0) < 1e-8
    surface_error = float(np.max(abs(np.linalg.norm(p, axis=1)-radius), initial=0))
    normal = p/np.linalg.norm(p, axis=1)[:, None]
    mu = np.clip(normal[:, 2], -1, 1)
    tangent = np.array([0., 0., 1.])[None, :]-mu[:, None]*normal
    norm = np.linalg.norm(tangent, axis=1)
    singular = norm < 1e-12
    tangent[~singular] /= norm[~singular, None]
    tangent[singular] = [1., 0., 0.]
    side = np.cross(normal, tangent)
    cosine = np.einsum('ij,ij->i', normal, d)
    assert np.min(cosine, initial=1) > -1e-7
    # Rotationally invariant joint coordinates: position polar angle, outward
    # direction polar angle, direction azimuth relative to source-axis plane,
    # and physical remaining energy fraction. Statistical W is not this energy.
    features = np.column_stack([np.arccos(mu), np.arccos(np.clip(cosine, 0, 1)),
                                np.arctan2(np.einsum('ij,ij->i', d, side),
                                           np.einsum('ij,ij->i', d, tangent)),
                                raw['energy'][escape_rows]/energy])
    photon_roots = np.flatnonzero((raw['pdg'][first] == 22) & (parent >= 0) &
                                 (~photon[np.maximum(parent, 0)]))
    event_to_index = {int(raw['event'][first[r]]): i for i, r in enumerate(primary)}
    assert len(event_to_index) == len(primary) and len(np.unique(raw['run'])) == 1
    events = raw['event'].copy()
    n = source['histories']
    # Events are not assumed sorted in the multithreaded output.
    assert set(event_to_index) == set(range(n))
    ledger = np.zeros((n, 3))
    np.add.at(ledger[:, 0], events[dep_rows], raw['dep'][dep_rows])
    np.add.at(ledger[:, 1], events[escape_rows], raw['energy'][escape_rows])
    np.add.at(ledger[:, 2], events[first[photon_roots]], raw['birth'][first[photon_roots]])
    family_residual = ledger.sum(axis=1)-energy
    assert np.max(abs(family_residual), initial=0) < 1e-8
    audit = dict(rows=len(raw), histories=n, radius_mm=radius, actual_energy_MeV=energy,
                 geometry=source['geometry'],
                 requested_energy_MeV=source['energy_MeV'],
                 certified_zero_state_navigation_gaps=int(len(gaps)),
                 requested_max_step_mm=step_limit, measured_max_charged_step_mm=max_step,
                 max_exit_surface_tolerance_mm=surface_error,
                 excluded_post_exit_vacuum_deposit_MeV=vacuum_deposit,
                 raw_sha256=source['sha256'],
                 max_step_residual_MeV=float(abs(residual).max(initial=0)),
                 max_history_residual_MeV=float(abs(family_residual).max(initial=0)),
                 energy_fractions=(ledger.sum(axis=0)/(n*energy)).tolist(),
                 channels=['charged_deposit', 'electron_exit', 'photon_birth_sink'],
                 photon_scope='First photon birth energy excluded with all descendants',
                 escaping_electrons=len(escape_rows), source_axis=axis.tolist())
    # Sphere fits use the incident-direction frame. A voxel reference needs
    # laboratory x for its explicitly uniform slit source and anisotropic grid.
    scoring_basis = basis if source['geometry'] == 'sphere_in_vacuum' else np.eye(3)
    return dict(pre=raw['pre'][dep_rows] @ scoring_basis.T, post=raw['post'][dep_rows] @ scoring_basis.T,
                weight=raw['dep'][dep_rows].copy(), event=events[dep_rows],
                exits=features, exit_weight=raw['energy'][escape_rows].copy(),
                exit_event=events[escape_rows], ledger=ledger, audit=audit)


def coarsen(features, weights, steps):
    """Retain total energy of every occupied joint cell, including rare tails."""
    if not len(weights):
        return np.empty((0, features.shape[1]), np.float32), np.empty(0, float)
    key = np.floor(features/np.asarray(steps)).astype(np.int32)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    representative = np.column_stack([np.bincount(inverse, weights=weights*v)/mass for v in features.T])
    assert np.all(mass > 0) and abs(mass.sum()/weights.sum()-1) < 1e-12
    return representative.astype(np.float32), mass


def deposition_points(data, order=8):
    coords, masses = [], []
    for t, w in zip(*nodes(order)):
        p = data['pre']+t*(data['post']-data['pre'])
        r = np.linalg.norm(p, axis=1)
        theta = np.arccos(np.clip(p[:, 2]/np.maximum(r, 1e-300), -1, 1))
        coords.append(np.column_stack([np.arcsinh(r/.01), theta]))
        masses.append(data['weight']*w)
    return np.concatenate(coords), np.concatenate(masses)


def deposit_metrics(coords):
    length = .01*np.sinh(coords[:, 0]); theta = coords[:, 1]
    r = length*np.sin(theta); z = length*np.cos(theta)
    return np.column_stack([r*r, z, z*z, periodic_roi(r, 0), periodic_roi(r, 1.8)])


def exit_metrics(coords):
    theta, alpha, beta, energy = coords.T
    mu = np.cos(theta)
    dz = np.cos(alpha)*mu+np.sin(alpha)*np.cos(beta)*np.sin(theta)
    return np.column_stack([energy, energy*energy, mu, mu*mu, dz, dz*dz,
                            energy*mu, energy*dz, np.cos(alpha)])


def metrics(data, energy, order=8):
    n = len(data['ledger']); total = n*energy
    d = np.zeros(5); blocks = np.zeros((20, 17))
    for t, w in zip(*nodes(order)):
        p = data['pre']+t*(data['post']-data['pre'])
        r = np.hypot(p[:, 0], p[:, 1]); z = p[:, 2]
        value = np.column_stack([r*r, z, z*z, periodic_roi(r, 0), periodic_roi(r, 1.8)])
        d += (data['weight']*w) @ value
        for k in range(5):
            blocks[:, k] += np.bincount(data['event'] % 20, weights=data['weight']*w*value[:, k], minlength=20)
    values = exit_metrics(data['exits'])
    e = data['exit_weight'] @ values
    for k in range(9):
        blocks[:, 5+k] = np.bincount(data['exit_event'] % 20, weights=data['exit_weight']*values[:, k], minlength=20)
    for k in range(3):
        blocks[:, 14+k] = np.bincount(np.arange(n) % 20, weights=data['ledger'][:, k], minlength=20)
    exposure = np.bincount(np.arange(n) % 20, minlength=20)*energy
    assert np.all(exposure > 0)
    blocks /= exposure[:, None]
    return np.r_[d, e, data['ledger'].sum(axis=0)]/total, blocks.std(axis=0, ddof=1)/np.sqrt(20)


NAMES = ['dep_r2', 'dep_z', 'dep_z2', 'dep_peak', 'dep_valley', 'exit_energy',
         'exit_energy2', 'exit_mu', 'exit_mu2', 'exit_dz', 'exit_dz2',
         'exit_energy_mu', 'exit_energy_dz', 'exit_cos_normal',
         'charged_deposit', 'electron_exit', 'photon_birth_sink']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    train, test = json.loads((args.input/'manifest.json').read_text())
    for key in ['energy_MeV', 'radius_mm', 'density_g_cm3', 'table', 'binary_sha256']:
        assert train[key] == test[key]
    assert train['seed'] != test['seed']
    data = extract(train); energy = data['audit']['actual_energy_MeV']
    baseline, train_se = metrics(data, energy)
    coarse, _ = metrics(data, energy, order=4)
    coords, weights = deposition_points(data)
    candidates = []
    chosen = None
    for factor in [1., .5, .25, .125]:
        # A fixed asinh bin can collapse the entire response of an eV/keV
        # electron into one atom. Resolve its actual radial extent instead.
        radial_step = min(.02, max(float(coords[:, 0].max()), 1e-12)/512)
        dc, dw = coarsen(coords, weights, np.array([radial_step, .02])*factor)
        ec, ew = coarsen(data['exits'], data['exit_weight'], np.array([.1, .1, .2, .02])*factor)
        total = train['histories']*energy
        result = np.r_[dw @ deposit_metrics(dc.astype(float)), ew @ exit_metrics(ec.astype(float)),
                       data['ledger'].sum(axis=0)]/total
        relative = np.divide(result-baseline, baseline, out=np.zeros_like(result), where=abs(baseline)>1e-12)
        passed = bool(np.max(abs(relative)) < .0025)
        candidate = dict(factor=factor, deposition_atoms=len(dw), exit_atoms=len(ew),
                         payload_bytes=len(dw)*16+len(ew)*24+24,
                         representation_relative_errors=dict(zip(NAMES, relative.tolist())),
                         representation_within_0p25pct=passed)
        candidates.append(candidate); print(json.dumps(candidate), flush=True)
        if passed:
            chosen = candidate
            break
    train_audit = data['audit']
    fractions = data['ledger'].sum(axis=0)/total
    np.savez_compressed(args.out/'model.npz', deposit=dc, deposit_cdf=np.cumsum(dw)/dw.sum(),
                        exit=ec, exit_cdf=np.cumsum(ew)/ew.sum() if len(ew) else np.empty(0), channel_fractions=fractions,
                        radius_mm=train['radius_mm'], energy_MeV=energy)
    del data, coords, weights
    data = extract(test)
    assert data['audit']['actual_energy_MeV'] == energy
    observed, test_se = metrics(data, energy)
    # Exploratory pointwise intervals; these are not simultaneous CT-dose or
    # repeated-macro-step bounds. No small photon/valley denominator is hidden.
    sem = np.hypot(train_se, test_se)
    checks = {}
    for i, name in enumerate(NAMES):
        denom = abs(observed[i])
        checks[name] = dict(reference=float(observed[i]), model=float(result[i]),
                            absolute_difference=float(result[i]-observed[i]),
                            relative_difference=float((result[i]-observed[i])/denom) if denom>1e-12 else None,
                            pointwise_95_relative_halfwidth=float(2.093*sem[i]/denom) if denom>1e-12 else None)
    report = dict(status='BOUNDED_MACRO_RESPONSE_DIAGNOSTIC', ct_acceptance=False,
                  composed_transport_validated=False, source_manifest_sha256=sha(args.input/'manifest.json'),
                  script_sha256=sha(Path(__file__)), train_audit=train_audit, test_audit=data['audit'],
                  candidates=candidates, selected=chosen, independent_checks=checks,
                  quadrature_4_vs_8_absolute_difference=(coarse-baseline).tolist(),
                  model_sha256=sha(args.out/'model.npz'), seconds=time.monotonic()-start,
                  note='A one-step response check is not a 1% full electron/CT-dose acceptance; validate repeated steps, interpolation, interfaces and peak/valley dose')
    save(args.out/'report.json', report)
    print('COMPLETE', report['seconds'], flush=True)


if __name__ == '__main__':
    main()

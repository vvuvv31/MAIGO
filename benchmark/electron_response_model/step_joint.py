#!/usr/bin/env python3
"""Research screen: energy-conditioned joint native electron steps <=0.25 mm.

This is NOT a production electron engine. Photon birth energy is an explicit
terminal channel, matching the charged-only diagnostic reference; its dose is
not deposited locally. Passing this screen cannot establish full CT accuracy.
No long step is cut or reinterpreted as a new scattering interaction.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from macro_model import DTYPE, extract
from macro_compose import basis
from prototype import NAMES, measures, periodic_roi, save, sha


def compile_model(source, bins, cap, seed, stratified=False, first_only=False,
                  primary_continuing=False, include_source_rows=False):
    # The existing extractor checks the entire graph, energy, continuity,
    # native maximum step, family ledger, and independent source hash.
    checked = extract(source)
    audit = checked['audit']
    del checked
    raw = np.memmap(source['path'], DTYPE, mode='r')
    order = np.lexsort((raw['step'], raw['track'], raw['event']))
    change = np.r_[True, (raw['event'][order[1:]] != raw['event'][order[:-1]]) |
                   (raw['track'][order[1:]] != raw['track'][order[:-1]])]
    starts = np.flatnonzero(change); first = order[starts]
    groups = np.empty(len(raw), np.int32); groups[order] = np.cumsum(change)-1
    lookup = {(int(raw['event'][i]), int(raw['track'][i])): g for g, i in enumerate(first)}
    parent = np.array([lookup[(int(raw['event'][i]), int(raw['parent'][i]))]
                       if raw['parent'][i] else -1 for i in first], np.int32)
    photon = np.full(len(first), -1, np.int8)
    for g in range(len(first)):
        chain = []; current = g
        while current >= 0 and photon[current] < 0:
            assert current not in chain
            chain.append(current); current = parent[current]
        flag = bool(photon[current]) if current >= 0 else False
        for item in reversed(chain):
            flag = flag or raw['pdg'][first[item]] == 22
            photon[item] = flag
    children = np.flatnonzero(parent >= 0); child_first = first[children]
    generating = order[starts[parent[children]]+raw['parent_step'][child_first]-1]
    assert np.all(raw['step'][generating] == raw['parent_step'][child_first])
    child_counts = np.bincount(generating, minlength=len(raw))
    nchild = int(child_counts.max())
    material = raw['pre_material'][first[parent < 0]][0]
    candidates = np.flatnonzero((photon[groups] == 0) & (raw['pre_material'] == material))
    if first_only:
        candidates = candidates[(raw['parent'][candidates] == 0) & (raw['step'][candidates] == 1)]
    if primary_continuing:
        assert not first_only
        candidates = candidates[(raw['parent'][candidates] == 0) & (raw['step'][candidates] > 1)]
    assert np.all(raw['pdg'][candidates] == 11)
    energy = raw['energy'][candidates].copy()
    # Log-energy cells are an explicit interpolation approximation. Terminal
    # sub-cutoff steps, every child birth and every rare photon birth survive
    # compaction; only ordinary no-child continuing steps use a reservoir.
    emin = float(energy.min()); emax = float(energy.max())
    edges = np.geomspace(emin, emax, bins+1)
    cell = np.minimum(np.searchsorted(edges, energy, side='right')-1, bins-1)
    assert cell.min() >= 0
    rng = np.random.default_rng(seed); selected = []; weights = []; counts = []
    idx = np.argsort(cell, kind='stable'); offsets = np.searchsorted(cell[idx], np.arange(bins+1))
    for b in range(bins):
        ids = candidates[idx[offsets[b]:offsets[b+1]]]
        protected = (child_counts[ids] > 0) | (raw['remaining'][ids] == 0)
        ordinary = ids[~protected]
        if stratified and cap and len(ordinary) > cap:
            cosine = np.einsum('ij,ij->i', raw['direction'][ordinary], raw['post_direction'][ordinary])
            loss = (raw['energy'][ordinary]-raw['remaining'][ordinary])/raw['energy'][ordinary]
            # Preserve the highest-angle 1% exactly, including arbitrarily
            # rare events. Stratify the rest jointly in scattering and loss;
            # keep real correlated outcomes, not independent fitted marginals.
            extreme = cosine <= np.quantile(cosine, .01)
            if np.all(extreme):
                extreme[:] = False  # Degenerate angular distribution, no angular tail.
            cos_edges = np.quantile(cosine[~extreme], np.linspace(0, 1, 17))
            loss_edges = np.quantile(loss[~extreme], np.linspace(0, 1, 5))
            cells = np.minimum(np.searchsorted(cos_edges, cosine, side='right')-1, 15)*4
            cells += np.minimum(np.searchsorted(loss_edges, loss, side='right')-1, 3)
            picked = [ordinary[extreme]]; weighted = [np.ones(extreme.sum())]
            for c in np.unique(cells[~extreme]):
                members = ordinary[(cells == c) & ~extreme]
                draw = rng.choice(members, min(max(1, cap//64), len(members)), replace=False)
                picked.append(draw); weighted.append(np.full(len(draw), len(members)/len(draw)))
            sample = np.concatenate(picked); sample_mass = np.concatenate(weighted)
        else:
            sample = rng.choice(ordinary, min(cap, len(ordinary)), replace=False) if cap else ordinary
            sample_mass = np.full(len(sample), len(ordinary)/max(1, len(sample)))
        keep = np.r_[ids[protected], sample]
        mass = np.r_[np.ones(protected.sum()), sample_mass]
        selected.append(keep); weights.append(mass); counts.append(len(keep))
    selected = np.concatenate(selected); mass = np.concatenate(weights)
    offsets = np.r_[0, np.cumsum(counts)]
    cdf = np.empty(len(mass))
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        if hi > lo:
            cdf[lo:hi] = np.cumsum(mass[lo:hi])/mass[lo:hi].sum()
            cdf[hi-1] = 1
    nonempty = np.flatnonzero(np.diff(offsets))
    nearest = nonempty[np.argmin(abs(np.arange(bins)[:, None]-nonempty), axis=1)]
    # Remap all selected generating rows and retain complete sibling lists.
    row_map = np.full(len(raw), -1, np.int32); row_map[selected] = np.arange(len(selected))
    keep = row_map[generating] >= 0
    child_rows = child_first[keep]; child_parents = row_map[generating[keep]]
    permutation = np.argsort(child_parents, kind='stable')
    child_rows = child_rows[permutation]; child_parents = child_parents[permutation]
    sibling_starts = np.flatnonzero(np.r_[True, np.diff(child_parents) != 0])
    child_slot = np.arange(len(child_rows))-np.repeat(sibling_starts, np.diff(np.r_[sibling_starts, len(child_rows)]))
    direction = raw['direction'][selected].copy()
    u, v = basis(direction); frame = np.stack([u, v, direction], axis=1)
    delta = np.einsum('nij,nj->ni', frame, raw['post'][selected]-raw['pre'][selected])
    postdir = np.einsum('nij,nj->ni', frame, raw['post_direction'][selected])
    fraction = np.zeros((len(selected), 2+nchild))
    fraction[:, 0] = raw['dep'][selected]/raw['energy'][selected]
    fraction[:, 1] = raw['remaining'][selected]/raw['energy'][selected]
    fraction[child_parents, child_slot+2] = raw['birth'][child_rows]/raw['energy'][selected[child_parents]]
    childdir = np.zeros((len(selected), nchild, 3))
    childdir[child_parents, child_slot] = np.einsum('nij,nj->ni', frame[child_parents], raw['direction'][child_rows])
    childpdg = np.zeros((len(selected), nchild), np.int32)
    childpdg[child_parents, child_slot] = raw['pdg'][child_rows]
    closure = float(np.max(abs(fraction.sum(axis=1)-1)))
    assert closure < 1e-7
    length = raw['length'][selected].copy()
    assert length.max() <= .25*(1+1e-6)
    # Correct only roundoff at the categorical CDF end; never dose-normalize.
    channel_cdf = np.cumsum(fraction, axis=1); channel_cdf[:, -1] = 1
    model = dict(edges=edges, offsets=offsets, nearest=nearest, cdf=cdf,
                 delta=delta, postdir=postdir, fraction=fraction, channel_cdf=channel_cdf,
                 childdir=childdir, childpdg=childpdg, length=length)
    if include_source_rows:
        # Used only by the full-family compiler to attach exact descendant
        # graphs. These indices never refer to a different native extraction.
        model['source_row'] = selected
    audit.update(model_source_steps=len(candidates), retained_steps=len(selected), bins=bins,
                 ordinary_step_cap=cap, reservoir_seed=seed, maximum_fraction_closure_error=closure,
                 empirical_payload_bytes=sum(a.nbytes for a in model.values()),
                 empty_energy_cells=int(np.count_nonzero(np.diff(offsets) == 0)),
                 max_children=nchild, source_sha256=source['sha256'], stratified=stratified, first_only=first_only,
                 primary_continuing=primary_continuing,
                 native_settings={k:source[k] for k in ['table', 'density_g_cm3', 'production_cut_mm', 'requested_max_step_mm']})
    return model, audit


def rotate(local, direction, phi):
    u, v = basis(direction)
    a = np.cos(phi)[:, None]*u+np.sin(phi)[:, None]*v
    b = -np.sin(phi)[:, None]*u+np.cos(phi)[:, None]*v
    return local[:, 0, None]*a+local[:, 1, None]*b+local[:, 2, None]*direction


def transport(continuation_model, incident, histories, rng, entry_model=None):
    position = np.zeros((histories, 3)); direction = np.zeros((histories, 3)); direction[:, 2] = 1
    kinetic = np.full(histories, incident); status = np.full(histories, -1, np.int8)
    active = np.arange(histories); advances = 0; low_endpoint = 0
    maximum_path = 0.; maximum_chord = 0.
    for iteration in range(2000):
        if not len(active):
            break
        model = entry_model if entry_model is not None and iteration == 0 else continuation_model
        e = kinetic[active]
        assert np.all(e > 0) and e.max() <= model['edges'][-1]*(1+1e-12)
        cell = np.clip(np.searchsorted(model['edges'], e, side='right')-1, 0, len(model['nearest'])-1)
        # Below the smallest measured incident energy, only verified terminal
        # native records can be used. This endpoint approximation is counted.
        low_endpoint += int(np.count_nonzero(e < model['edges'][0]))
        cell = model['nearest'][cell]
        rows = np.empty(len(active), np.int64)
        for b in np.unique(cell):
            mask = cell == b; lo, hi = model['offsets'][b:b+2]
            rows[mask] = lo+np.searchsorted(model['cdf'][lo:hi], rng.random(mask.sum()), side='right')
        below = e < model['edges'][0]
        assert not np.any(below & (model['fraction'][rows, 0] < 1-1e-7))
        coin = rng.random(len(active)); choice = (coin[:, None] >= model['channel_cdf'][rows]).sum(axis=1)
        phi = rng.random(len(active))*2*np.pi
        delta = rotate(model['delta'][rows], direction[active], phi)
        maximum_path = max(maximum_path, float(model['length'][rows].max()))
        maximum_chord = max(maximum_chord, float(np.linalg.norm(delta, axis=1).max()))
        deposited = choice == 0; parent = choice == 1; child = choice >= 2
        ids = active[deposited]
        position[ids] += rng.random(len(ids))[:, None]*delta[deposited]
        status[ids] = 0
        ids = active[parent]
        position[ids] += delta[parent]
        direction[ids] = rotate(model['postdir'][rows[parent]], direction[ids], phi[parent])
        kinetic[ids] *= model['fraction'][rows[parent], 1]
        ids = active[child]; r = rows[child]; c = choice[child]-2
        position[ids] += delta[child]
        direction[ids] = rotate(model['childdir'][r, c], direction[ids], phi[child])
        kinetic[ids] *= model['fraction'][r, c+2]
        pdg = model['childpdg'][r, c]
        assert np.all(np.isin(pdg, [11, 22]))
        status[ids[pdg == 22]] = 2
        advances += len(active); active = active[status[active] < 0]
        if len(active):
            assert np.max(abs(np.linalg.norm(direction[active], axis=1)-1)) < 1e-7
    assert len(active) == 0, 'Unfinished energy packets'
    assert np.all(np.isin(status, [0, 2]))
    return position, status, dict(advances=advances, iterations=iteration,
            below_minimum_terminal_steps=low_endpoint, max_path_mm=maximum_path, max_chord_mm=maximum_chord)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--histories', type=int, default=1000000)
    p.add_argument('--bins', type=int, default=1024)
    p.add_argument('--cap', type=int, default=256, help='0 keeps every native step')
    p.add_argument('--stratified', action='store_true')
    p.add_argument('--model-from', type=Path, help='Validate a frozen model at an independent source energy; never refit it')
    p.add_argument('--native-entry', action='store_true', help='Diagnostic: native evaluation TRAIN first steps, frozen continuation model')
    p.add_argument('--seed', type=int, default=100303001)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    assert a.histories >= 200 and a.histories % 20 == 0 and a.bins > 0 and a.cap >= 0
    sources = json.loads((a.source/'manifest.json').read_text())
    assert sources[0]['seed'] != sources[1]['seed']
    reference = json.loads((a.reference/'report.json').read_text())
    assert reference['train_audit']['raw_sha256'] == sources[0]['sha256']
    assert reference['test_audit']['raw_sha256'] == sources[1]['sha256']
    assert 0 < sources[0]['requested_max_step_mm'] <= .25
    start = time.monotonic()
    if a.model_from:
        previous = json.loads((a.model_from/'report.json').read_text())
        assert sha(a.model_from/'model.npz') == previous['model_sha256']
        audit = previous['source']
        assert audit['native_settings'] == {k:sources[0][k] for k in audit['native_settings']}
        assert sources[1]['sha256'] != audit['source_sha256']
        with np.load(a.model_from/'model.npz') as saved:
            model = {k:saved[k].copy() for k in saved.files}
    else:
        model, audit = compile_model(sources[0], a.bins, a.cap, a.seed, a.stratified)
    save(a.out/'model_audit.json', audit)
    np.savez(a.out/'model.npz', **model)
    entry_model = None; entry_audit = None
    if a.native_entry:
        entry_model, entry_audit = compile_model(sources[0], 1, a.cap, a.seed, True, True)
        assert entry_audit['actual_energy_MeV'] == reference['test_audit']['actual_energy_MeV']
        np.savez(a.out/'entry_model.npz', **entry_model)
        save(a.out/'entry_model_audit.json', entry_audit)
    print('compiled', audit, flush=True)
    rng = np.random.default_rng(a.seed+1); metrics = []; profiles = []; work = []; counts = np.zeros(3, np.int64)
    centres = np.arange(-1.75, 1.8, .1)
    for block in range(20):
        pos, status, stats = transport(model, reference['test_audit']['actual_energy_MeV'], a.histories//20, rng, entry_model)
        points = pos[status == 0]; r = np.hypot(points[:, 0], points[:, 1]); z = points[:, 2]
        metrics.append(measures(r, z).sum(axis=0)/(a.histories//20))
        profiles.append(np.array([periodic_roi(r, float(x), halfwidth=.05).sum() for x in centres])/(a.histories//20))
        work.append(stats); counts += np.bincount(status, minlength=3)
        save(a.out/'progress.json', dict(blocks=block+1, seconds=time.monotonic()-start))
        print('block', block+1, time.monotonic()-start, flush=True)
    values = np.array(metrics); mean = values.mean(axis=0); sem = values.std(axis=0, ddof=1)/np.sqrt(20)
    checks = {}
    for i, name in enumerate(NAMES):
        old = reference['independent_absolute_errors'][name]; ref = old['reference']
        checks[name] = dict(reference=ref, model=float(mean[i]), relative_difference=float((mean[i]-ref)/ref) if abs(ref)>1e-12 else None,
            conditional_relative_halfwidth=float(2.44*np.hypot(sem[i], old['test_block_standard_error'])/abs(ref)) if abs(ref)>1e-12 else None)
    ref = np.load(a.reference/'local_profile.npz'); profiles = np.array(profiles)
    diff = profiles.mean(axis=0)/ref['reference']-1
    report = dict(status='CHARGED_ONLY_SINGLE_STEP_SCREEN', ct_acceptance=False, production_promoted=False,
        scope='Homogeneous lung; first photon branches are explicit energy sinks, excluding their descendants; not full CT replay scope',
        evaluation_energy_MeV=reference['test_audit']['actual_energy_MeV'],
        frozen_model_from=str(a.model_from) if a.model_from else None,
        native_entry=entry_audit,
        native_entry_model_sha256=sha(a.out/'entry_model.npz') if entry_model is not None else None,
        approximation='Joint empirical native steps conditioned on log energy, uniformly rotated; remaining energies scaled by incoming energy. No hidden MSC history or voxel geometry state.',
        source=audit, script_sha256=sha(Path(__file__)), model_sha256=sha(a.out/'model.npz'),
        reference_report_sha256=sha(a.reference/'report.json'), seconds=time.monotonic()-start,
        histories=a.histories, seed=a.seed+1, counts=dict(deposit=int(counts[0]), photon_sink=int(counts[2])),
        packet_energy_closure=bool(counts[0]+counts[2] == a.histories), metrics=checks,
        max_local_0p1mm_relative_difference=float(abs(diff).max()),
        peak_valley_points_within_1pct=all(abs(checks[k]['relative_difference']) < .01 for k in ['peak', 'valley']),
        work=work, uncertainty_note='Conditional packet/reference block error excludes training-model uncertainty; no GPU throughput claim')
    save(a.out/'report.json', report)
    np.savez(a.out/'profile.npz', centres=centres, model=profiles.mean(axis=0), reference=ref['reference'], relative_difference=diff)
    print('COMPLETE', json.dumps({k: report[k] for k in ['seconds', 'metrics', 'max_local_0p1mm_relative_difference']}), flush=True)


if __name__ == '__main__':
    main()

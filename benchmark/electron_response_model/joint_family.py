#!/usr/bin/env python3
"""Research operator: sampled primary short steps, exact native child families.

This is not a production CT engine. All descendants, including photons, retain
their joint native trajectory and birth state. A finite native-source exit is
coverage missing, never patient escape or local dose. Only primary continuing
steps are resampled; primary entry has its own table. The runtime currently
supports the explicit homogeneous diagnostic box, not material interfaces.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from macro_model import DTYPE
from macro_compose import basis
from prototype import save, sha
from step_joint import compile_model
from voxel_profile import chord_profile, PEAK, VALLEY

DEPOSIT, ESCAPE, PHOTON_DEPOSIT, COVERAGE = range(4)


def genealogy(raw):
    """Resolve actual step keys, including certified native navigation gaps."""
    order = np.lexsort((raw['step'], raw['track'], raw['event'], raw['run']))
    changed = np.r_[True, (raw['track'][order[1:]] != raw['track'][order[:-1]]) |
                    (raw['event'][order[1:]] != raw['event'][order[:-1]]) |
                    (raw['run'][order[1:]] != raw['run'][order[:-1]])]
    starts = np.flatnonzero(changed)
    first = order[starts]
    lookup = {(int(raw['run'][r]), int(raw['event'][r]), int(raw['track'][r])): i
              for i, r in enumerate(first)}
    children = first[raw['parent'][first] != 0]
    ends = np.r_[starts[1:], len(order)]
    generating = np.empty(len(children), np.int64)
    # Secondary counts are small; avoid an all-step Python dictionary.
    for i, r in enumerate(children):
        g = lookup[(int(raw['run'][r]), int(raw['event'][r]), int(raw['parent'][r]))]
        rows = order[starts[g]:ends[g]]
        j = np.searchsorted(raw['step'][rows], raw['parent_step'][r])
        assert j < len(rows) and raw['step'][rows[j]] == raw['parent_step'][r]
        generating[i] = rows[j]
    next_row = np.full(len(raw), -1, np.int32)
    next_row[order[:-1][~changed[1:]]] = order[1:][~changed[1:]]
    return first, children, generating, next_row


def compile_family(source, out, seed):
    out.mkdir(parents=True, exist_ok=False)
    # The standard compiler audits source SHA, genealogy, energy, continuity,
    # finite-domain exits and native true path lengths before selecting rows.
    continuation, ca = compile_model(source, 1024, 256, seed, True,
                                    primary_continuing=True, include_source_rows=True)
    entry, ea = compile_model(source, 1, 256, seed+1, True, True,
                             include_source_rows=True)
    raw = np.memmap(source['path'], DTYPE, mode='r')
    first, children, generating, next_row = genealogy(raw)
    primary = first[raw['parent'][first] == 0]
    material = raw['pre_material'][primary[0]]
    selected = np.flatnonzero(raw['parent'] != 0)
    remap = np.full(len(raw), -1, np.int32)
    remap[selected] = np.arange(len(selected))
    counts = np.bincount(generating, minlength=len(raw))
    nchild = int(counts.max(initial=0))
    # Use precisely the same sibling ordering as compile_model (track order).
    permutation = np.argsort(generating, kind='stable')
    ordered_parent = generating[permutation]
    groups = np.flatnonzero(np.r_[True, np.diff(ordered_parent) != 0])
    slots = np.arange(len(children))-np.repeat(groups, np.diff(np.r_[groups, len(children)]))
    child_rows = np.full((len(raw), nchild), -1, np.int32)
    child_rows[ordered_parent, slots] = children[permutation]
    graph_children = child_rows[selected]
    child_energy = np.where(graph_children >= 0, raw['birth'][np.maximum(graph_children, 0)], 0.)
    fraction = np.column_stack([raw['dep'][selected], raw['remaining'][selected], child_energy]) / raw['energy'][selected, None]
    assert np.all(np.isfinite(fraction)) and np.all(fraction >= 0)
    closure = float(abs(fraction.sum(axis=1)-1).max(initial=0))
    assert closure < 1e-7
    graph = dict(delta=(raw['post'][selected]-raw['pre'][selected]).copy(),
                 length=raw['length'][selected].copy(), energy=raw['energy'][selected].copy(),
                 pdg=raw['pdg'][selected].copy(), inside=(raw['pre_material'][selected] == material),
                 fraction=fraction, child=np.where(graph_children >= 0, remap[np.maximum(graph_children, 0)], -1),
                 next=np.where(next_row[selected] >= 0, remap[np.maximum(next_row[selected], 0)], -1))
    assert np.all((graph_children < 0) | (graph['child'] >= 0))
    assert np.all((next_row[selected] < 0) | (graph['next'] >= 0))
    max_path = float(graph['length'][graph['inside'] & (graph['pdg'] == 11)].max(initial=0))
    assert max_path <= source['requested_max_step_mm']*(1+1e-6) <= .250001
    for model in [continuation, entry]:
        rows = model.pop('source_row')
        native_child = child_rows[rows]
        model['child'] = np.where(native_child >= 0, remap[np.maximum(native_child, 0)], -1)
        model['source_direction'] = raw['direction'][rows].copy()
        model['source_energy'] = raw['energy'][rows].copy()
        assert np.all((model['fraction'][:, 2:] == 0) | (model['child'] >= 0))
    paths = dict(continuation=out/'continuation.npz', entry=out/'entry.npz', graph=out/'graph.npz')
    for name, value in [('continuation', continuation), ('entry', entry), ('graph', graph)]:
        np.savez(paths[name], **value)
    report = dict(status='COMPILED_NOT_VALIDATED', ct_acceptance=False, production_promoted=False,
                  scope='One material, monoenergetic primary entry; exact native descendants with explicit source coverage',
                  source=source, continuation_audit=ca, entry_audit=ea,
                  graph_rows=len(selected), photon_rows=int(np.count_nonzero(graph['pdg'] == 22)),
                  maximum_graph_electron_path_mm=max_path, maximum_graph_fraction_closure=closure,
                  payload_bytes=sum(a.nbytes for m in [continuation, entry, graph] for a in m.values()),
                  hashes={k:sha(v) for k,v in paths.items()}, script_sha256=sha(Path(__file__)))
    save(out/'report.json', report)
    return report


def load_model(folder):
    report = json.loads((folder/'report.json').read_text())
    result = {}
    for name in ['continuation', 'entry', 'graph']:
        path = folder/(name+'.npz')
        assert sha(path) == report['hashes'][name]
        with np.load(path) as f:
            result[name] = {k:f[k].copy() for k in f.files}
    return result, report


def frame(direction, phi):
    u, v = basis(direction)
    a = np.cos(phi)[:, None]*u+np.sin(phi)[:, None]*v
    b = -np.sin(phi)[:, None]*u+np.cos(phi)[:, None]*v
    return np.stack([a, b, direction], axis=2)


def draw_rows(model, energy, rng):
    assert np.all(np.isfinite(energy)) and np.all(energy > 0)
    assert energy.max() <= model['edges'][-1]*(1+1e-7)
    cell = np.clip(np.searchsorted(model['edges'], energy, side='right')-1, 0, len(model['nearest'])-1)
    cell = model['nearest'][cell]
    rows = np.empty(len(energy), np.int64)
    for b in np.unique(cell):
        mask = cell == b
        lo, hi = model['offsets'][b:b+2]
        rows[mask] = lo+np.searchsorted(model['cdf'][lo:hi], rng.random(mask.sum()), side='right')
    below = energy < model['edges'][0]
    assert not np.any(below & (model['fraction'][rows, 0] < 1-1e-7))
    return rows, int(below.sum())


def box_fraction(position, delta, half):
    """First straight-chord intersection; never invents a scattering step."""
    assert np.all(abs(position) <= half+1e-9)
    distance = np.full_like(delta, np.inf)
    np.divide(np.where(delta > 0, half-position, -half-position), delta,
              out=distance, where=delta != 0)
    return np.clip(distance.min(axis=1), 0., 1.)


def transport(models, incident, histories, rng, half=20.):
    graph = models['graph']
    pos = np.column_stack([rng.uniform(-.25, .25, histories), rng.uniform(-1., 1., histories),
                           np.full(histories, .25)])
    source_x = pos[:, 0].copy()
    direction = np.zeros((histories, 3)); direction[:, 2] = 1.
    kinetic = np.full(histories, incident)
    cursor = np.full(histories, -1, np.int64)
    rotation = np.broadcast_to(np.eye(3), (histories, 3, 3)).copy()
    photon = np.zeros(histories, bool)
    status = np.full(histories, -1, np.int8)
    advances = graph_advances = below_count = 0
    max_charged_path = 0.
    for iteration in range(4000):
        active = np.flatnonzero(status < 0)
        if not len(active):
            break
        graph_ids = active[cursor[active] >= 0]
        sampled_ids = active[cursor[active] < 0]
        # Process the two queues from the iteration snapshot; a child created
        # by sampled transport must not get an extra transaction this round.
        for ids, exact in [(sampled_ids, False), (graph_ids, True)]:
            if not len(ids):
                continue
            if exact:
                rows = cursor[ids]
                outside = ~graph['inside'][rows]
                status[ids[outside]] = COVERAGE
                ids, rows = ids[~outside], rows[~outside]
                if not len(ids):
                    continue
                operator = graph
                target = rotation[ids]
                delta = np.einsum('nij,nj->ni', target, graph['delta'][rows])
                pdg = graph['pdg'][rows]
                graph_advances += len(ids)
            else:
                operator = models['entry'] if iteration == 0 else models['continuation']
                rows, below = draw_rows(operator, kinetic[ids], rng)
                below_count += below
                target = frame(direction[ids], rng.random(len(ids))*2*np.pi)
                delta = np.einsum('nij,nj->ni', target, operator['delta'][rows])
                pdg = np.full(len(ids), 11)
            max_charged_path = max(max_charged_path, float(operator['length'][rows[pdg == 11]].max(initial=0)))
            frac = operator['fraction'][rows]
            boundary_fraction = box_fraction(pos[ids], delta, half)
            boundary = boundary_fraction < 1.
            coin = rng.random(len(ids))
            dep_probability = frac[:, 0]*np.where(pdg == 11, boundary_fraction, ~boundary)
            deposit = coin < dep_probability
            terminal = deposit | boundary
            depos = ids[deposit]
            along = np.where(pdg[deposit] == 11, rng.random(len(depos))*boundary_fraction[deposit], 1.)
            pos[depos] += along[:, None]*delta[deposit]
            status[depos] = np.where(photon[depos], PHOTON_DEPOSIT, DEPOSIT)
            escaped = boundary & ~deposit
            pos[ids[escaped]] += boundary_fraction[escaped, None]*delta[escaped]
            status[ids[escaped]] = ESCAPE
            surviving = ~terminal
            sid, rr = ids[surviving], rows[surviving]
            ff = frac[surviving]
            cdf = np.cumsum(ff, axis=1); cdf[:, -1] = 1.
            choice = (coin[surviving, None] >= cdf).sum(axis=1)
            assert np.all(choice >= 1)
            pos[sid] += delta[surviving]
            kinetic[sid] *= ff[np.arange(len(sid)), choice]
            parent = choice == 1
            if exact:
                cursor[sid[parent]] = graph['next'][rr[parent]]
                restart = sid[parent][cursor[sid[parent]] == -2]
                if len(restart):
                    rrestart = rr[parent][cursor[sid[parent]] == -2]
                    direction[restart] = np.einsum('nij,nj->ni', rotation[restart], graph['postdir'][rrestart])
                missing = sid[parent][cursor[sid[parent]] == -1]
                status[missing] = COVERAGE
            else:
                direction[sid[parent]] = np.einsum('nij,nj->ni', target[surviving][parent], operator['postdir'][rr[parent]])
            child_ids, child_rows, slot = sid[~parent], rr[~parent], choice[~parent]-2
            child_cursor = operator['child'][child_rows, slot]
            assert np.all(child_cursor >= 0)
            cursor[child_ids] = child_cursor
            photon[child_ids] |= graph['pdg'][child_cursor] == 22
            if not exact and len(child_ids):
                d = operator['source_direction'][child_rows]
                u, v = basis(d)
                source_frame = np.stack([u, v, d], axis=1)
                rotation[child_ids] = target[surviving][~parent] @ source_frame
            advances += len(ids)
    assert np.all(status >= 0), 'Unfinished full-family packets'
    assert max_charged_path <= .250001
    return pos, status, source_x, dict(advances=advances, graph_advances=graph_advances,
                                      iterations=iteration, below_minimum_terminal_steps=below_count,
                                      maximum_electron_path_mm=max_charged_path)


def attach_entry(models, report, entry_folder):
    """Cross-energy control: freeze continuation; change only native entry."""
    incoming, other = load_model(entry_folder)
    for key in ['table', 'density_g_cm3', 'production_cut_mm', 'requested_max_step_mm', 'binary_sha256']:
        assert report['source'][key] == other['source'][key], key
    assert other['source']['energy_MeV'] <= report['source']['energy_MeV']
    graph = incoming['graph']
    reached = np.zeros(len(graph['pdg']), bool)
    active = np.unique(incoming['entry']['child'])
    active = active[active >= 0]
    while len(active):
        reached[active] = True
        following = np.unique(np.r_[graph['next'][active], graph['child'][active].ravel()])
        following = following[following >= 0]
        active = following[~reached[following]]
    ids = np.flatnonzero(reached)
    base_count = len(models['graph']['pdg'])
    remap = np.full(len(reached), -1, np.int64)
    remap[ids] = base_count+np.arange(len(ids))
    new_graph = {k:v[ids].copy() for k,v in graph.items()}
    for key in ['next', 'child']:
        r = new_graph[key]
        new_graph[key] = np.where(r >= 0, remap[np.maximum(r, 0)], -1)
    entry = incoming['entry']
    r = entry['child']
    entry['child'] = np.where(r >= 0, remap[np.maximum(r, 0)], -1)
    for key in models['graph']:
        assert models['graph'][key].shape[1:] == new_graph[key].shape[1:]
        models['graph'][key] = np.concatenate([models['graph'][key], new_graph[key]])
    models['entry'] = entry
    return other['continuation_audit']['actual_energy_MeV'], dict(
        report_sha256=sha(entry_folder/'report.json'), source=other['source'],
        attached_descendant_rows=len(ids), continuation_unchanged=True)


def evaluate(folder, out, histories, seed, entry_folder=None):
    out.mkdir(parents=True, exist_ok=False)
    models, report = load_model(folder)
    assert histories >= 200 and histories % 20 == 0
    incident = report['continuation_audit']['actual_energy_MeV']
    entry_control = None
    if entry_folder is not None:
        incident, entry_control = attach_entry(models, report, entry_folder)
    rng = np.random.default_rng(seed); blocks = []; work = []
    counts = np.zeros(4, np.int64); start = time.monotonic()
    for block in range(20):
        n = histories//20
        pos, status, source_x, stats = transport(models, incident, n, rng)
        # Preserve physical source positions for photon geometry. The charged
        # response cannot be slit-convolved after conditioning on box escape.
        profiles = []
        for code in [DEPOSIT, PHOTON_DEPOSIT]:
            x = pos[status == code, 0]
            profiles.append(chord_profile(x, x, np.full(len(x), 1/n), np.zeros(len(x), np.int64), 1)[0])
        h = np.array(profiles)
        assert abs(h.sum()-np.count_nonzero(np.isin(status, [DEPOSIT, PHOTON_DEPOSIT]))/n) < 1e-10
        blocks.append(h); counts += np.bincount(status, minlength=4); work.append(stats)
        print('block', block+1, 'seconds', time.monotonic()-start, 'counts', counts.tolist(), flush=True)
    blocks = np.array(blocks)
    summaries = {}
    for label, profiles in [('charged', blocks[:, 0]), ('photon', blocks[:, 1]), ('full', blocks.sum(axis=1))]:
        rois = np.column_stack([profiles[:, PEAK].sum(axis=1), profiles[:, VALLEY].sum(axis=1)])
        summaries[label] = dict(mean=profiles.mean(axis=0).tolist(), sem=(profiles.std(axis=0, ddof=1)/np.sqrt(20)).tolist(),
                               roi_mean=rois.mean(axis=0).tolist(), roi_sem=(rois.std(axis=0, ddof=1)/np.sqrt(20)).tolist())
    result = dict(status='HOMOGENEOUS_FULL_FAMILY_SCREEN', ct_acceptance=False, production_promoted=False,
                  model_report_sha256=sha(folder/'report.json'), histories=histories, seed=seed,
                  incident_energy_MeV=incident, independent_entry=entry_control,
                  summaries=summaries, counts=dict(zip(['charged_deposit','escape','photon_deposit','coverage_missing'], map(int, counts))),
                  packet_energy_closure=bool(counts.sum() == histories), work=work, seconds=time.monotonic()-start,
                  scope='Actual box escape; all native descendants replayed, monoenergetic entry only; no CT interface model',
                  script_sha256=sha(Path(__file__)))
    np.savez(out/'blocks.npz', profile=blocks)
    save(out/'report.json', result)
    assert counts[COVERAGE] == 0, 'Finite native source coverage is not patient escape'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('compile'); c.add_argument('--source', type=Path, required=True)
    c.add_argument('--source-split', choices=['train','test'], default='train', help='Independent model-training replicate; not a refit to dose')
    e = sub.add_parser('evaluate'); e.add_argument('--model', type=Path, required=True)
    e.add_argument('--entry-from', type=Path, help='Frozen cross-energy diagnostic, using only this native primary entry')
    e.add_argument('--histories', type=int, default=2000000)
    for cmd in [c, e]:
        cmd.add_argument('--out', type=Path, required=True)
        cmd.add_argument('--seed', type=int, default=100305101)
    a = p.parse_args()
    if a.command == 'compile':
        sources=json.loads((a.source/'manifest.json').read_text())
        result = compile_family(next(s for s in sources if s['name']==a.source_split), a.out, a.seed)
    else:
        result = evaluate(a.model, a.out, a.histories, a.seed, a.entry_from)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

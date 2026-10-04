"""Native short-step joint compiler and audited record schema.

Current package builder support. Physical genealogy/energy/step checks are
retained from the validated extraction; historical experimental CLIs are not
runtime or build dependencies.
"""
import json
from pathlib import Path
import numpy as np
from electron_package_common import save

FIELDS = dict(run=(0, '<i4'), event=(4, '<i4'), track=(8, '<i4'), parent=(12, '<i4'),
              pdg=(16, '<i4'), step=(20, '<i4'), birth=(48, '<f8'),
              pre=(56, ('<f8', 3)), post=(80, ('<f8', 3)), dep=(104, '<f8'),
              energy=(112, '<f8'), remaining=(120, '<f8'), version=(144, '<i4'),
              direction=(232, ('<f8', 3)))

EXTRA = dict(post_direction=(256, ('<f8', 3)), parent_step=(196, '<i4'),
             pre_material=(296, '<i4'), post_material=(300, '<i4'), status=(304, '<i4'), length=(280, '<f8'))

FIELDS = dict(FIELDS, **EXTRA)

DTYPE = np.dtype(dict(names=list(FIELDS), formats=[v[1] for v in FIELDS.values()],
                     offsets=[v[0] for v in FIELDS.values()], itemsize=312))

def extract(source):
    path = Path(source['path'])
    assert source['accepted'] and source['geometry'] in ['sphere_in_vacuum', 'voxel_box_in_vacuum']
    assert source['records_include_vacuum']
    assert path.stat().st_size % 312 == 0
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

def basis(direction):
    helper = np.eye(3)[np.argmin(abs(direction), axis=1)]
    u = np.cross(helper, direction); u /= np.linalg.norm(u, axis=1)[:, None]
    return u, np.cross(direction, u)

def compile_model(source, bins, cap, seed, stratified=False, first_only=False,
                  primary_continuing=False, include_source_rows=False):
    # The existing extractor checks the entire graph, energy, continuity,
    # native maximum step, family ledger, and source schema.
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
                 max_children=nchild, stratified=stratified, first_only=first_only,
                 primary_continuing=primary_continuing,
                 native_settings={k:source[k] for k in ['table', 'density_g_cm3', 'production_cut_mm', 'requested_max_step_mm']})
    return model, audit

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
    # The standard compiler audits source records, genealogy, energy, continuity,
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
                  files={k:str(v) for k,v in paths.items()})
    save(out/'report.json', report)
    return report

def load_model(folder):
    report = json.loads((folder/'report.json').read_text())
    result = {}
    for name in ['continuation', 'entry', 'graph']:
        path = folder/(name+'.npz')
        with np.load(path) as f:
            result[name] = {k:f[k].copy() for k in f.files}
    return result, report

STEP=np.dtype([('cdf','<f8'),('delta','<f8',(3,)),('post','<f8',(3,)),('source','<f8',(3,)),
               ('energy','<f8'),('length','<f8'),('fraction','<f8',(4,)),('child','<i4',(2,))])

GRAPH=np.dtype([('delta','<f8',(3,)),('length','<f8'),('fraction','<f8',(4,)),('postdir','<f8',(3,)),
                ('next','<i4'),('child','<i4',(2,)),('pdg','<i4'),('inside','<i4'),('pad','<i4')])

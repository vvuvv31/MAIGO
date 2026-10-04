"""Shared native record schema and metadata output for electron operators."""
import json
from pathlib import Path
import numpy as np

COLLISION_DTYPE = np.dtype(
    [(k, '<i4') for k in ['run', 'event', 'step', 'kind', 'pdg', 'process']]
    + [(k, '<f8') for k in ['energy', 'remaining', 'deposit', 'children']]
    + [('pre', '<f8', 3), ('post', '<f8', 3), ('length', '<f8'), ('density', '<f8')]
)
assert COLLISION_DTYPE.itemsize == 120


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def audit_collision_records(directory, identity, expected_pdg):
    """Audit all recorded branches, event ancestry and the native request ledger."""
    d = Path(directory)
    path = d / 'conditional.phsp'
    r = np.fromfile(path, COLLISION_DTYPE)
    assert r.nbytes == path.stat().st_size and np.isin(r['kind'], [0, 1]).all()
    zero_child = (r['kind'] == 1) & (r['remaining'] == 0)
    for k in COLLISION_DTYPE.names[6:]:
        assert np.isfinite(r[k][~zero_child] if k == 'post' else r[k]).all(), k
    c = r[r['kind'] == 0]
    child = r[r['kind'] == 1]
    assert len(c) > 0 and np.array_equal(c['event'], np.arange(len(c)))
    assert np.all(c['step'] == 1) and np.all(c['pdg'] == expected_pdg)
    assert np.all(c['energy'] > 0) and np.all(c['remaining'] >= 0) and np.all(c['deposit'] >= 0)
    assert np.max(abs(c['energy']-c['remaining']-c['deposit']-c['children'])) < 1e-9
    assert np.isin(child['pdg'], [11, 22]).all() and np.all(child['remaining'] >= 0)
    assert np.all(child['event'] >= 0) and np.all(child['event'] < len(c))
    for k in ['run', 'process', 'energy']:
        assert np.array_equal(child[k], c[k][child['event']])
    transferred = np.bincount(child['event'], weights=child['remaining'], minlength=len(c))
    assert np.max(abs(transferred-c['children'])) < 1e-9
    assert np.max(abs(np.linalg.norm(c['pre'], axis=1)-1), initial=0) < 1e-10
    assert np.max(abs(np.linalg.norm(child['post'][child['remaining'] > 0], axis=1)-1), initial=0) < 1e-10
    audit = np.atleast_1d(np.genfromtxt(d/'conditional_audit.csv', delimiter=',', names=True))
    requests = np.loadtxt(d/'requests.txt', ndmin=2)
    assert len(audit) == len(requests) and len(c) == int(audit['accepted'].sum())
    assert np.all(audit['accepted'] == audit['requested'])
    for key, column in [('cell', 0), ('process', 1), ('energy_MeV', 2), ('requested', 3)]:
        assert np.array_equal(audit[key], requests[:, column])
    for row in audit:
        selected = (c['run'] == int(row['cell'])) & (c['process'] == int(row['process']))
        assert selected.sum() == row['accepted'] and np.all(c['energy'][selected] == row['energy_MeV'])
        assert row['rate_per_mm'] > 0 and row['max_energy_residual_MeV'] < 1e-9
    rho = identity['runtime_density_g_cm3']
    assert np.all(abs(r['density']-rho) <= 4*np.finfo(float).eps*rho)
    return dict(collision_samples=len(c), child_births=len(child), requests=len(audit),
                material_identity=identity, raw_bytes=r.nbytes,
                maximum_residual_MeV=float(audit['max_energy_residual_MeV'].max()),
                null_collision_count=int(audit['null_collisions'].sum()))

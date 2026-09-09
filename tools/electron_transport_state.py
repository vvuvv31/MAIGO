"""Strict opt-in V4 electron step-state reader; no V3 direction inference."""
from pathlib import Path
import re
import numpy as np
from compare_electron_response_geometries import COLUMNS, INT_POS

COLUMNS_V4 = COLUMNS + ['parent_step_id', 'parent_post_ke_MeV',
    'parent_post_dir_x', 'parent_post_dir_y', 'parent_post_dir_z',
    'pre_dir_x', 'pre_dir_y', 'pre_dir_z', 'post_dir_x', 'post_dir_y', 'post_dir_z',
    'step_length_mm', 'post_density_g_cm3', 'pre_material_index', 'post_material_index',
    'post_step_status', 'track_status']
INTS_V4 = INT_POS | {29, 42, 43, 44, 45}
DTYPE_V4 = np.dtype([(name, '<i4' if i in INTS_V4 else '<f8') for i, name in enumerate(COLUMNS_V4)])


def validate_states(d):
    if d.ndim != 2 or d.shape[1] != 46 or not len(d) or not np.isfinite(d).all():
        raise ValueError('Expected nonempty finite V4 transport states')
    if np.any(d[:, 21] != 4):
        raise ValueError('Wrong transport state schema')
    if np.any(d[:, list(INTS_V4)] != np.floor(d[:, list(INTS_V4)])):
        raise ValueError('Nonintegral identity/status')
    if (np.any(d[:, [16, 17, 18, 40, 41]] < 0) or np.any(d[:, 20] <= 0)
            or np.any(d[:, 42] < 0) or np.any(d[:, 43] < -1)):
        raise ValueError('Invalid step energy/length/material')
    # A stopped particle may have zero post direction. Live particles may not.
    for sl, live in ((slice(34, 37), d[:, 17] > 0), (slice(37, 40), d[:, 18] > 0)):
        norm = np.linalg.norm(d[:, sl], axis=1)
        if np.any(np.abs(norm[live]-1) > 1e-6):
            raise ValueError('Non-unit live momentum direction')
    # G4ParticleChangeForMSC independently proposes the post-step position.
    # The recorded displacement is not an exact microscopic path: real stopped
    # electron rows can have chord > GetStepLength(). Preserve both, never clamp
    # either or substitute chord for the slowing-down length. Compiler reports
    # these rows separately; state validity is not a transport-model acceptance.
    if np.any(~np.isin(d[:, 44], np.arange(8))) or np.any(~np.isin(d[:, 45], np.arange(6))):
        raise ValueError('Unknown Geant4 step/track status')
    absent = d[:, 43] == -1
    if np.any(absent & ((d[:, 41] != 0) | (d[:, 44] != 0))):
        raise ValueError('Missing material away from world boundary')
    if np.any(~absent & (d[:, 41] <= 0)):
        raise ValueError('Present post material has invalid density')
    # These are Geant4 run-local material indices, NOT Schneider section IDs.
    # A separately pinned material dictionary is required by a future compiler.
    return d


def displacement_length_audit(d):
    """Report condensed-history geometry differences without editing states."""
    chord = np.linalg.norm(d[:, 13:16]-d[:, 10:13], axis=1)
    excess = chord-d[:, 40]
    selected = excess > 1e-6*np.maximum(1., d[:, 40])
    return dict(chord_exceeds_step_rows=int(np.count_nonzero(selected)),
                max_chord_excess_mm=float(max(0., np.max(excess))),
                affected_deposit_MeV=float(d[selected, 16].sum()))


def map_transport_states(path):
    """Map the exact typed payload without making a full 46-double copy.

    Header/byte-count checks occur immediately. Numerical state validation
    remains mandatory when rows are consumed by the compiler.
    """
    path = Path(path)
    header = path.with_suffix('.header').read_text()
    actual = re.findall(r'^\s*([if]\d+):\s*(\S+)\s*$', header, re.M)
    expected = [('i4' if i in INTS_V4 else 'f8', n) for i, n in enumerate(COLUMNS_V4)]
    if actual != expected:
        raise ValueError('Requires exact V4 header; V3 chord is not electron momentum')
    match = re.search(r'Number of Scored Entries: (\d+)', header)
    if not match or int(match[1]) <= 0 or path.stat().st_size != int(match[1])*DTYPE_V4.itemsize:
        raise ValueError('Incomplete V4 payload')
    return np.memmap(path, dtype=DTYPE_V4, mode='r')


def iter_transport_states(path, rows_per_chunk=262144):
    """Bound working memory for the high-statistics low-density payloads."""
    if type(rows_per_chunk) is not int or rows_per_chunk<=0:
        raise ValueError('Invalid state reader chunk size')
    raw=map_transport_states(path)
    for start in range(0,len(raw),rows_per_chunk):
        part=raw[start:start+rows_per_chunk]
        yield validate_states(np.column_stack([part[n] for n in COLUMNS_V4]))


def load_transport_states(path):
    raw=map_transport_states(path)
    return validate_states(np.column_stack([raw[n] for n in COLUMNS_V4]))

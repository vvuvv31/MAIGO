#!/usr/bin/env python3
"""Compose sphere operators and compare with independent full electron tracks.

Vectorized CPU accuracy experiment, not a GPU throughput claim. Each packet
keeps statistical energy weight while choosing deposit/exit/photon channels.
Linear stochastic interpolation mixes dimensionless kernels and scales the
sampled remaining-energy fraction by the actual incoming energy; this new
approximation must pass independent full-history dose checks.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from prototype import NAMES, measures, periodic_roi, sha, save


def basis(direction):
    helper = np.eye(3)[np.argmin(abs(direction), axis=1)]
    u = np.cross(helper, direction); u /= np.linalg.norm(u, axis=1)[:, None]
    return u, np.cross(direction, u)


def transport(models, energies, incident, histories, rng):
    position = np.zeros((histories, 3)); direction = np.zeros((histories, 3)); direction[:, 2] = 1
    kinetic = np.full(histories, incident); active = np.arange(histories)
    endpoint = np.full((histories, 3), np.nan)
    status = np.full(histories, -1, np.int8)
    advances = 0; max_iterations = 0
    for iteration in range(10000):
        if not len(active):
            break
        e = kinetic[active]
        if np.any(e < energies[0]) or np.any(e > energies[-1]):
            raise ValueError(f'Energy coverage gap [{e.min()}, {e.max()}] in [{energies[0]}, {energies[-1]}]')
        upper = np.clip(np.searchsorted(energies, e, side='right'), 1, len(energies)-1)
        lower = upper-1
        probability = (e-energies[lower])/(energies[upper]-energies[lower])
        selected = np.where(rng.random(len(active)) < probability, upper, lower)
        for index in np.unique(selected):
            ids = active[selected == index]; model = models[index]
            fractions = model['channel_fractions']; choice = rng.random(len(ids))
            if abs(fractions.sum()-1) > 1e-10:
                raise ValueError('Unclosed model channel mass')
            deposited = ids[choice < fractions[0]]
            exiting = ids[(choice >= fractions[0]) & (choice < fractions[0]+fractions[1])]
            photon = ids[choice >= fractions[0]+fractions[1]]
            status[photon] = 2
            if len(deposited):
                row = np.searchsorted(model['deposit_cdf'], rng.random(len(deposited)), side='right')
                if 'deposit_pre' in model:
                    t = rng.random(len(deposited))[:, None]
                    points = model['deposit_pre'][row]+t*(model['deposit_post'][row]-model['deposit_pre'][row])
                    distance = np.linalg.norm(points, axis=1)
                    theta = np.arccos(np.clip(points[:, 2]/np.maximum(distance, 1e-300), -1, 1))
                else:
                    coords = model['deposit'][row].astype(float)
                    distance = .01*np.sinh(coords[:, 0]); theta = coords[:, 1]
                if np.any(distance > model['radius_mm']*(1+1e-6)):
                    raise ValueError('Compressed deposition lies outside sphere')
                phi = rng.random(len(deposited))*2*np.pi
                u, v = basis(direction[deposited])
                transverse = np.cos(phi)[:, None]*u+np.sin(phi)[:, None]*v
                offset = np.sin(theta)[:, None]*transverse+np.cos(theta)[:, None]*direction[deposited]
                endpoint[deposited] = position[deposited]+distance[:, None]*offset
                status[deposited] = 0
            if len(exiting):
                row = np.searchsorted(model['exit_cdf'], rng.random(len(exiting)), side='right')
                theta, alpha, beta, fraction = model['exit'][row].astype(float).T
                if np.any(fraction <= 0) or np.any(fraction >= 1):
                    raise ValueError('Invalid energy continuation')
                phi = rng.random(len(exiting))*2*np.pi
                u, v = basis(direction[exiting])
                transverse = np.cos(phi)[:, None]*u+np.sin(phi)[:, None]*v
                normal = np.sin(theta)[:, None]*transverse+np.cos(theta)[:, None]*direction[exiting]
                tangent = np.sin(theta)[:, None]*direction[exiting]-np.cos(theta)[:, None]*transverse
                side = np.cross(normal, tangent)
                new_direction = np.cos(alpha)[:, None]*normal+np.sin(alpha)[:, None]*(
                    np.cos(beta)[:, None]*tangent+np.sin(beta)[:, None]*side)
                assert np.max(abs(np.linalg.norm(new_direction, axis=1)-1)) < 1e-10
                direction[exiting] = new_direction
                position[exiting] += float(model['radius_mm'])*normal
                kinetic[exiting] *= fraction
        advances += len(active); max_iterations = iteration+1
        active = active[status[active] < 0]
    if len(active):
        raise RuntimeError(f'Unfinished packets: {len(active)}')
    assert np.all(np.isin(status, [0, 2]))
    return endpoint, status, advances, max_iterations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=Path('out/electron_mono_model_20261002'))
    p.add_argument('--histories', type=int, default=2000000)
    p.add_argument('--seed', type=int, default=1002250001)
    p.add_argument('--raw', action='store_true', help='Diagnostic control: uncompressed exact charged segments and exit states')
    p.add_argument('--energy-stride', type=int, default=1, help='Coarser-grid sensitivity control; retain endpoints')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    campaign = json.loads((args.campaign/'status.json').read_text())
    assert campaign['status'] == 'complete'
    cases = sorted(campaign['cases'], key=lambda c: c['actual_energy_MeV'])
    if args.energy_stride < 1:
        p.error('Energy stride must be positive')
    indices = sorted(set(range(0, len(cases), args.energy_stride)) | {len(cases)-1})
    cases = [cases[i] for i in indices]
    energies = np.array([c['actual_energy_MeV'] for c in cases])
    models = []
    for case in cases:
        path = Path(case['model'])/'model.npz'
        assert sha(path) == case['model_sha256']
        if args.raw:
            from macro_model import extract
            source = json.loads((Path(case['reference'])/'manifest.json').read_text())[0]
            data = extract(source)
            weight = data['weight']; ew = data['exit_weight']
            models.append(dict(deposit_pre=data['pre'], deposit_post=data['post'],
                               deposit_cdf=np.cumsum(weight)/weight.sum(), exit=data['exits'],
                               exit_cdf=np.cumsum(ew)/ew.sum() if len(ew) else np.empty(0),
                               channel_fractions=np.array(data['audit']['energy_fractions']),
                               radius_mm=np.array(source['radius_mm'])))
            print('raw model loaded', case['actual_energy_MeV'], flush=True)
        else:
            with np.load(path) as data:
                models.append({key: data[key] for key in data.files})
    # The independent full-track test is the previously generated 0.65 MeV
    # lung reference. Require its conditions explicitly rather than silently
    # applying it to another material/energy campaign.
    reference = json.loads((args.reference/'report.json').read_text())
    assert abs(energies[-1]-.65) < 1e-7
    for case in cases:
        metadata = json.loads((Path(case['reference'])/'manifest.json').read_text())[0]
        assert metadata['table'] == 's1_mid9' and abs(metadata['density_g_cm3']-.2994157075881958)<1e-12
        # A reference with a different native maximum step is a different
        # benchmark; do not accidentally claim a matched-physics acceptance.
        assert metadata.get('requested_max_step_mm', 0) == reference.get('transport_settings', {}).get('max_step_mm', 0)
    assert args.histories >= 200 and args.histories % 20 == 0
    rng = np.random.default_rng(args.seed); block_count = args.histories//20
    metric_blocks = []; profile_blocks = []; totals = np.zeros(3, np.int64)
    centres = np.arange(-1.75, 1.8, .1)
    steps = 0; max_iterations = 0; start = time.monotonic()
    for block in range(20):
        endpoints, statuses, count, iteration = transport(models, energies, energies[-1], block_count, rng)
        steps += count; max_iterations = max(iteration, max_iterations)
        pnt = endpoints[statuses == 0]; r = np.hypot(pnt[:, 0], pnt[:, 1]); z = pnt[:, 2]
        metric_blocks.append(measures(r, z).sum(axis=0)/block_count)
        profile_blocks.append(np.array([periodic_roi(r, float(x), halfwidth=.05).sum() for x in centres])/block_count)
        totals += np.bincount(statuses, minlength=3)
        save(args.out/'progress.json', dict(blocks_complete=block+1, histories=int(totals.sum()), pid=__import__('os').getpid()))
        print('block', block+1, 'seconds', time.monotonic()-start, flush=True)
    values = np.array(metric_blocks); mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1)/np.sqrt(20)
    checks = {}
    for i, name in enumerate(NAMES):
        old = reference['independent_absolute_errors'][name]
        observed = old['reference']; ref_sem = old['test_block_standard_error']
        diff = (mean[i]-observed)/observed if abs(observed)>1e-12 else None
        half = 2.44*np.hypot(sem[i], ref_sem)/abs(observed) if abs(observed)>1e-12 else None
        checks[name] = dict(reference=observed, model=float(mean[i]), relative_difference=diff,
                            conditional_two_roi_relative_halfwidth=float(half) if half is not None else None,
                            conditional_envelope=abs(diff)+half if half is not None else None)
    point_pass = all(abs(checks[k]['relative_difference']) < .01 for k in ['peak', 'valley'])
    profile = np.array(profile_blocks)
    with np.load(args.reference/'local_profile.npz') as ref_profile:
        independent_profile = ref_profile['reference'].copy()
        ref_profile_sem = ref_profile['test_sem'].copy()
    profile_diff = (profile.mean(axis=0)-independent_profile)/independent_profile
    report = dict(status='HOMOGENEOUS_COMPOSITION_SCREEN', ct_acceptance=False,
                  raw_response_control=args.raw,
                  energy_stride=args.energy_stride, selected_energy_nodes_MeV=energies.tolist(),
                  histories=args.histories, seed=args.seed, seconds=time.monotonic()-start,
                  payload_bytes=sum(sum(v.nbytes for v in m.values()) for m in models), packet_advances=steps, max_macro_steps=max_iterations,
                  counts=dict(deposit=int(totals[0]), photon_sink=int(totals[2])),
                  packet_energy_closure=bool(totals[0]+totals[2] == args.histories),
                  metrics=checks, peak_valley_point_estimates_within_1pct=point_pass,
                  max_local_0p1mm_profile_relative_difference=float(abs(profile_diff).max()),
                  interpolation='Linear stochastic mixing of dimensionless sphere kernels; remaining E fraction times incoming E',
                  statistical_scope='Intervals include sampling and independent full-track reference noise, conditional on the single trained bank; bank training uncertainty is not included',
                  campaign_sha256=sha(args.campaign/'status.json'),
                  reference_sha256=sha(args.reference/'report.json'), script_sha256=sha(Path(__file__)))
    np.savez_compressed(args.out/'dose_profiles.npz', x_mm=centres, model=profile.mean(axis=0),
                        reference=independent_profile, model_sem=profile.std(axis=0, ddof=1)/np.sqrt(20),
                        reference_sem=ref_profile_sem, relative_difference=profile_diff)
    save(args.out/'report.json', report); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

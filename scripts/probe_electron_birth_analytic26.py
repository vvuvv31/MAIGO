#!/usr/bin/env python3
"""Check an energy-weighted Born delta-electron law against native birth banks.

Offline only. No package is replaced and no GPU accuracy claim is made.
The reference records are sampled jointly, never as independent marginals.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

CHANNEL = np.dtype([('low','<f8'), ('high','<f8'), ('fraction','<f8'),
                    ('offset','<u8'), ('count','<u8')])
SAMPLE = np.dtype([('cdf','<f8'), ('energy','<f8'), ('cosine','<f8')])
ME = .51099895


def native_nodes(path):
    with path.open('rb') as f:
        magic, version, count, z, a = struct.unpack('<8s4I', f.read(24))
        if (magic, version) != (b'MAIBIR02', 2):
            raise ValueError('Native birth schema required')
        for _ in range(count):
            section, density, samples, bins = struct.unpack('<idQI', f.read(24))
            channels = np.frombuffer(f.read(40*bins), CHANNEL).copy()
            offset = f.tell()
            states = np.memmap(path, SAMPLE, mode='r', offset=offset, shape=(samples,))
            yield section, density, z, a, channels, states
            f.seek(offset+24*samples)
        if f.read(1):
            raise ValueError('Trailing birth bytes')


def cuts(root):
    result = {}
    for line in (root/'index.txt').read_text().splitlines()[1:]:
        section, density, relative = line.split()
        with (root/relative).open('rb') as f:
            header = struct.unpack('<8sIid4IQ4d', f.read(80))
        if header[2] != int(section) or header[3] != float(density):
            raise ValueError('Charged material identity mismatch')
        result[int(section), float(density)] = header[9]
    return result


def born_joint(low, high, cut, a, parent_points=32, transfer_points=256):
    """Joint quadrature of the energy-weighted relativistic free-electron law.

    Parent-energy averaging assumes a uniform slowing interval; the candidate
    intentionally does not fit native density nodes or silently calibrate yield.
    """
    mass = 938.27208816 if a == 1 else 11174.862/a
    xp, wp = np.polynomial.legendre.leggauss(parent_points)
    xt, wt = np.polynomial.legendre.leggauss(transfer_points)
    parent = low+(xp+1)*(high-low)/2
    gamma = 1+parent/mass
    beta2 = 1-1/gamma**2
    total = a*(mass+parent)
    momentum = a*np.sqrt(parent*(parent+2*mass))
    ceiling = 2*ME*(gamma**2-1)/(1+2*gamma*ME/(a*mass)+(ME/(a*mass))**2)
    valid = ceiling > cut
    if not valid.any():
        return np.empty(0), np.empty(0), np.empty(0)
    ceiling, beta2, total, momentum, wp = [x[valid] for x in (ceiling,beta2,total,momentum,wp)]
    log_width = np.log(ceiling/cut)
    transfer = cut*np.exp((xt[None,:]+1)*log_width[:,None]/2)
    # d(sigma)/dT times T, transformed to d(log T). The spin-half term is
    # present for proton; spin-zero C12 has no such term.
    law = 1-beta2[:,None]*transfer/ceiling[:,None]
    if a == 1:
        law += transfer**2/(2*total[:,None]**2)
    weight = wp[:,None]*wt[None,:]*log_width[:,None]*law/beta2[:,None]
    cosine = np.sqrt(transfer/(transfer+2*ME))*(total[:,None]+ME)/momentum[:,None]
    if not np.isfinite(weight).all() or np.any(weight<0) or np.any(abs(cosine)>1+1e-12):
        raise ValueError('Born joint support')
    weight /= weight.sum()
    return transfer.ravel(), cosine.ravel(), weight.ravel()


def compare(reference, energy, cosine, weight):
    rw = np.diff(np.r_[0., reference['cdf']])
    if np.any(rw<0) or abs(rw.sum()-1)>1e-12:
        raise ValueError('Reference probability intervals')
    re, rc = reference['energy'], reference['cosine']
    thresholds = np.unique(np.quantile(re, np.linspace(0,1,33)))
    angles = np.linspace(0,1,17)
    energy_ks = max(abs(rw[re<=t].sum()-weight[energy<=t].sum()) for t in thresholds)
    joint = max(abs(rw[(re<=t)&(rc<=mu)].sum()-weight[(energy<=t)&(cosine<=mu)].sum())
                for t in thresholds[::4] for mu in angles)
    native_mean = float(rw@re)
    return dict(energy_cdf_max=float(energy_ks), joint_cdf_max=float(joint),
                mean_energy_relative=float(weight@energy/native_mean-1),
                mean_cosine_difference=float(weight@cosine-rw@rc),
                effective_samples=float(1/(rw@rw)), native_mean_energy_MeV=native_mean)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/archive/electron_density_nodes_20261006'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minimum-energy-mevu',type=float,default=50.)
    p.add_argument('--channel-stride',type=int,default=5)
    args=p.parse_args()
    if args.channel_stride<1 or args.minimum_energy_mevu<0:
        p.error('Invalid energy/stride')
    lookup=cuts(args.root/'shared/charged');rows=[]
    for projectile in ['proton','carbon']:
        for section,density,z,a,channels,states in native_nodes(args.root/projectile/'birth_bank.bin'):
            cut=lookup[section,density]
            for i,c in enumerate(channels):
                if c['low']<args.minimum_energy_mevu or i%args.channel_stride or not c['count']:
                    continue
                block=states[int(c['offset']):int(c['offset']+c['count'])]
                energy,cosine,weight=born_joint(float(c['low']),float(c['high']),cut,a)
                if not len(weight):
                    raise ValueError('Native positive yield has no analytic support')
                row=dict(projectile=projectile,section=section,density=float(density),
                         channel=i,cut_MeV=cut,**compare(block,energy,cosine,weight))
                rows.append(row)
            print(projectile,section,density,flush=True)
    summaries={}
    for projectile in ['proton','carbon']:
        group=[r for r in rows if r['projectile']==projectile]
        summaries[projectile]={k:dict(maximum=max(abs(r[k]) for r in group),
                                      median=float(np.median([abs(r[k]) for r in group])))
                               for k in ['energy_cdf_max','joint_cdf_max','mean_energy_relative','mean_cosine_difference']}
    report=dict(status='OFFLINE_BORN_PROTOTYPE_NOT_RUNTIME_OR_DOSE_ACCEPTANCE',
                assumptions='free-electron law; uniform parent slowing bins; no fitted density residuals; yield not evaluated',
                configuration=dict(minimum_energy_MeVu=args.minimum_energy_mevu,channel_stride=args.channel_stride),
                summaries=summaries,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()

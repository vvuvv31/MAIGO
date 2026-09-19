#!/usr/bin/env python3
"""Step 22: generate schneider-secondary-v2 campaign manifest + case files.

Reads out/reachable_channel_audit.json, the secondary rate table, and both
CINEL03 packages. Writes:
  /mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2/
    cases/<tag>/run.txt          (one TOPAS config per (proj,target,E) task)
    scripts/worker_<pid>.py      (direct-TOPAS runner, mirrors v1 template)
    scripts/sbatch_<pid>.sh      (12 CPUs / 8 GiB per projectile job)
    scripts/sbatch_c12floors.sh
    campaign_manifest.json

Design (documented, no silent choices):
- gap-fill: equidistant inserts so every adjacent gap <= 5.0 MeV/u.
- floor extension to 0.5 MeV/u (rate-table low edge) for all 168+13 channels.
- ceil extension to 430 MeV/u (rate-table high edge) for all channels.
- p+H: full grid [1.0, 430] (proton rate is exactly 0 below 1.0, so no
  proton query can occur below the planned floor).
- C12: floor extension only (v1 gaps already <= 2.8, ceil already ~430).
- Tier-L probe (non-blocking): 0.1-0.4 MeV/u x top-5 demand channels to
  measure sub-0.5 yield; empty nodes never enter the package.
- histories: E<5 -> 10000, Tier-L -> 20000, else 2000.
- Existing nodes are never regenerated (skip within 1e-6).
- Seeds: 3000000 + manifest task index, written as i:Ts/Seed.
"""
import hashlib
import json
import math
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(os.environ.get("MAIGO_REPO_ROOT", Path(__file__).resolve().parents[3]))
V2 = Path(os.environ.get("MAIGO_V2_CAMPAIGN_ROOT", REPO / "extensions" / "work" / "schneider-secondary-v2"))
MAX_GAP = 5.0
RATE_EMIN, RATE_EMAX = 0.5, 430.0
PH_FLOOR = 1.0
CEIL_TARGET = 430.0
FLOOR_TARGET = 0.5
SEED_BASE = 3000000
CAMPAIGN_UUID = "00000000-0000-4000-8000-000000000022"

PROJ_ORDER = [(5, 11), (5, 10), (4, 9), (4, 7), (4, 10), (3, 7), (3, 6),
              (2, 4), (2, 3), (1, 1), (1, 2), (1, 3), (6, 11)]
PROJ_PARTICLE = {(5, 11): "GenericIon(5,11)", (5, 10): "GenericIon(5,10)",
                 (4, 9): "GenericIon(4,9)", (4, 7): "GenericIon(4,7)",
                 (4, 10): "GenericIon(4,10)", (3, 7): "GenericIon(3,7)",
                 (3, 6): "GenericIon(3,6)", (2, 4): "alpha",
                 (2, 3): "He3", (1, 1): "proton", (1, 2): "deuteron",
                 (1, 3): "triton", (6, 11): "GenericIon(6,11)",
                 (6, 12): "GenericIon(6,12)"}
PROJ_ID = {(5, 11): "b11", (5, 10): "b10", (4, 9): "be9", (4, 7): "be7",
           (4, 10): "be10", (3, 7): "li7", (3, 6): "li6", (2, 4): "he4",
           (2, 3): "he3", (1, 1): "h1", (1, 2): "h2", (1, 3): "h3",
           (6, 11): "c11", (6, 12): "c12"}
TARGETS = [(1, "H", "Target_H", "Hydrogen", 0.1, 200.0),
           (6, "C", "Target_C", "Carbon", 2.0, 100.0),
           (7, "N", "Target_N", "Nitrogen", 1.0, 100.0),
           (8, "O", "Target_O", "Oxygen", 1.0, 100.0),
           (11, "Na", "Target_Na", "Sodium", 0.97, 100.0),
           (12, "Mg", "Target_Mg", "Magnesium", 1.74, 100.0),
           (15, "P", "Target_P", "Phosphorus", 1.82, 100.0),
           (16, "S", "Target_S", "Sulfur", 2.07, 100.0),
           (17, "Cl", "Target_Cl", "Chlorine", 1.56, 100.0),
           (18, "Ar", "Target_Ar", "Argon", 1.40, 100.0),
           (19, "K", "Target_K", "Potassium", 0.86, 100.0),
           (20, "Ca", "Target_Ca", "Calcium", 1.55, 100.0),
           (22, "Ti", "Target_Ti", "Titanium", 4.54, 50.0)]
TGT = {z: t for t in TARGETS for z in [t[0]]}

TEMPLATE = """s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 1000.0 mm
d:Ge/World/HLY = 1000.0 mm
d:Ge/World/HLZ = 2000.0 mm
b:Ge/QuitIfOverlapDetected = "False"

s:Ge/BeamPosition/Parent = "World"
d:Ge/BeamPosition/TransX = 0.0 mm
d:Ge/BeamPosition/TransY = 0.0 mm
d:Ge/BeamPosition/TransZ = 1500.0 mm
d:Ge/BeamPosition/RotX = 180.0 deg

s:Ge/Phantom/Type = "TsBox"
s:Ge/Phantom/Parent = "World"
s:Ge/Phantom/Material = "{mat_name}"
d:Ge/Phantom/HLX = 50.0 mm
d:Ge/Phantom/HLY = 50.0 mm
d:Ge/Phantom/HLZ = {hlz:.1f} mm
d:Ge/Phantom/TransX = 0.0 mm
d:Ge/Phantom/TransY = 0.0 mm
d:Ge/Phantom/TransZ = 0.0 mm

sv:Ma/{mat_name}/Components = 1 "{elem_name}"
uv:Ma/{mat_name}/Fractions = 1 1.0
d:Ma/{mat_name}/Density = {density} g/cm3

s:So/PrimaryBeam/Type = "Beam"
s:So/PrimaryBeam/Component = "BeamPosition"
s:So/PrimaryBeam/BeamParticle = "{part_name}"
d:So/PrimaryBeam/BeamEnergy = {beam_energy:.2f} MeV
u:So/PrimaryBeam/BeamEnergySpread = 0.0
s:So/PrimaryBeam/BeamPositionDistribution = "None"
s:So/PrimaryBeam/BeamAngularDistribution = "None"
i:So/PrimaryBeam/NumberOfHistoriesInRun = {histories}
i:Ts/Seed = {seed}

s:Sc/CarbonInelasticExposure/Quantity = "CarbonInelasticExposureNtuple"
s:Sc/CarbonInelasticExposure/Component = "Phantom"
s:Sc/CarbonInelasticExposure/OutputType = "ASCII"
s:Sc/CarbonInelasticExposure/OutputFile = "cinel03_exposure"
s:Sc/CarbonInelasticExposure/IfOutputFileAlreadyExists = "Overwrite"
i:Sc/CarbonInelasticExposure/ProjectileZ = {pz}
i:Sc/CarbonInelasticExposure/ProjectileA = {pa}
b:Sc/CarbonInelasticExposure/IncludeSecondaries = "FALSE"
d:Sc/CarbonInelasticExposure/EnergyBinMin = 0.0 MeV
d:Sc/CarbonInelasticExposure/EnergyBinWidth = 1.0 MeV
i:Sc/CarbonInelasticExposure/EnergyBinCount = 501
b:Sc/CarbonInelasticExposure/RequireAuthoritativeCollisionState = "TRUE"

sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4ion-inclxx" "CarbonInelasticCapturePhysics" "g4h-elastic_HP" "g4stopping"
d:Ph/Default/CutForAllParticles = 0.05 mm
"""


def load_nodes(bin_path):
    with open(bin_path, "rb") as f:
        h = f.read(136)
        cc, ic, pc = struct.unpack_from("<3Q", h, 8 + 28)
        f.seek(136 + cc * 36)
        raw = f.read(ic * 476)
    chans = defaultdict(list)
    for i in range(ic):
        b = raw[i * 476:(i + 1) * 476]
        pz = struct.unpack_from("<h", b, 36)[0]
        pa = struct.unpack_from("<h", b, 38)[0]
        epu = struct.unpack_from("<f", b, 56)[0]
        tz = struct.unpack_from("<h", b, 108)[0]
        chans[(pz, pa, tz)].append(epu)
    return {k: sorted(v) for k, v in chans.items()}


def load_demand():
    import numpy as np
    p = REPO / "data/schneider/secondary_inelastic_rates_v1.bin"
    with open(p, "rb") as f:
        hdr = f.read(52)
        magic, ver, np_, ns, nt, ne, emin, emax, estep = struct.unpack("<8s5I3d", hdr)
        projs = [tuple(struct.unpack("<ii", f.read(8))) for _ in range(np_)]
        tgt = list(struct.unpack(f"<{nt}i", f.read(4 * nt)))
        partial = np.fromfile(f, dtype=np.float64,
                              count=np_ * ns * nt * ne).reshape(np_, ns, nt, ne)
    dem = {}
    for pi, pr in enumerate(projs):
        for ti, tz in enumerate(tgt):
            dem[(pr[0], pr[1], tz)] = float(partial[pi, :, ti, :].sum())
    return dem


def fill_gaps(nodes):
    """Equidistant inserts so every adjacent gap <= MAX_GAP."""
    out = []
    a = sorted(nodes)
    for i in range(len(a) - 1):
        gap = a[i + 1] - a[i]
        n = max(1, math.ceil(gap / MAX_GAP - 1e-9))
        for k in range(1, n):
            out.append(a[i] + k * gap / n)
    return out


def extend_ends(nodes, lo, hi):
    a = sorted(nodes)
    out = []
    if a[0] > lo + 1e-9:
        n = math.ceil((a[0] - lo) / MAX_GAP - 1e-9)
        for k in range(n):
            out.append(lo + k * (a[0] - lo) / n)
    if a[-1] < hi - 1e-9:
        n = math.ceil((hi - a[-1]) / MAX_GAP - 1e-9)
        for k in range(1, n + 1):
            out.append(a[-1] + k * (hi - a[-1]) / n)
    return out


def main():
    sec_nodes = load_nodes(REPO / "data/schneider/cinel03_secondary_targets.bin")
    c12_nodes = load_nodes(REPO / "data/schneider/cinel03_c12_targets.bin")
    demand = load_demand()

    tasks = []  # (pz,pa,tz,E,histories,kind,demand)
    # 1. secondary gap-fill + floor/ceil extension
    for (pz, pa, tz), nodes in sorted(sec_nodes.items()):
        dem = demand.get((pz, pa, tz), 0.0)
        for e in fill_gaps(nodes):
            tasks.append((pz, pa, tz, e, 2000 if e >= 5 else 10000, "gap-fill", dem))
        for e in extend_ends(nodes, FLOOR_TARGET, CEIL_TARGET):
            tasks.append((pz, pa, tz, e, 10000 if e < 5 else 2000, "domain-extension", dem))
    # 2. p+H full grid
    dem_ph = demand.get((1, 1, 1), 0.0)
    n_ph = math.ceil((CEIL_TARGET - PH_FLOOR) / MAX_GAP - 1e-9)
    for k in range(n_ph + 1):
        e = PH_FLOOR + k * (CEIL_TARGET - PH_FLOOR) / n_ph
        tasks.append((1, 1, 1, e, 10000 if e < 5 else 2000, "missing-channel-pH", dem_ph))
    # 3. C12 floor extension only
    for (pz, pa, tz), nodes in sorted(c12_nodes.items()):
        for e in extend_ends(nodes, FLOOR_TARGET, nodes[-1]):
            tasks.append((pz, pa, tz, e, 10000 if e < 5 else 2000, "c12-floor-extension", 0.0))
    # 4. Tier-L probe: sub-0.5 yield measurement, top-5 demand secondary channels
    top5 = sorted(demand.items(), key=lambda kv: -kv[1])[:5]
    for (pz, pa, tz), dem in top5:
        for e in (0.1, 0.2, 0.3, 0.4):
            tasks.append((pz, pa, tz, e, 20000, "tierL-subhalf-probe", dem))

    # drop points within 1e-6 of an existing node of the same channel
    have = defaultdict(list, {k: list(v) for k, v in sec_nodes.items()})
    have.update({k: list(v) for k, v in c12_nodes.items()})
    filtered = []
    for t in tasks:
        pz, pa, tz, e = t[0], t[1], t[2], t[3]
        if any(abs(e - x) < 1e-6 for x in have[(pz, pa, tz)]):
            continue
        have[(pz, pa, tz)].append(e)
        filtered.append(t)
    # priority by demand (Tier-L probe deprioritized: measurement only)
    filtered.sort(key=lambda t: (t[5] == "tierL-subhalf-probe", -t[6]))

    cases_dir = V2 / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    manifest_tasks = []
    for i, (pz, pa, tz, e, hist, kind, dem) in enumerate(filtered):
        pid = PROJ_ID[(pz, pa)]
        sym = TGT[tz][1]
        tag = f"{pid}_{sym}_E{e:.2f}".replace(".", "p")
        case_dir = cases_dir / tag
        case_dir.mkdir(parents=True, exist_ok=True)
        _, _, mat_name, elem_name, density, hlz = TGT[tz]
        content = TEMPLATE.format(
            mat_name=mat_name, hlz=hlz, elem_name=elem_name, density=density,
            part_name=PROJ_PARTICLE[(pz, pa)], beam_energy=e * pa,
            histories=hist, seed=SEED_BASE + i, pz=pz, pa=pa)
        (case_dir / "run.txt").write_text(content)
        cfg_hash = hashlib.sha256(content.encode()).hexdigest()
        manifest_tasks.append({
            "task_index": i, "projectile_Z": pz, "projectile_A": pa,
            "target_Z": tz, "energy_MeV_per_u": round(e, 4),
            "requested_events": 30, "requested_histories": hist,
            "priority": i + 1, "kind": kind, "demand_proxy": dem,
            "source_of_demand": ("reachable-channel-audit+secondary-rate-table"
                                 if kind != "tierL-subhalf-probe" else
                                 "sub-0.5-yield-probe(top-5 demand channels)"),
            "expected_output": f"cases/{tag}/cinel03_exposure.worker_*.compact.csv",
            "random_seed": SEED_BASE + i, "TOPAS_config_hash": cfg_hash,
            "case_dir": f"cases/{tag}",
        })

    manifest = {
        "schema_version": 1, "campaign": "schneider-secondary-v2",
        "campaign_uuid": CAMPAIGN_UUID,
        "fixed_max_gap_MeV_per_u": MAX_GAP,
        "rate_domain_MeV_per_u": [RATE_EMIN, RATE_EMAX],
        "seed_base": SEED_BASE,
        "topas_binary": os.environ.get("TOPAS_BINARY", "topas"),
        "topas_version": "4.2.p3", "geant4_version": "geant4-11-03-patch-02",
        "physics_list": 'g4em-standard_opt4 g4h-phy_QGSP_BIC_HP g4ion-inclxx CarbonInelasticCapturePhysics g4h-elastic_HP g4stopping',
        "generator": "tools/generate_v2_manifest.py (repo working tree)",
        "task_count": len(manifest_tasks),
        "tasks": manifest_tasks,
    }
    (V2 / "campaign_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"tasks: {len(manifest_tasks)} "
          f"(gap/dp/pH/c12/tierL = "
          f"{sum(1 for t in manifest_tasks if t['kind']=='gap-fill')}/"
          f"{sum(1 for t in manifest_tasks if t['kind']=='domain-extension')}/"
          f"{sum(1 for t in manifest_tasks if t['kind']=='missing-channel-pH')}/"
          f"{sum(1 for t in manifest_tasks if t['kind']=='c12-floor-extension')}/"
          f"{sum(1 for t in manifest_tasks if t['kind']=='tierL-subhalf-probe')})")
    print(f"total requested histories: {sum(t['requested_histories'] for t in manifest_tasks)}")


if __name__ == "__main__":
    sys.exit(main())

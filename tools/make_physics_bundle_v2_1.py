#!/usr/bin/env python3
"""Generate data/schneider/schneider_physics_bundle_v2_1.json.

Cross-consistency is verified AT GENERATION (fail-closed):
- secondary rate keys == 14p package channel projectile set == registry
- secondary rate domains == 14p channel bounds (all 182)
- primary rate domains == C12 v2.1 package channel bounds (all 13)
- target orders canonical; grids match; file SHAs recorded.
The C++ startup check re-verifies SHAs + registry + domains at load.
"""
import datetime
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path("/mnt/sdb/wuwei/MAIGO")
D = REPO / "data/schneider"
CANON_Z = [1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22]
FILES = {
    "primary_rate": "schneider_inelastic_rates_v2_1.bin",
    "primary_package": "cinel03_c12_targets_v2_1.bin",
    "secondary_rate": "secondary_inelastic_rates_v2_1.bin",
    "secondary_package": "cinel03_secondary_targets_v2_1_14p.bin",
    "stopping_table": "schneider_stopping_v1.bin",
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_rate_keys(path, magic, nkeys):
    with open(path, "rb") as f:
        assert f.read(8) == magic
        ver = struct.unpack("<I", f.read(4))[0]
        assert ver == 3, (path, ver)
        if magic == b"SCHN2RAT":
            np, ns, nt, ne = struct.unpack("<IIII", f.read(16))
            assert (np, ns, nt) == (14, 25, 13), (path, np, ns, nt)
        else:
            ns, nt, ne = struct.unpack("<III", f.read(12))
            assert (ns, nt) == (25, 13)
        emin, emax, estep = struct.unpack("<ddd", f.read(24))
        assert abs(emin - 0.1) < 1e-9 and abs(emax - 460.1) < 1e-9, path
        keys = []
        if magic == b"SCHN2RAT":
            for _ in range(np):
                keys.append(tuple(struct.unpack("<ii", f.read(8))))
        tg = list(struct.unpack(f"<{nt}i", f.read(4 * nt)))
        assert tg == CANON_Z, (path, tg)
        # domain block at end: np or 1 x 13 entries
        f.seek(0, 2)
        size = f.tell()
        ndom = (np if magic == b"SCHN2RAT" else 1) * 13
        f.seek(size - ndom * 24)
        doms = {}
        order = keys if magic == b"SCHN2RAT" else [(6, 12)]
        for p in order:
            for tz in CANON_Z:
                lo, hi = struct.unpack("<dd", f.read(16))
                has = struct.unpack("<B", f.read(1))[0]
                f.read(7)
                assert has == 1, (path, p, tz)
                doms[p + (tz,)] = (lo, hi)
    return keys, doms


def main():
    for name, fn in FILES.items():
        assert (D / fn).exists(), f"missing {fn}"
    sec_keys, sec_doms = read_rate_keys(D / FILES["secondary_rate"], b"SCHN2RAT", 14)
    _, pri_doms = read_rate_keys(D / FILES["primary_rate"], b"SCHNRATE", 0)

    sec_ch = json.load(open(D / "cinel03_secondary_targets_v2_1_14p.channels.json"))["channels"]
    sec_pkg = {(c["projectile_z"], c["projectile_a"], c["target_element_z"]):
               (c["energy_min_MeV_per_u"], c["energy_max_MeV_per_u"]) for c in sec_ch}
    assert len(sec_pkg) == 182, len(sec_pkg)
    assert sorted(set(k[:2] for k in sec_pkg)) == sorted(sec_keys), "registry != package projectiles"
    assert sec_doms == sec_pkg, "secondary rate domains != package bounds"

    pri_ch = json.load(open(D / "cinel03_c12_targets_v2_1.channels.json"))["channels"]
    pri_pkg = {(c["projectile_z"], c["projectile_a"], c["target_element_z"]):
               (c["energy_min_MeV_per_u"], c["energy_max_MeV_per_u"]) for c in pri_ch}
    assert len(pri_pkg) == 13
    assert pri_doms == pri_pkg, "primary rate domains != package bounds"

    bundle = {
        "schema_version": 1,
        "bundle_name": "schneider_physics_bundle_v2_1",
        "primary_rate": {"file": "data/schneider/" + FILES["primary_rate"],
                         "sha256": sha256_file(D / FILES["primary_rate"])},
        "primary_package": {"file": "data/schneider/" + FILES["primary_package"],
                            "sha256": sha256_file(D / FILES["primary_package"]),
                            "channels_file": "data/schneider/cinel03_c12_targets_v2_1.channels.json",
                            "channels_sha256": sha256_file(
                                D / "cinel03_c12_targets_v2_1.channels.json")},
        "secondary_rate": {"file": "data/schneider/" + FILES["secondary_rate"],
                           "sha256": sha256_file(D / FILES["secondary_rate"])},
        "secondary_package": {"file": "data/schneider/" + FILES["secondary_package"],
                              "sha256": sha256_file(D / FILES["secondary_package"]),
                              "channels_file": "data/schneider/cinel03_secondary_targets_v2_1_14p.channels.json",
                              "channels_sha256": sha256_file(
                                  D / "cinel03_secondary_targets_v2_1_14p.channels.json")},
        "stopping_table": {"file": "data/schneider/" + FILES["stopping_table"],
                           "sha256": sha256_file(D / FILES["stopping_table"])},
        "schneider_source": {"file": "data/HUtoMaterialSchneider.txt",
                             "sha256": sha256_file(REPO / "data/HUtoMaterialSchneider.txt")},
        "projectile_registry": [{"z": z, "a": a} for z, a in sec_keys],
        "target_order": CANON_Z,
        "energy_grid": {"emin": 0.1, "emax": 460.1, "step": 0.5, "count": 921},
        "physics_list_primary": "FTFP_INCLXX",
        "physics_list_secondary": "g4em-standard_opt4 + g4ion-binarycascade",
        "topas_version": "4.2.p3", "geant4_version": "11.03.p02",
        "generator_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "generation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    out = D / "schneider_physics_bundle_v2_1.json"
    out.write_text(json.dumps(bundle, indent=1))
    print(f"bundle written: {out}")
    print(f"  registry: {len(sec_keys)} projectiles; domains verified: "
          f"{len(sec_doms)} secondary + {len(pri_doms)} primary")


if __name__ == "__main__":
    sys.exit(main())

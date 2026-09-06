"""Prepare fresh HU=-975/-951 reference-only density pilots; never submit jobs.
Clones the validated CCTG geometry/composition; only density payload changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def density(hu, table):
    lines = table.read_text().splitlines()
    def vector(key):
        line = next(x for x in lines if x.startswith(key))
        parts = line.split("=",1)[1].split()
        return list(map(float, parts[1:1+int(parts[0])]))
    if not -1000 <= hu < -950:
        raise ValueError("Only section-0 diagnostic inputs allowed")
    off = vector("uv:Ge/Patient/SchneiderDensityOffset")[0]
    factor = vector("uv:Ge/Patient/SchneiderDensityFactor")[0]
    shift = vector("uv:Ge/Patient/SchneiderDensityFactorOffset")[0]
    correction = vector("dv:Ge/Patient/DensityCorrection")[hu+1000]
    return float(np.float32((off+factor*(shift+hu))*correction))

def prepare(template, root, table, grid):
    template, root = template.resolve(), root.resolve()
    base = grid.read_bytes()
    magic, version, nx, ny, nz = struct.unpack_from("<5I", base)
    if (magic,version,nx,ny,nz) != (0x47544343,3,100,100,440):
        raise ValueError("Unexpected CCTG template")
    if struct.unpack_from("<6f",base,20) != (-100.,-100.,0.,2.,2.,.5):
        raise ValueError("Unexpected geometry")
    n = nx*ny*nz
    oldrho = np.frombuffer(base, "<f4", count=n, offset=44)
    if not np.all(oldrho == np.float32(density(-1000,table))):
        raise ValueError("Reference density differs from Schneider formula")
    if any(base[44+4*n:44+5*n]):
        raise ValueError("Non-section-0 material in template")
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="REFERENCE_ONLY_UNSUPPORTED_DENSITY",
        energy_MeV_u=175, allowed_total_cpus=96, allowed_total_mem_GB=40,
        grid_template_sha256=digest(grid), schneider_sha256=digest(table), cases={})
    for hu in (-975,-951):
        dest = root/f"hu{-hu}"
        dest.mkdir()
        rho = density(hu,table)
        phantom = dest/"phantom.cctg"
        phantom.write_bytes(base[:44] + np.full(n,rho,dtype="<f4").tobytes() + base[44+4*n:])
        contract=json.loads((template/"campaign.json").read_text())
        contract.update(root=str(dest),density_g_cm3=rho,hu=hu,status="reference_only_density_pilot")
        contract["limitations"].append("Candidate must remain inactive at this unsupported density.")
        (dest/"campaign.json").write_text(json.dumps(contract,indent=2)+"\n")
        for name in ("topas_s1","topas_s2","gpu_base","gpu_long"):
            case=dest/name
            case.mkdir()
            if name.startswith("topas"):
                text=(template/name/"run.txt").read_text().replace(str(template),str(dest))
                text=text.replace("PatientTissueFromHUNegative1000",f"PatientTissueFromHUNegative{-hu}")
                (case/"run.txt").write_text(text)
                job=(template/name/"run.slurm").read_text().replace(str(template),str(dest))
                job=job.replace("hold175_",f"rho{-hu}_")
                (case/"run.slurm").write_text(job)
            else:
                text=(template/name/"run.yaml").read_text().replace(str(template),str(dest))
                text=text.replace(str(grid),str(phantom))
                (case/"run.yaml").write_text(text)
        manifest["cases"][str(hu)] = dict(density_g_cm3=rho,
            phantom_sha256=digest(phantom),
            material_id=0, unchanged_header_and_composition=True)
    manifest["prepared_files"]={str(p.relative_to(root)):digest(p)
        for p in sorted(root.rglob("*")) if p.is_file()}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--template",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--schneider",type=Path,required=True)
    p.add_argument("--grid",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.template,a.out,a.schneider.resolve(),a.grid.resolve()),indent=2))

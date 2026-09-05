#!/usr/bin/env python3
"""Patient-axis line profiles through the exact isocenter, from saved 3D dose."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.ndimage import map_coordinates

REPO = Path(__file__).resolve().parents[1]
CASES = {
    "RT06423": Path("/mnt/sda/wuwei/topas10x_threecase_20260905_r2/RT06423"),
    "RT07575": Path("/mnt/sda/wuwei/topas10x_threecase_20260905_r2/RT07575"),
    "20022516": Path("/mnt/sda/wuwei/topas10x_threecase_20260905_r3/20022516"),
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_mhd(path):
    fields = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    fields = {k.strip(): v.strip() for k, v in fields.items()}
    if fields["ElementType"] != "MET_FLOAT" or fields["BinaryDataByteOrderMSB"] != "False":
        raise ValueError("Expected little-endian float32 dose")
    if fields.get("CompressedData", "False") != "False":
        raise ValueError("Compressed dose not supported")
    np.testing.assert_array_equal(np.fromstring(fields["TransformMatrix"], sep=" "), np.eye(3).ravel())
    dims = np.fromstring(fields["DimSize"], sep=" ", dtype=int)
    spacing = np.fromstring(fields["ElementSpacing"], sep=" ")
    origin = np.fromstring(fields["Offset"], sep=" ")
    raw = path.parent / fields["ElementDataFile"]
    dose = np.fromfile(raw, dtype="<f4").reshape(tuple(dims[::-1]))
    if not np.isfinite(dose).all() or np.any(dose < 0):
        raise ValueError("Invalid dose")
    return dose, origin, spacing, dims, raw


def isocenter_index(text, dims, spacing):
    def parameter(name, unit):
        match = re.search(rf"^d:Ge/Patient/{name}\s*=\s*([-+\d.eE]+)\s+{re.escape(unit)}\s*$", text, re.M)
        if match is None:
            raise ValueError(f"Missing or unsupported Patient/{name}")
        return float(match[1])

    translation = np.array([parameter("Trans" + a, "mm") for a in "XYZ"])
    if parameter("RotX", "deg") != 0 or parameter("RotY", "deg") != 0:
        raise ValueError("Only current axial patient rotations supported")
    angle = np.deg2rad(parameter("RotZ", "deg"))
    c, s = np.cos(angle), np.sin(angle)
    # TOPAS passive placement: world->patient is R(+RotZ) * (world - T).
    # PBSBeamFrame/IEC_G/IEC_F have zero placement, world isocenter = (0,0,0).
    local = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ -translation
    index = (np.asarray(dims) - 1) / 2 + local / spacing
    if np.any(index < 0) or np.any(index > dims - 1):
        raise ValueError("Isocenter outside voxel-center domain")
    return index, local


def extract_profile(dose, iso_xyz, spacing, axis):
    dims = np.array(dose.shape[::-1])
    indices = np.arange(dims[axis], dtype=float)
    xyz = np.repeat(np.asarray(iso_xyz, dtype=float)[:, None], len(indices), axis=1)
    xyz[axis] = indices
    values = map_coordinates(dose, xyz[::-1], order=1, mode="constant", cval=np.nan, prefilter=False)
    if not np.isfinite(values).all():
        raise ValueError("Profile requires extrapolation")
    return (indices - iso_xyz[axis]) * spacing[axis], values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "benchmark/topas10x/gpu_current_20260905_profiles")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refuse overwrite of existing profiles; choose new --output")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True)
    overview, panels = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    audit = {"method": "Exact-isocenter trilinear line extraction from full 3D dose; patient XYZ axes; native axis sampling; absolute cumulative Gy; no smoothing/scaling/registration; not IDD", "script_sha256": sha(Path(__file__)), "cases": {}}
    for row, (case, folder) in enumerate(CASES.items()):
        manifest = json.loads((folder / "manifest.json").read_text())
        gamma = json.loads((folder / "gamma.json").read_text())
        reference, origin, spacing, dims, ref_raw = read_mhd(folder / "topas_sum.mhd")
        gpu, go, gs, gd, gpu_raw = read_mhd(folder / "gpu_sum_patient.mhd")
        for a, b in ((origin, go), (spacing, gs), (dims, gd)):
            np.testing.assert_array_equal(a, b)
        if sha(ref_raw) != gamma["reference_sha256"] or sha(folder / "gpu_sum.raw") != gamma["gpu_sha256"]:
            raise ValueError("Dose SHA differs from frozen Gamma inputs")
        native = np.fromfile(folder / "gpu_sum.raw", dtype="<f4").reshape(manifest["gpu_shape_zyx"])
        if manifest["mapping"] == "packed_xneg":
            native = np.flip(native.transpose(1, 2, 0), axis=2)
        elif manifest["mapping"] != "native":
            raise ValueError("Unknown mapping")
        np.testing.assert_array_equal(native, gpu)
        parameter_file = Path(manifest["replicas"][0]["path"]) / "run_full_plan.txt"
        iso, local = isocenter_index(parameter_file.read_text(), dims, spacing)
        fig, axes = plt.subplots(2, 3, figsize=(16, 6), sharex="col", height_ratios=[3, 1], constrained_layout=True)
        fig.suptitle(f"{case}: full-statistics GPU / TOPAS, lines through isocenter")
        record = {"isocenter_patient_mm_xyz": (origin + iso * spacing).tolist(), "isocenter_index_xyz": iso.tolist(), "isocenter_centered_patient_mm_xyz": local.tolist(), "parameter_file": str(parameter_file), "parameter_sha256": sha(parameter_file), "reference_sha256": sha(ref_raw), "gpu_patient_sha256": sha(gpu_raw), "profiles": {}}
        for axis, label in enumerate("XYZ"):
            x, t = extract_profile(reference, iso, spacing, axis)
            _, g = extract_profile(gpu, iso, spacing, axis)
            residual = 100 * (g.astype(float) - t) / float(reference.max())
            csv = args.output / f"{case}_{label}.csv"
            np.savetxt(csv, np.column_stack((x, origin[axis] + iso[axis] * spacing[axis] + x, t, g, residual)), delimiter=",", header="distance_from_isocenter_mm,patient_coordinate_mm,topas_Gy,gpu_Gy,difference_pct_reference_3d_Dmax", comments="", fmt="%.10g")
            for ax in (axes[0, axis], panels[row, axis]):
                ax.plot(x, t, color="#202020", lw=1.5, label="TOPAS")
                ax.plot(x, g, color="#df6a1b", lw=1.3, ls="--", label="GPU")
                ax.axvline(0, color="#5289b5", lw=.7, alpha=.7)
                ax.set(title=f"{case} - patient {label}", ylabel="Dose (Gy)")
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            panels[row, axis].set_xlabel(f"{label} - isocenter (mm)")
            axes[1, axis].plot(x, residual, color="#307ba0", lw=1)
            axes[1, axis].axhline(0, color="gray", lw=.7)
            axes[1, axis].set(xlabel=f"{label} - isocenter (mm)", ylabel="G - T (% Dmax)")
            axes[1, axis].grid(alpha=.2)
            record["profiles"][label] = {"csv": csv.name, "sha256": sha(csv), "samples": len(x)}
        fig.savefig(args.output / f"{case}_isocenter_profiles.png", dpi=180)
        plt.close(fig)
        audit["cases"][case] = record
        print(case, "patient isocenter (mm):", record["isocenter_patient_mm_xyz"], flush=True)
    overview.suptitle("Isocenter line profiles | absolute cumulative Gy | full 20-shard GPU vs TOPAS")
    overview.savefig(args.output / "threecase_isocenter_profiles.png", dpi=180)
    plt.close(overview)
    (args.output / "profiles_manifest.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()

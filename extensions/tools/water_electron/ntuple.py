"""Read strict TOPAS V2/V3 electron genealogy records."""
import re
import numpy as np

V1 = ["run", "event", "track", "parent", "pdg", "step",
      "birth_x_mm", "birth_y_mm", "birth_z_mm", "birth_ke_MeV",
      "pre_x_mm", "pre_y_mm", "pre_z_mm", "post_x_mm", "post_y_mm",
      "post_z_mm", "edep_MeV", "pre_ke_MeV", "post_ke_MeV",
      "weight", "density_g_cm3"]
V2_EXTRA = ["schema_version", "parent_valid", "parent_ke_MeV",
            "parent_dir_x", "parent_dir_y", "parent_dir_z",
            "creator_process_id", "birth_density_g_cm3"]
COLUMNS = V1 + V2_EXTRA
INT_POS = set(range(6)) | {21, 22, 27}

def load_v2_binary(steps):
    """Compatibility name: explicit v2 and v3 headers, never schema aliasing."""
    header = steps.with_suffix(".header").read_text()
    actual = re.findall(r"^\s*([if]\d+):\s*(\S+)\s*$", header, re.MULTILINE)
    expected = [("i4" if i in INT_POS else "f8", n) for i, n in enumerate(COLUMNS)]
    columns, ints, version = COLUMNS, INT_POS, 2
    if actual != expected:
        columns = COLUMNS + ["parent_step_id", "parent_post_ke_MeV", "parent_post_dir_x", "parent_post_dir_y", "parent_post_dir_z"]
        ints, version = INT_POS | {29}, 3
        if actual != [("i4" if i in ints else "f8", n) for i,n in enumerate(columns)]:
            raise ValueError(f"{steps}: not a v2/v3 binary ntuple header")
    dtype = np.dtype([(n, "<i4" if i in ints else "<f8") for i, n in enumerate(columns)])
    entries = int(re.search(r"Number of Scored Entries: (\d+)", header)[1])
    if steps.stat().st_size != entries * dtype.itemsize:
        raise ValueError(f"{steps}: binary byte count mismatch")
    raw = np.memmap(steps, dtype=dtype, mode="r")
    d = np.column_stack([raw[n] for n in columns])
    del raw
    if len(d) != entries or d.shape[1] != len(columns) or not np.isfinite(d).all():
        raise ValueError(f"{steps}: truncated or non-finite ntuple")
    if np.any(d[:, 21] != version):
        raise ValueError(f"{steps}: schema_version != {version}")
    return d

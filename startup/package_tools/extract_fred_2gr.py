#!/usr/bin/env python3
"""Extract the six FRED 3.76 2GR tables into MAIGO's M2GR v1 package."""
from __future__ import annotations
import argparse, hashlib, json, struct
from pathlib import Path

ORDER = ["w1_2GR.txt", "sigma_c_2GR.txt", "w2_2GR.txt",
         "sigma_t_2GR.txt", "b_2GR.txt", "m_2GR.txt"]

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    blob = args.archive.read_bytes(); offset = 0
    count, = struct.unpack_from("<I", blob, offset); offset += 4
    names = []
    for _ in range(count):
        length, = struct.unpack_from("<I", blob, offset); offset += 4
        names.append(blob[offset:offset + length].decode()); offset += length
    sizes = list(struct.unpack_from(f"<{count}I", blob, offset)); offset += 4 * count
    offsets = list(struct.unpack_from(f"<{count}I", blob, offset)); offset += 4 * count
    checks = list(struct.unpack_from(f"<{count}I", blob, offset)); offset += 4 * count
    digests = []
    for _ in range(count):
        digests.append(blob[offset:offset + 32].split(b"\0", 1)[0].decode())
        offset += 32
    values, resources = [], []
    for short_name in ORDER:
        name = "data/mcs/" + short_name; index = names.index(name)
        raw = blob[offsets[index]:offsets[index] + sizes[index]]
        rows = [[float(value) for value in line.split()]
                for line in raw.decode().splitlines() if line.strip()]
        if len(rows) != 51 or any(len(row) != 48 for row in rows):
            raise RuntimeError(f"unexpected 2GR dimensions in {name}")
        values.extend(value for row in rows for value in row)
        resources.append({"resource": name, "size": sizes[index],
                          "offset": offsets[index], "archive_index": index,
                          "index_checksum_u32": checks[index],
                          "toc_digest": digests[index],
                          "resource_sha256": hashlib.sha256(raw).hexdigest()})
    payload = struct.pack("<4sIIII", b"M2GR", 1, 48, 51, 6)
    payload += struct.pack(f"<{len(values)}f", *values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    metadata = {"format": "M2GR", "version": 1,
                "source": "FRED 3.76.0 libFred.data",
                "source_sha256": hashlib.sha256(blob).hexdigest(),
                "energy_bins": 48, "log_areal_density_bins": 51,
                "parameter_order": ["w1", "sigma_c", "w2", "sigma_t", "b", "m"],
                "energy_coordinate": "iT=(T_MeV_per_u-1)/5",
                "areal_density_coordinate": "ilogs=(log10(rhodz_g_per_cm2)+4)/0.1",
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "resources": resources}
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")

if __name__ == "__main__": main()

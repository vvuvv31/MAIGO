#!/usr/bin/env python3
"""CINEL03 elemental-target package reader, writer, and validator.

Key contract:
event key = projectile_Z, projectile_A, target_element_Z, energy_node
material section is NOT part of the event key.
"""

from __future__ import annotations

import binascii
import hashlib
import json
import math
import struct
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Tuple, Dict

PACKAGE_MAGIC = b"CINPKG04"
PACKAGE_VERSION = 4
PACKAGE_HEADER = struct.Struct("<8sIIIIIIIQQQQffQQ40sI")
PACKAGE_INDEX = struct.Struct("<hhhhIQQff")
PACKAGE_ENERGY_NODE = struct.Struct("<hhhhf")
PACKAGE_ENERGY_OFFSET = struct.Struct("<Q")
PACKAGE_EVENT_INDEX = struct.Struct("<Q")
PACKAGE_HEADER_SIZE = PACKAGE_HEADER.size

RAW_FIXED_FORMAT = struct.Struct(
    "<QIQIII"       # run, thread, event, track, parent, sequence
    "ihhfff"        # projectile PDG/Z/A/charge/rest mass/excitation
    "ff"            # collision energy and energy/u
    "fff"           # collision position
    "fff"           # collision direction
    "ffff"          # global time, proper time, weight, step length
    "ii"            # material and cuts-couple ids
    "hhI"           # target_element_Z/target_A/isotope id
    "iii"           # process type/subtype/model id
    "iihh"          # parent status/PDG/Z/A
    + "f" * 13      # parent charge, mass, excitation, E, dir, pos, time, proper time, weight
    + "ff"          # process-local and non-ionizing deposits
    + "IIf"         # direct count, unsupported count, unsupported energy ledger
    + "f" * 10      # audit: pre E, pre dir, pre pos, whole-step Edep, source E, depth
    + "64s32s64s64s"# material, isotope, process, model names
)
PRODUCT_FORMAT = struct.Struct("<ihh" + "f" * 15 + "i")

FLAG_AUTHORITATIVE = 1 << 0
FLAG_TARGET_KNOWN = 1 << 1
FLAG_PROCESS_LOCAL_DEPOSIT = 1 << 2
FLAG_PARENT_FINAL_STATE = 1 << 3
FLAG_GLOBAL_ENERGY_INDEX = 1 << 4
FLAG_CRC32_CHECKSUM = 1 << 5

PRODUCT_DIRECT_SECONDARY = 0
PRODUCT_PARENT_CONTINUATION = 1
PRODUCT_UNSUPPORTED_BUT_RECORDED = 2

# Target Element Registry for Schneider CT Materials
SCHNEIDER_TARGET_ELEMENTS = {1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20, 22}

def is_valid_elemental_target(z: int) -> bool:
    return 1 <= z <= 100

def compute_crc32(data: bytes, prev_crc: int = 0) -> int:
    return binascii.crc32(data, prev_crc) & 0xFFFFFFFF

class Cinel03Package:
    def __init__(self):
        self.minimum_energy_MeV_per_u = 0.0
        self.energy_bin_width_MeV_per_u = 1.0
        self.minimum_events_per_bin = 1
        self.campaign_uuid = str(uuid.uuid4())
        self.cells: List[Dict[str, Any]] = []
        self.interactions: List[bytes] = []
        self.products: List[bytes] = []
        self.energy_nodes: List[Tuple[int, int, int, float]] = []
        self.event_offsets: List[int] = []
        self.event_indices: List[int] = []

    @classmethod
    def read_binary(cls, path: Path | str) -> Cinel03Package:
        path = Path(path)
        data = path.read_bytes()
        if len(data) < PACKAGE_HEADER_SIZE:
            raise ValueError(f"Truncated CINPKG04 file: {len(data)} < {PACKAGE_HEADER_SIZE}")

        magic, ver, hdr_sz, endian, idx_sz, int_sz, prd_sz, flags, \
        n_cells, n_ints, n_prods, file_sz, e_min, e_w, min_ev, n_nodes, uuid_bytes, crc = \
            PACKAGE_HEADER.unpack_from(data, 0)

        if magic != PACKAGE_MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}, expected {PACKAGE_MAGIC!r}")
        if ver != PACKAGE_VERSION:
            raise ValueError(f"Invalid version: {ver}, expected {PACKAGE_VERSION}")
        if endian != 0x01020304:
            raise ValueError(f"Invalid endian marker: 0x{endian:08x}")
        if idx_sz != PACKAGE_INDEX.size:
            raise ValueError(f"Index record size mismatch: {idx_sz} != {PACKAGE_INDEX.size}")
        if int_sz != RAW_FIXED_FORMAT.size:
            raise ValueError(f"Interaction record size mismatch: {int_sz} != {RAW_FIXED_FORMAT.size}")
        if prd_sz != PRODUCT_FORMAT.size:
            raise ValueError(f"Product record size mismatch: {prd_sz} != {PRODUCT_FORMAT.size}")
        if file_sz != len(data):
            raise ValueError(f"File size mismatch: {file_sz} != {len(data)}")

        # Verify CRC32
        if flags & FLAG_CRC32_CHECKSUM:
            payload = data[PACKAGE_HEADER_SIZE:]
            calc_crc = compute_crc32(payload)
            if crc != calc_crc:
                raise ValueError(f"CINPKG04 CRC32 mismatch: 0x{crc:08x} vs 0x{calc_crc:08x}")

        pkg = cls()
        pkg.minimum_energy_MeV_per_u = e_min
        pkg.energy_bin_width_MeV_per_u = e_w
        pkg.minimum_events_per_bin = min_ev
        pkg.campaign_uuid = uuid_bytes.rstrip(b"\0").decode("ascii")

        offset = PACKAGE_HEADER_SIZE
        # Cells
        for _ in range(n_cells):
            pz, pa, tz, _, e_bin, int_off, int_cnt, e_low, e_up = PACKAGE_INDEX.unpack_from(data, offset)
            pkg.cells.append({
                "projectile_z": pz, "projectile_a": pa, "target_element_z": tz,
                "energy_bin": e_bin, "interaction_offset": int_off, "interaction_count": int_cnt,
                "energy_lower_MeV_per_u": e_low, "energy_upper_MeV_per_u": e_up
            })
            offset += PACKAGE_INDEX.size

        # Interactions
        for _ in range(n_ints):
            pkg.interactions.append(data[offset:offset + RAW_FIXED_FORMAT.size])
            offset += RAW_FIXED_FORMAT.size

        # Products
        for _ in range(n_prods):
            pkg.products.append(data[offset:offset + PRODUCT_FORMAT.size])
            offset += PRODUCT_FORMAT.size

        # Energy nodes
        for _ in range(n_nodes):
            pz, pa, tz, _, e_node = PACKAGE_ENERGY_NODE.unpack_from(data, offset)
            pkg.energy_nodes.append((pz, pa, tz, e_node))
            offset += PACKAGE_ENERGY_NODE.size

        # Event offsets
        for _ in range(n_nodes + 1):
            pkg.event_offsets.append(PACKAGE_ENERGY_OFFSET.unpack_from(data, offset)[0])
            offset += PACKAGE_ENERGY_OFFSET.size

        # Event indices
        for _ in range(n_ints):
            pkg.event_indices.append(PACKAGE_EVENT_INDEX.unpack_from(data, offset)[0])
            offset += PACKAGE_EVENT_INDEX.size

        if offset != len(data):
            raise ValueError(f"Trailing bytes in CINPKG04: {offset} != {len(data)}")

        return pkg

    def write_binary(self, path: Path | str) -> None:
        path = Path(path)
        payload_parts = []

        # Cells
        for c in self.cells:
            payload_parts.append(PACKAGE_INDEX.pack(
                c["projectile_z"], c["projectile_a"], c["target_element_z"], 0,
                c["energy_bin"], c["interaction_offset"], c["interaction_count"],
                c["energy_lower_MeV_per_u"], c["energy_upper_MeV_per_u"]
            ))

        # Interactions
        payload_parts.extend(self.interactions)

        # Products
        payload_parts.extend(self.products)

        # Energy nodes
        for node in self.energy_nodes:
            payload_parts.append(PACKAGE_ENERGY_NODE.pack(node[0], node[1], node[2], 0, node[3]))

        # Event offsets
        for off in self.event_offsets:
            payload_parts.append(PACKAGE_ENERGY_OFFSET.pack(off))

        # Event indices
        for idx in self.event_indices:
            payload_parts.append(PACKAGE_EVENT_INDEX.pack(idx))

        payload_bytes = b"".join(payload_parts)
        payload_crc = compute_crc32(payload_bytes)

        file_sz = PACKAGE_HEADER_SIZE + len(payload_bytes)
        flags = FLAG_AUTHORITATIVE | FLAG_TARGET_KNOWN | FLAG_PROCESS_LOCAL_DEPOSIT | \
                FLAG_PARENT_FINAL_STATE | FLAG_GLOBAL_ENERGY_INDEX | FLAG_CRC32_CHECKSUM

        header = PACKAGE_HEADER.pack(
            PACKAGE_MAGIC,
            PACKAGE_VERSION,
            PACKAGE_HEADER_SIZE,
            0x01020304,
            PACKAGE_INDEX.size,
            RAW_FIXED_FORMAT.size,
            PRODUCT_FORMAT.size,
            flags,
            len(self.cells),
            len(self.interactions),
            len(self.products),
            file_sz,
            self.minimum_energy_MeV_per_u,
            self.energy_bin_width_MeV_per_u,
            self.minimum_events_per_bin,
            len(self.energy_nodes),
            self.campaign_uuid.encode("ascii"),
            payload_crc
        )

        path.write_bytes(header + payload_bytes)

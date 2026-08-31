#!/usr/bin/env python3
"""CINEL02 raw reader, validator, and correlated package compiler.

The module is intentionally dependency-free.  It is shared by the command
line tools and the synthetic contract tests so the bytes written by the TOPAS
extension have one auditable interpretation.
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
from typing import Any, Iterable, Iterator


RAW_MAGIC = b"CINEL02\0"
RAW_VERSION = 2
RAW_HEADER = struct.Struct("<8sIIIIQQQQQ")
RAW_HEADER_SIZE = RAW_HEADER.size
RECORD_MAGIC = b"CIR2"
RECORD_PREFIX = struct.Struct("<4sHHII")
RECORD_VERSION = 2

# Keep this format in lock-step with CarbonInelasticEventWriter.hh.  The
# payload is packed so its representation does not depend on host alignment.
RAW_FIXED_FORMAT = struct.Struct(
    "<QIQIII"       # run, thread, event, track, parent, sequence
    "ihhfff"        # projectile PDG/Z/A/charge/rest mass/excitation
    "ff"            # collision energy and energy/u
    "fff"           # collision position
    "fff"           # collision direction
    "ffff"          # global time, proper time, weight, step length
    "ii"            # material and cuts-couple ids
    "hhI"            # target Z/A/isotope id
    "iii"            # process type/subtype/model id
    "iihh"          # parent status/PDG/Z/A
    + "f" * 13       # parent charge, mass, excitation, E, dir, pos, time, proper time, weight
    + "ff"          # process-local and non-ionizing deposits
    + "IIf"          # direct count, unsupported count, unsupported energy ledger
    + "f" * 10       # audit: pre E, pre dir, pre pos, whole-step Edep, source E, depth
    + "64s32s64s64s" # material, isotope, process, model names
)
# PDG/Z/A + charge/mass/excitation/E + global dir + projectile-local dir
# + position + time/weight + role.  Keep this at 72 bytes: the local frame is
# part of the authoritative correlated-event contract and is not optional.
PRODUCT_FORMAT = struct.Struct("<ihh" + "f" * 15 + "i")
CRC_FORMAT = struct.Struct("<I")

PACKAGE_MAGIC = b"CINPKG03"
PACKAGE_VERSION = 3
PACKAGE_HEADER = struct.Struct("<8sIIIIIIIQQQQffQQ40s")
PACKAGE_INDEX = struct.Struct("<hhhhIQQff")
PACKAGE_ENERGY_NODE = struct.Struct("<hhhhf")
PACKAGE_ENERGY_OFFSET = struct.Struct("<Q")
PACKAGE_EVENT_INDEX = struct.Struct("<Q")
PACKAGE_HEADER_SIZE = PACKAGE_HEADER.size
DEFAULT_MINIMUM_EVENTS_PER_BIN = 32
DEFAULT_MAX_UNSUPPORTED_PRODUCT_ENERGY_FRACTION = 1.0e-4
QUALIFICATION_SCHEMA = "CINEL02_QUALIFICATION_V1"

FLAG_AUTHORITATIVE = 1 << 0
FLAG_TARGET_KNOWN = 1 << 1
FLAG_PROCESS_LOCAL_DEPOSIT = 1 << 2
FLAG_PARENT_FINAL_STATE = 1 << 3
FLAG_GLOBAL_ENERGY_INDEX = 1 << 4

PRODUCT_DIRECT_SECONDARY = 0
PRODUCT_PARENT_CONTINUATION = 1
PRODUCT_UNSUPPORTED_BUT_RECORDED = 2

_FIELD_NAMES = (
    "run_id", "thread_id", "event_id", "track_id", "parent_track_id",
    "interaction_sequence", "projectile_pdg", "projectile_z", "projectile_a",
    "projectile_charge", "projectile_rest_mass", "projectile_excitation",
    "collision_energy_MeV", "collision_energy_MeV_per_u", "collision_x_mm",
    "collision_y_mm", "collision_z_mm", "collision_direction_x",
    "collision_direction_y", "collision_direction_z", "collision_time_ns",
    "proper_time_ns", "track_weight", "step_length_mm", "material_id",
    "cuts_couple_id", "target_z", "target_a", "target_isotope_id",
    "process_type", "process_subtype", "model_id", "parent_status",
    "parent_pdg", "parent_z", "parent_a", "parent_charge", "parent_rest_mass",
    "parent_excitation", "parent_energy_MeV", "parent_direction_x",
    "parent_direction_y", "parent_direction_z", "parent_x_mm", "parent_y_mm",
    "parent_z_mm", "parent_time_ns", "parent_proper_time_ns", "parent_weight",
    "process_local_deposit_MeV", "nonionizing_deposit_MeV", "direct_product_count",
    "unsupported_product_count", "unsupported_product_energy_MeV",
    "audit_pre_energy_MeV", "audit_pre_direction_x", "audit_pre_direction_y",
    "audit_pre_direction_z", "audit_pre_x_mm", "audit_pre_y_mm", "audit_pre_z_mm",
    "audit_whole_step_deposit_MeV", "source_initial_energy_MeV", "absolute_depth_mm",
    "material_name", "target_isotope_name", "process_name", "model_name",
)

_PRODUCT_NAMES = (
    "pdg", "z", "a", "charge", "rest_mass", "excitation", "kinetic_energy_MeV",
    "direction_x", "direction_y", "direction_z", "local_direction_x",
    "local_direction_y", "local_direction_z", "position_x_mm", "position_y_mm",
    "position_z_mm", "creation_time_ns", "weight", "role",
)


def _decode_fixed(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8", errors="strict")


def _encode_fixed(value: str, size: int, label: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) >= size:
        raise ValueError(f"{label} is longer than {size - 1} bytes")
    return encoded + b"\0" * (size - len(encoded))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _unit_direction(values: Iterable[Any], label: str) -> tuple[float, float, float]:
    direction = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in direction):
        raise ValueError(f"{label} contains a non-finite component")
    norm = math.sqrt(sum(value * value for value in direction))
    if norm <= 0.0 or abs(norm - 1.0) > 2.0e-3:
        raise ValueError(f"{label} is not a unit direction: {direction}")
    return tuple(value / norm for value in direction)


def _rotate_local_direction(
    local_direction: Iterable[Any], incident_direction: Iterable[Any]
) -> tuple[float, float, float]:
    """Reconstruct a global direction using the device transport frame.

    This is intentionally the same deterministic right-handed basis as
    ``rotate_local_direction`` in ``src/detail/sycl_device_math.inc`` and the
    TOPAS writer.  Keeping the check here makes a captured package reject a
    global/local direction pair that would be silently rotated on replay.
    """

    axis = _unit_direction(incident_direction, "collision direction")
    reference = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0)
    projection = sum(reference[index] * axis[index] for index in range(3))
    local_x_raw = tuple(reference[index] - projection * axis[index] for index in range(3))
    local_x_norm = math.sqrt(sum(value * value for value in local_x_raw))
    if local_x_norm <= 0.0 or not math.isfinite(local_x_norm):
        raise ValueError("projectile-local transverse basis is degenerate")
    local_x = tuple(value / local_x_norm for value in local_x_raw)
    local_y = (
        axis[1] * local_x[2] - axis[2] * local_x[1],
        axis[2] * local_x[0] - axis[0] * local_x[2],
        axis[0] * local_x[1] - axis[1] * local_x[0],
    )
    local_y = _unit_direction(local_y, "projectile-local transverse basis")
    local = _unit_direction(local_direction, "product projectile-local direction")
    output = tuple(
        local[0] * local_x[index]
        + local[1] * local_y[index]
        + local[2] * axis[index]
        for index in range(3)
    )
    return _unit_direction(output, "reconstructed product direction")


def _unpack_fixed(payload: bytes) -> dict[str, Any]:
    if len(payload) < RAW_FIXED_FORMAT.size:
        raise ValueError("CINEL02 record has a truncated interaction payload")
    values = list(RAW_FIXED_FORMAT.unpack_from(payload))
    result = dict(zip(_FIELD_NAMES, values))
    result["material_name"] = _decode_fixed(result["material_name"])
    result["target_isotope_name"] = _decode_fixed(result["target_isotope_name"])
    result["process_name"] = _decode_fixed(result["process_name"])
    result["model_name"] = _decode_fixed(result["model_name"])
    return result


def _pack_fixed(record: dict[str, Any]) -> bytes:
    values = [record[name] for name in _FIELD_NAMES]
    values[-4:] = [
        _encode_fixed(record["material_name"], 64, "material_name"),
        _encode_fixed(record["target_isotope_name"], 32, "target_isotope_name"),
        _encode_fixed(record["process_name"], 64, "process_name"),
        _encode_fixed(record["model_name"], 64, "model_name"),
    ]
    return RAW_FIXED_FORMAT.pack(*values)


def _unpack_product(payload: bytes, offset: int) -> dict[str, Any]:
    values = PRODUCT_FORMAT.unpack_from(payload, offset)
    return dict(zip(_PRODUCT_NAMES, values))


def _pack_product(product: dict[str, Any]) -> bytes:
    return PRODUCT_FORMAT.pack(*(product[name] for name in _PRODUCT_NAMES))


def _normalise_product(product: dict[str, Any]) -> dict[str, Any]:
    result = dict(product)
    result.setdefault("role", PRODUCT_DIRECT_SECONDARY)
    # Older synthetic fixtures may only provide the global direction.  The
    # authoritative writer always supplies both frames; defaulting here keeps
    # the helper convenient without weakening validation of the resulting
    # unit-vector contract.
    result.setdefault("local_direction_x", result.get("direction_x", 0.0))
    result.setdefault("local_direction_y", result.get("direction_y", 0.0))
    result.setdefault("local_direction_z", result.get("direction_z", 1.0))
    for key in ("pdg", "z", "a", "role"):
        result[key] = int(result[key])
    for key in (
        "charge", "rest_mass", "excitation", "kinetic_energy_MeV", "direction_x",
        "direction_y", "direction_z", "local_direction_x", "local_direction_y",
        "local_direction_z", "position_x_mm", "position_y_mm",
        "position_z_mm", "creation_time_ns", "weight",
    ):
        result[key] = float(result[key])
    return result


_INELASTIC_NUCLEON_REST_MASS_MEV = 931.49410242
_INELASTIC_PROTON_REST_MASS_MEV = 938.27208816
_INELASTIC_NEUTRON_REST_MASS_MEV = 939.56542052
_INELASTIC_ELECTRON_REST_MASS_MEV = 0.51099895
_INELASTIC_CHARGED_PION_REST_MASS_MEV = 139.57039
_INELASTIC_NEUTRAL_PION_REST_MASS_MEV = 134.9768
_INELASTIC_ETA_MESON_REST_MASS_MEV = 547.862


def _identity_close(
    value: Any,
    expected: float,
    *,
    relative_tolerance: float = 1.0e-3,
    absolute_tolerance: float = 1.0e-3,
) -> bool:
    if not _finite(value) or not math.isfinite(expected):
        return False
    return abs(float(value) - expected) <= max(
        absolute_tolerance, relative_tolerance * max(1.0, abs(expected))
    )


def _ground_state_ion_pdg(atomic_number: int, mass_number: int) -> int:
    if atomic_number <= 0 or mass_number < atomic_number:
        return 0
    return 1_000_000_000 + 10_000 * atomic_number + 10 * mass_number


def _ion_mass_is_explainable(mass_number: int, rest_mass: Any) -> bool:
    if mass_number <= 0 or not _finite(rest_mass) or float(rest_mass) <= 0.0:
        return False
    expected = mass_number * _INELASTIC_NUCLEON_REST_MASS_MEV
    tolerance = max(25.0, 0.02 * expected)
    return abs(float(rest_mass) - expected) <= tolerance


def _classify_inelastic_identity(
    pdg: int,
    atomic_number: int,
    mass_number: int,
    charge: Any,
    rest_mass: Any,
    excitation: Any,
) -> str | None:
    if (
        pdg == 0
        or not _finite(charge)
        or not _finite(rest_mass)
        or float(rest_mass) < 0.0
        or not _finite(excitation)
        or float(excitation) < 0.0
        or float(excitation) > 1.0e-4
    ):
        return None
    if pdg == 22:
        return (
            "gamma"
            if atomic_number == 0
            and mass_number == 0
            and _identity_close(charge, 0.0, relative_tolerance=0.0)
            and _identity_close(rest_mass, 0.0, relative_tolerance=0.0)
            else None
        )
    if pdg == 2112:
        return (
            "neutron"
            if atomic_number == 0
            and mass_number == 1
            and _identity_close(charge, 0.0, relative_tolerance=0.0)
            and _identity_close(rest_mass, _INELASTIC_NEUTRON_REST_MASS_MEV)
            else None
        )
    if pdg == 2212:
        return (
            "ion"
            if atomic_number == 1
            and mass_number == 1
            and _identity_close(charge, 1.0, relative_tolerance=0.0)
            and _identity_close(rest_mass, _INELASTIC_PROTON_REST_MASS_MEV)
            else None
        )
    if pdg in (11, -11):
        expected_charge = -1.0 if pdg == 11 else 1.0
        return (
            ("electron" if pdg == 11 else "positron")
            if atomic_number == 0
            and mass_number == 0
            and _identity_close(charge, expected_charge, relative_tolerance=0.0)
            and _identity_close(rest_mass, _INELASTIC_ELECTRON_REST_MASS_MEV)
            else None
        )
    if pdg in (211, -211):
        expected_charge = 1.0 if pdg == 211 else -1.0
        return (
            "charged_pion"
            if atomic_number == 0
            and mass_number == 0
            and _identity_close(charge, expected_charge, relative_tolerance=0.0)
            and _identity_close(rest_mass, _INELASTIC_CHARGED_PION_REST_MASS_MEV)
            else None
        )
    if pdg == 111:
        return (
            "neutral_pion"
            if atomic_number == 0
            and mass_number == 0
            and _identity_close(charge, 0.0, relative_tolerance=0.0)
            and _identity_close(rest_mass, _INELASTIC_NEUTRAL_PION_REST_MASS_MEV)
            else None
        )
    if pdg == 221:
        return (
            "eta_meson"
            if atomic_number == 0
            and mass_number == 0
            and _identity_close(charge, 0.0, relative_tolerance=0.0)
            and _identity_close(rest_mass, _INELASTIC_ETA_MESON_REST_MASS_MEV)
            else None
        )
    if (
        atomic_number <= 0
        or mass_number < atomic_number
        or pdg != _ground_state_ion_pdg(atomic_number, mass_number)
        or not _identity_close(
            charge, float(atomic_number), relative_tolerance=0.0
        )
        or not _ion_mass_is_explainable(mass_number, rest_mass)
    ):
        return None
    return "ion"


def pack_record(record: dict[str, Any], products: list[dict[str, Any]]) -> bytes:
    fixed = _pack_fixed(record)
    product_bytes = b"".join(_pack_product(_normalise_product(product)) for product in products)
    payload = fixed + product_bytes
    length = RECORD_PREFIX.size + len(payload) + CRC_FORMAT.size
    prefix = RECORD_PREFIX.pack(RECORD_MAGIC, RECORD_VERSION, 0, length, len(payload))
    crc = CRC_FORMAT.pack(binascii.crc32(payload) & 0xFFFFFFFF)
    return prefix + payload + crc


def read_raw(path: Path, *, allow_truncated_tail: bool = False) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    data = path.read_bytes()
    if len(data) < RAW_HEADER_SIZE:
        raise ValueError(f"CINEL02 raw file is shorter than its header: {path}")
    header = RAW_HEADER.unpack_from(data)
    magic, version, header_size, endian, flags, interaction_count, product_count, bytes_written, _, _ = header
    if magic != RAW_MAGIC or version != RAW_VERSION or header_size != RAW_HEADER_SIZE:
        raise ValueError(f"Unsupported CINEL02 raw header: {path}")
    if endian != 0x01020304:
        raise ValueError(f"CINEL02 raw file is not little-endian: {path}")
    if bytes_written not in (0, len(data)):
        raise ValueError(f"CINEL02 raw file-size field does not match: {path}")

    records: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    offset = RAW_HEADER_SIZE
    while offset < len(data):
        start = offset
        if len(data) - offset < RECORD_PREFIX.size:
            if allow_truncated_tail:
                break
            raise ValueError(f"truncated CINEL02 record prefix at byte {offset}: {path}")
        record_magic, record_version, _, record_length, payload_length = RECORD_PREFIX.unpack_from(data, offset)
        if record_magic != RECORD_MAGIC or record_version != RECORD_VERSION:
            raise ValueError(f"invalid CINEL02 record prefix at byte {offset}: {path}")
        if record_length < RECORD_PREFIX.size + CRC_FORMAT.size or payload_length != record_length - RECORD_PREFIX.size - CRC_FORMAT.size:
            raise ValueError(f"invalid CINEL02 record length at byte {offset}: {path}")
        end = offset + record_length
        if end > len(data):
            if allow_truncated_tail:
                break
            raise ValueError(f"truncated CINEL02 record at byte {offset}: {path}")
        payload_start = offset + RECORD_PREFIX.size
        payload_end = payload_start + payload_length
        payload = data[payload_start:payload_end]
        expected_crc = CRC_FORMAT.unpack_from(data, payload_end)[0]
        actual_crc = binascii.crc32(payload) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise ValueError(f"CINEL02 CRC mismatch at byte {offset}: {path}")
        record = _unpack_fixed(payload)
        product_start = RAW_FIXED_FORMAT.size
        product_bytes = payload[product_start:]
        expected_product_bytes = int(record["direct_product_count"]) * PRODUCT_FORMAT.size
        if len(product_bytes) != expected_product_bytes:
            raise ValueError(f"CINEL02 product count/record length mismatch at byte {offset}: {path}")
        products = [
            _unpack_product(product_bytes, index * PRODUCT_FORMAT.size)
            for index in range(int(record["direct_product_count"]))
        ]
        records.append((record, products))
        offset = end
        if offset <= start:
            raise ValueError(f"CINEL02 reader made no progress at byte {start}: {path}")
    if offset != len(data) and not allow_truncated_tail:
        raise ValueError(f"CINEL02 raw reader did not consume file: {path}")
    if interaction_count not in (0, len(records)):
        raise ValueError(f"CINEL02 interaction-count mismatch in {path}")
    actual_products = sum(len(products) for _, products in records)
    if product_count not in (0, actual_products):
        raise ValueError(f"CINEL02 product-count mismatch in {path}")
    return records


def _metadata_contract(metadata: dict[str, Any], *, single_target: tuple[int, int] | None) -> None:
    schema = metadata.get("schema")
    if not isinstance(schema, dict) or schema.get("name") != "CINEL02" or int(schema.get("version", -1)) != RAW_VERSION:
        raise ValueError("metadata must declare schema CINEL02 version 2")
    collision = metadata.get("collision_state_source")
    if not isinstance(collision, dict) or collision.get("authoritative") is not True or collision.get("object") != "G4Track passed to wrapped PostStepDoIt":
        raise ValueError("metadata does not declare the authoritative PostStepDoIt input")
    if collision.get("pre_step_authoritative") is True:
        raise ValueError("pre-step state cannot be authoritative")
    final_state = metadata.get("final_state_source")
    if not isinstance(final_state, dict) or final_state.get("object") != "G4VParticleChange returned by the same delegated PostStepDoIt" or final_state.get("delayed_secondary_reconstruction") is True:
        raise ValueError("metadata does not declare same-call ParticleChange final state")
    local = metadata.get("local_deposit")
    if not isinstance(local, dict) or local.get("runtime_safe") is not True or local.get("definition") != "G4VParticleChange::GetLocalEnergyDeposit":
        raise ValueError("metadata must use process-local ParticleChange deposit")
    forbidden = {
        "depth_conditioned_production": metadata.get("depth_conditioned_production"),
        "scalar_energy_scaling": metadata.get("scalar_energy_scaling"),
        "nearest_fill": metadata.get("nearest_fill"),
        "first_step_products": metadata.get("first_step_products"),
        "whole_step_local_deposit": metadata.get("whole_step_local_deposit"),
    }
    bad = [name for name, value in forbidden.items() if value is True]
    if bad:
        raise ValueError("forbidden CINEL02 production flags: " + ", ".join(bad))
    campaign = metadata.get("campaign", {})
    declared_target = campaign.get("single_target") if isinstance(campaign, dict) else None
    if single_target is None and isinstance(declared_target, dict):
        single_target = (int(declared_target.get("Z", 0)), int(declared_target.get("A", 0)))
    if single_target is not None and (single_target[0] <= 0 or single_target[1] < single_target[0]):
        raise ValueError("single-target campaign has invalid Z/A")


def _canonical_campaign_uuid(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("CINPKG03 metadata must declare campaign.uuid")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise ValueError(f"invalid CINPKG03 campaign UUID: {value!r}") from error
    canonical = str(parsed)
    if value.lower() != canonical:
        raise ValueError("CINPKG03 campaign UUID must use canonical hyphenated form")
    return canonical


def _campaign_uuid(metadata: dict[str, Any]) -> str:
    campaign = metadata.get("campaign")
    candidates: list[Any] = []
    if isinstance(campaign, dict):
        candidates.extend(campaign.get(key) for key in ("uuid", "campaign_uuid"))
    candidates.append(metadata.get("campaign_uuid"))
    provenance = metadata.get("provenance")
    if isinstance(provenance, dict):
        candidates.append(provenance.get("campaign_uuid"))
    for candidate in candidates:
        if candidate not in (None, ""):
            return _canonical_campaign_uuid(candidate)
    raise ValueError("CINPKG03 metadata must declare campaign.uuid")


def _supported_product_identity(product: dict[str, Any]) -> bool:
    return _classify_inelastic_identity(
        product["pdg"], product["z"], product["a"], product["charge"],
        product["rest_mass"], product["excitation"],
    ) in ("ion", "gamma", "neutron")



def _ground_state_isomer_pdg(
    pdg: int, z: int, a: int, charge: float, rest_mass: float,
    excitation: float,
) -> int | None:
    """Return the same-Z/A ground-state PDG for a valid nuclear isomer."""
    if z <= 0 or a < z:
        return None
    ground_pdg = _ground_state_ion_pdg(z, a)
    if not (ground_pdg < int(pdg) <= ground_pdg + 9):
        return None
    if not _finite(excitation) or float(excitation) <= 1.0e-4:
        return None
    if not _identity_close(float(charge), float(z), relative_tolerance=0.0):
        return None
    if not _ion_mass_is_explainable(a, float(rest_mass)):
        return None
    return ground_pdg
def _ground_state_isomer_product(product: dict[str, Any]) -> dict[str, Any] | None:
    """Return a transport-equivalent ground-state ion for a ledgered isomer."""
    result = _normalise_product(product)
    z, a, pdg = result["z"], result["a"], result["pdg"]
    if result["role"] != PRODUCT_UNSUPPORTED_BUT_RECORDED or z <= 0 or a < z:
        return None
    ground_pdg = _ground_state_ion_pdg(z, a)
    if not (ground_pdg < pdg <= ground_pdg + 9):
        return None
    if not _finite(result["excitation"]) or result["excitation"] <= 1.0e-4:
        return None
    if not _identity_close(result["charge"], float(z), relative_tolerance=0.0):
        return None
    if not _ion_mass_is_explainable(a, result["rest_mass"]):
        return None
    result["pdg"] = ground_pdg
    result["excitation"] = 0.0
    result["role"] = PRODUCT_DIRECT_SECONDARY
    if not _supported_product_identity(result):
        raise ValueError("ground-state isomer normalisation produced an invalid ion")
    return result


def _validate_record(record: dict[str, Any], products: list[dict[str, Any]], *, single_target: tuple[int, int] | None) -> None:
    required_strings = ("material_name", "process_name", "model_name")
    if any(not record.get(name) for name in required_strings):
        raise ValueError("interaction is missing material/process/model identity")
    for name in (
        "collision_energy_MeV", "collision_energy_MeV_per_u", "process_local_deposit_MeV",
        "nonionizing_deposit_MeV", "unsupported_product_energy_MeV",
    ):
        if not _finite(record[name]) or float(record[name]) < 0.0:
            raise ValueError(f"invalid interaction value: {name}")
    if record["collision_energy_MeV_per_u"] <= 0.0:
        raise ValueError("collision energy per u must be positive")
    for name in ("track_weight", "parent_weight"):
        if not _finite(record[name]) or not math.isclose(
            float(record[name]), 1.0, abs_tol=1.0e-6
        ):
            raise ValueError("CINEL02 replay requires unit interaction weights")
    if int(record["parent_status"]) not in (0, 2):
        raise ValueError("CINPKG03 supports only alive (0) or stop-and-kill (2) parent status")
    if not _finite(record["parent_energy_MeV"]) or float(record["parent_energy_MeV"]) < 0.0:
        raise ValueError("invalid parent final energy")
    collision_energy = float(record["collision_energy_MeV"])
    parent_energy = float(record["parent_energy_MeV"])
    if parent_energy > collision_energy + 1.0e-3:
        raise ValueError("parent final energy exceeds captured collision energy")
    if int(record["parent_status"]) == 2 and parent_energy > 1.0e-4:
        raise ValueError("stop-and-kill parent retained non-zero kinetic energy")
    projectile_identity = (
        int(record["projectile_pdg"]),
        int(record["projectile_z"]),
        int(record["projectile_a"]),
    )
    parent_identity = (
        int(record["parent_pdg"]),
        int(record["parent_z"]),
        int(record["parent_a"]),
    )
    if projectile_identity != parent_identity:
        raise ValueError("CINPKG03 parent/projectile identity mismatch")
    projectile_kind = _classify_inelastic_identity(
        *projectile_identity,
        record["projectile_charge"], record["projectile_rest_mass"],
        record["projectile_excitation"],
    )
    parent_kind = _classify_inelastic_identity(
        *parent_identity,
        record["parent_charge"], record["parent_rest_mass"],
        record["parent_excitation"],
    )
    projectile_isomer_pdg = _ground_state_isomer_pdg(
        *projectile_identity, record["projectile_charge"],
        record["projectile_rest_mass"], record["projectile_excitation"],
    )
    parent_isomer_pdg = _ground_state_isomer_pdg(
        *parent_identity, record["parent_charge"],
        record["parent_rest_mass"], record["parent_excitation"],
    )
    if (
        (projectile_kind != "ion" and projectile_isomer_pdg is None)
        or (parent_kind != "ion" and parent_isomer_pdg is None)
    ):
        raise ValueError("CINPKG03 projectile/parent identity or state is unsupported")
    if int(record["parent_status"]) == 0:
        charge_tolerance = 1.0e-3 * max(1.0, abs(float(record["projectile_charge"])))
        mass_tolerance = 1.0e-3 * max(1.0, abs(float(record["projectile_rest_mass"])))
        if (
            abs(float(record["parent_charge"]) - float(record["projectile_charge"]))
            > charge_tolerance
            or abs(float(record["parent_rest_mass"]) - float(record["projectile_rest_mass"]))
            > mass_tolerance
        ):
            raise ValueError("CINPKG03 surviving parent changed projectile identity")
    collision_direction = _unit_direction(
        (record["collision_direction_x"], record["collision_direction_y"], record["collision_direction_z"]),
        "collision direction",
    )
    _unit_direction(
        (record["parent_direction_x"], record["parent_direction_y"], record["parent_direction_z"]),
        "parent final direction",
    )
    target = (int(record["target_z"]), int(record["target_a"]))
    if target[0] <= 0 or target[1] < target[0]:
        if single_target is None:
            raise ValueError("mixed-target interaction has unknown target isotope")
        record["target_z"], record["target_a"] = single_target
        target = single_target
    if int(record["direct_product_count"]) != len(products):
        raise ValueError("interaction direct product count does not match payload")
    unsupported_energy = 0.0
    unsupported_count = 0
    for product in products:
        product = _normalise_product(product)
        if not _finite(product["kinetic_energy_MeV"]) or product["kinetic_energy_MeV"] < 0.0:
            raise ValueError("invalid product kinetic energy")
        product_kind = _classify_inelastic_identity(
            product["pdg"], product["z"], product["a"], product["charge"],
            product["rest_mass"], product["excitation"],
        )
        if not _finite(product["weight"]) or not math.isclose(
            product["weight"], 1.0, abs_tol=1.0e-6
        ):
            raise ValueError("CINEL02 replay requires unit product weights")
        global_direction = _unit_direction(
            (product["direction_x"], product["direction_y"], product["direction_z"]),
            "product direction",
        )
        local_direction = _unit_direction(
            (product["local_direction_x"], product["local_direction_y"],
             product["local_direction_z"]),
            "product projectile-local direction",
        )
        reconstructed_direction = _rotate_local_direction(local_direction, collision_direction)
        if any(
            abs(actual - expected) > 3.0e-3
            for actual, expected in zip(global_direction, reconstructed_direction)
        ):
            raise ValueError("product global/local direction frames are inconsistent")
        if product["role"] not in (PRODUCT_DIRECT_SECONDARY, PRODUCT_PARENT_CONTINUATION, PRODUCT_UNSUPPORTED_BUT_RECORDED):
            raise ValueError("unknown product role")
        supported_identity = product_kind in ("ion", "gamma", "neutron")
        if product["role"] != PRODUCT_UNSUPPORTED_BUT_RECORDED and product_kind is None:
            raise ValueError(
                "transported product PDG/Z/A, charge, mass, or excitation is invalid")
        if product["role"] == PRODUCT_UNSUPPORTED_BUT_RECORDED:
            unsupported_count += 1
            unsupported_energy += product["kinetic_energy_MeV"]
        elif not supported_identity:
            raise ValueError("unsupported product is not explicitly ledgered")
    if unsupported_count != int(record["unsupported_product_count"]) or not math.isclose(
        unsupported_energy, float(record["unsupported_product_energy_MeV"]), rel_tol=2e-5, abs_tol=2e-5
    ):
        raise ValueError("unsupported product energy ledger does not close")


def validate_raw_files(paths: list[Path], metadata: dict[str, Any], *, single_target: tuple[int, int] | None = None) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    _metadata_contract(metadata, single_target=single_target)
    all_records: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    for path in paths:
        records = read_raw(path)
        for record, products in records:
            _validate_record(record, products, single_target=single_target)
            key = tuple(int(record[name]) for name in ("run_id", "thread_id", "event_id", "track_id", "interaction_sequence"))
            if key in seen:
                raise ValueError(f"duplicate CINEL02 interaction identity: {key}")
            seen.add(key)
            all_records.append((record, products))
    if not all_records:
        raise ValueError("CINEL02 input contains no complete interactions")
    return all_records


def _record_with_products(record: dict[str, Any], products: list[dict[str, Any]]) -> bytes:
    return _pack_fixed(record) + b"".join(_pack_product(_normalise_product(product)) for product in products)


def _qualification_cell_key(value: dict[str, Any]) -> tuple[int, int, int, int, int]:
    fields = (
        "projectile_z", "projectile_a", "target_z", "target_a", "energy_bin_id"
    )
    try:
        result = tuple(int(value[field]) for field in fields)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("qualification report cell has an invalid identity") from error
    if (
        result[0] <= 0 or result[1] < result[0]
        or result[2] <= 0 or result[3] < result[2] or result[4] < 0
    ):
        raise ValueError("qualification report cell identity is outside its physical domain")
    return result


def _load_qualification_report(
    path: Path,
    raw_paths: list[Path],
    campaign_uuid: str,
    groups: dict[tuple[int, int, int, int, int], list[tuple[dict[str, Any], list[dict[str, Any]]]]],
    *,
    energy_min_mevu: float,
    energy_bin_width_mevu: float,
) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CINEL02 qualification report {path}: {error}") from error
    if not isinstance(report, dict) or report.get("format") != QUALIFICATION_SCHEMA:
        raise ValueError(f"qualification report must declare {QUALIFICATION_SCHEMA}")
    if report.get("structurally_valid") is not True:
        raise ValueError("qualification report is not structurally valid")
    if report.get("campaign_uuid") != campaign_uuid:
        raise ValueError("qualification report campaign UUID disagrees with package metadata")
    report_grid = report.get("energy_grid")
    if not isinstance(report_grid, dict):
        raise ValueError("qualification report has no energy grid")
    try:
        report_minimum = float(report_grid["minimum_MeV_per_u"])
        report_width = float(report_grid["width_MeV_per_u"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("qualification report energy grid is invalid") from error
    if (
        not math.isclose(report_minimum, energy_min_mevu, abs_tol=1.0e-9)
        or not math.isclose(report_width, energy_bin_width_mevu, abs_tol=1.0e-9)
    ):
        raise ValueError("qualification report energy grid disagrees with package grid")
    report_files = report.get("raw_files")
    expected_hashes = sorted(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in raw_paths
    )
    if not isinstance(report_files, list):
        raise ValueError("qualification report has no raw file hash list")
    actual_hashes = []
    for entry in report_files:
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ValueError("qualification report raw file entry is invalid")
        actual_hashes.append(entry["sha256"])
    if sorted(actual_hashes) != expected_hashes:
        raise ValueError("qualification report raw file hashes disagree with package input")
    report_cells = report.get("cells")
    if not isinstance(report_cells, list):
        raise ValueError("qualification report has no cells")
    keyed_cells: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}
    for cell in report_cells:
        if not isinstance(cell, dict):
            raise ValueError("qualification report cell is not an object")
        key = _qualification_cell_key(cell)
        if key in keyed_cells:
            raise ValueError(f"qualification report contains duplicate cell: {key}")
        keyed_cells[key] = cell
    package_keys = set(groups)
    if set(keyed_cells) != package_keys:
        missing = sorted(package_keys - set(keyed_cells))
        extra = sorted(set(keyed_cells) - package_keys)
        raise ValueError(
            "qualification report cell coverage disagrees with package: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    for key, cell in keyed_cells.items():
        if not isinstance(cell.get("qualified"), bool):
            raise ValueError(f"qualification report cell {key} has no boolean qualified status")
        effective_count = cell.get("effective_event_count")
        if (
            not isinstance(effective_count, (int, float))
            or not math.isfinite(float(effective_count))
            or effective_count < 0.0
        ):
            raise ValueError(f"qualification report cell {key} has invalid effective event count")
    if not isinstance(report.get("qualified"), bool):
        raise ValueError("qualification report has no boolean overall qualified status")
    if report["qualified"] != all(cell["qualified"] for cell in keyed_cells.values()):
        raise ValueError("qualification report overall status disagrees with cell statuses")
    return report


def compile_package(
    raw_paths: list[Path], metadata_path: Path, output_path: Path, output_metadata_path: Path,
    *, energy_min_mevu: float = 0.0, energy_bin_width_mevu: float = 1.0,
    minimum_events_per_bin: int = DEFAULT_MINIMUM_EVENTS_PER_BIN,
    maximum_unsupported_product_energy_fraction: float = DEFAULT_MAX_UNSUPPORTED_PRODUCT_ENERGY_FRACTION,
    qualification_report_path: Path | None = None,
    single_target: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if not math.isfinite(energy_min_mevu) or not math.isfinite(energy_bin_width_mevu) or energy_bin_width_mevu <= 0.0:
        raise ValueError("CINEL02 energy grid must be finite with positive width")
    if minimum_events_per_bin <= 0:
        raise ValueError("minimum_events_per_bin must be positive")
    if (
        not math.isfinite(maximum_unsupported_product_energy_fraction)
        or maximum_unsupported_product_energy_fraction <= 0.0
    ):
        raise ValueError("maximum unsupported-product energy fraction must be positive and finite")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = validate_raw_files(raw_paths, metadata, single_target=single_target)
    interaction_isomer_normalisation = {
        "count": 0,
        "excitation_energy_MeV": 0.0,
        "by_pdg": {},
    }
    runtime_records = []
    for record, products in records:
        ground_pdg = _ground_state_isomer_pdg(
            record["projectile_pdg"], record["projectile_z"],
            record["projectile_a"], record["projectile_charge"],
            record["projectile_rest_mass"], record["projectile_excitation"],
        )
        if ground_pdg is None:
            runtime_records.append((record, products))
            continue
        parent_ground_pdg = _ground_state_isomer_pdg(
            record["parent_pdg"], record["parent_z"], record["parent_a"],
            record["parent_charge"], record["parent_rest_mass"],
            record["parent_excitation"],
        )
        if parent_ground_pdg != ground_pdg:
            raise ValueError("projectile/parent isomer normalisation disagrees")
        runtime_record = dict(record)
        original_pdg = str(int(record["projectile_pdg"]))
        excitation = float(record["projectile_excitation"])
        runtime_record["projectile_pdg"] = ground_pdg
        runtime_record["projectile_excitation"] = 0.0
        runtime_record["parent_pdg"] = ground_pdg
        runtime_record["parent_excitation"] = 0.0
        interaction_isomer_normalisation["count"] += 1
        interaction_isomer_normalisation["excitation_energy_MeV"] += excitation
        interaction_isomer_normalisation["by_pdg"][original_pdg] = (
            interaction_isomer_normalisation["by_pdg"].get(original_pdg, 0) + 1
        )
        runtime_records.append((runtime_record, products))
    records = runtime_records
    campaign_uuid = _campaign_uuid(metadata)
    groups: dict[tuple[int, int, int, int, int], list[tuple[dict[str, Any], list[dict[str, Any]]]]] = defaultdict(list)
    for record, products in records:
        index = math.floor((float(record["collision_energy_MeV_per_u"]) - energy_min_mevu) / energy_bin_width_mevu)
        if index < 0:
            raise ValueError("interaction is below CINEL02 energy grid")
        groups[(int(record["projectile_z"]), int(record["projectile_a"]), int(record["target_z"]), int(record["target_a"]), index)].append((record, products))
    by_species: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for key, members in groups.items():
        if len(members) < minimum_events_per_bin:
            raise ValueError(f"sparse CINEL02 cell {key}: {len(members)} < {minimum_events_per_bin}")
        by_species[key[:4]].append(key[4])
    for species, bins in by_species.items():
        expected = set(range(min(bins), max(bins) + 1))
        missing = sorted(expected - set(bins))
        if missing:
            raise ValueError(f"empty CINEL02 energy cells for {species}: {missing[:20]}")
    qualification_report = None
    if qualification_report_path is not None:
        qualification_report = _load_qualification_report(
            qualification_report_path,
            raw_paths,
            campaign_uuid,
            groups,
            energy_min_mevu=energy_min_mevu,
            energy_bin_width_mevu=energy_bin_width_mevu,
        )

    ordered_keys = sorted(groups)
    ordered_records: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    index_rows: list[tuple[int, int, int, int, int, int, int, float, float]] = []
    for key in ordered_keys:
        members = sorted(
            groups[key],
            key=lambda item: (
                float(item[0]["collision_energy_MeV_per_u"]),
                int(item[0]["run_id"]),
                int(item[0]["thread_id"]),
                int(item[0]["event_id"]),
                int(item[0]["track_id"]),
                int(item[0]["interaction_sequence"]),
            ),
        )
        offset = len(ordered_records)
        ordered_records.extend(members)
        lower = energy_min_mevu + key[4] * energy_bin_width_mevu
        index_rows.append((*key[:4], key[4], offset, len(members), lower, lower + energy_bin_width_mevu))
    isomer_normalisation = {
        "count": 0,
        "kinetic_energy_MeV": 0.0,
        "excitation_energy_MeV": 0.0,
        "by_pdg": {},
    }
    supported_ledger_normalisation = {
        "count": 0,
        "kinetic_energy_MeV": 0.0,
        "by_pdg": {},
    }
    runtime_records = []
    for record, products in ordered_records:
        runtime_record = dict(record)
        runtime_products = []
        for product in products:
            normalised = _ground_state_isomer_product(product)
            normalisation_kind = "isomer" if normalised is not None else None
            if normalised is None:
                candidate = _normalise_product(product)
                if (
                    candidate["role"] == PRODUCT_UNSUPPORTED_BUT_RECORDED
                    and _supported_product_identity(candidate)
                ):
                    normalised = dict(candidate)
                    normalised["role"] = PRODUCT_DIRECT_SECONDARY
                    normalisation_kind = "supported_ledger"
                else:
                    runtime_products.append(product)
                    continue
            kinetic = float(product["kinetic_energy_MeV"])
            excitation = float(product["excitation"])
            original_pdg = str(int(product["pdg"]))
            runtime_record["unsupported_product_count"] -= 1
            runtime_record["unsupported_product_energy_MeV"] -= kinetic
            if (
                runtime_record["unsupported_product_count"] < 0
                or runtime_record["unsupported_product_energy_MeV"] < -2.0e-5
            ):
                raise ValueError(
                    "runtime-supported normalisation underflowed the unsupported ledger"
                )
            runtime_record["unsupported_product_energy_MeV"] = max(
                0.0, runtime_record["unsupported_product_energy_MeV"]
            )
            if normalisation_kind == "isomer":
                isomer_normalisation["count"] += 1
                isomer_normalisation["kinetic_energy_MeV"] += kinetic
                isomer_normalisation["excitation_energy_MeV"] += excitation
                isomer_normalisation["by_pdg"][original_pdg] = (
                    isomer_normalisation["by_pdg"].get(original_pdg, 0) + 1
                )
            else:
                supported_ledger_normalisation["count"] += 1
                supported_ledger_normalisation["kinetic_energy_MeV"] += kinetic
                supported_ledger_normalisation["by_pdg"][original_pdg] = (
                    supported_ledger_normalisation["by_pdg"].get(original_pdg, 0) + 1
                )
            runtime_products.append(normalised)
        runtime_records.append((runtime_record, runtime_products))
    ordered_records = runtime_records


    # The cell index above remains the on-disk compatibility view.  The
    # global index is the runtime lookup view: one sorted node per exact
    # captured energy/species tuple, followed by a prefix-offset array and a
    # permutation of complete interaction indices.  Events are never split
    # into products or rescaled when a query crosses a cell boundary.
    global_groups: dict[tuple[int, int, int, int, float], list[int]] = defaultdict(list)
    for interaction_index, (record, _) in enumerate(ordered_records):
        global_groups[
            (
                int(record["projectile_z"]),
                int(record["projectile_a"]),
                int(record["target_z"]),
                int(record["target_a"]),
                float(record["collision_energy_MeV_per_u"]),
            )
        ].append(interaction_index)
    global_event_indices: list[int] = []
    global_event_offsets = [0]
    global_node_payload_parts: list[bytes] = []
    for key in sorted(global_groups):
        global_node_payload_parts.append(PACKAGE_ENERGY_NODE.pack(*key))
        global_event_indices.extend(global_groups[key])
        global_event_offsets.append(len(global_event_indices))
    global_index_payload = (
        b"".join(global_node_payload_parts)
        + b"".join(PACKAGE_ENERGY_OFFSET.pack(offset) for offset in global_event_offsets)
        + b"".join(PACKAGE_EVENT_INDEX.pack(index) for index in global_event_indices)
    )

    unsupported_by_pdg: dict[str, dict[str, float | int]] = {}
    unsupported_product_count = 0
    unsupported_product_energy_MeV = 0.0
    captured_incident_energy_MeV = math.fsum(
        float(record["collision_energy_MeV"]) for record, _ in ordered_records
    )
    for _, products in ordered_records:
        for product in products:
            normalised = _normalise_product(product)
            if _supported_product_identity(normalised):
                continue
            pdg_key = str(normalised["pdg"])
            entry = unsupported_by_pdg.setdefault(
                pdg_key, {"count": 0, "kinetic_energy_MeV": 0.0}
            )
            entry["count"] = int(entry["count"]) + 1
            entry["kinetic_energy_MeV"] = (
                float(entry["kinetic_energy_MeV"])
                + normalised["kinetic_energy_MeV"]
            )
            unsupported_product_count += 1
            unsupported_product_energy_MeV += normalised["kinetic_energy_MeV"]
    if captured_incident_energy_MeV > 0.0:
        unsupported_product_energy_fraction = (
            unsupported_product_energy_MeV / captured_incident_energy_MeV
        )
    elif unsupported_product_energy_MeV > 0.0:
        raise ValueError("unsupported products have energy but captured incident energy is zero")
    else:
        unsupported_product_energy_fraction = 0.0
    if not unsupported_product_energy_fraction < maximum_unsupported_product_energy_fraction:
        raise ValueError(
            "unsupported-product energy fraction exceeds the CINPKG03 gate: "
            f"{unsupported_product_energy_fraction:.9g} >= "
            f"{maximum_unsupported_product_energy_fraction:.9g}"
        )

    # Package records are stored as two contiguous arrays. Product offsets are
    # recovered deterministically from direct_product_count in interaction
    # order; keeping arrays separate also makes device upload straightforward.
    package_payload = (
        b"".join(_pack_fixed(record) for record, _ in ordered_records)
        + b"".join(
            _pack_product(_normalise_product(product))
            for _, products in ordered_records
            for product in products
        )
    )
    index_payload = b"".join(PACKAGE_INDEX.pack(*row) for row in index_rows)
    header_size = PACKAGE_HEADER_SIZE
    file_size = header_size + len(index_payload) + len(package_payload) + len(global_index_payload)
    header = PACKAGE_HEADER.pack(
        PACKAGE_MAGIC, PACKAGE_VERSION, header_size, 0x01020304,
        PACKAGE_INDEX.size, RAW_FIXED_FORMAT.size, PRODUCT_FORMAT.size,
        FLAG_AUTHORITATIVE | FLAG_TARGET_KNOWN | FLAG_PROCESS_LOCAL_DEPOSIT |
        FLAG_PARENT_FINAL_STATE | FLAG_GLOBAL_ENERGY_INDEX,
        len(index_rows), len(ordered_records), sum(len(products) for _, products in ordered_records), file_size,
        float(energy_min_mevu), float(energy_bin_width_mevu), minimum_events_per_bin,
        len(global_node_payload_parts),
        campaign_uuid.encode("ascii") + b"\0" * (40 - len(campaign_uuid)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(header + index_payload + package_payload + global_index_payload)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    statistically_qualified = (
        qualification_report is not None and qualification_report.get("qualified") is True
    )
    qualification_cells = []
    qualification_energy_gaps = []
    if qualification_report is not None:
        qualification_cells = [
            {
                field: cell[field]
                for field in (
                    "projectile_z", "projectile_a", "target_z", "target_a",
                    "energy_bin_id", "raw_event_count", "effective_event_count",
                    "product_count", "qualified", "gates",
                )
                if field in cell
            }
            for cell in qualification_report["cells"]
        ]
        qualification_energy_gaps = qualification_report.get("energy_gaps", [])
    compiled = {
        "format": "CINPKG03",
        "format_version": PACKAGE_VERSION,
        "runtime_safe": True,
        "runtime_disposition": (
            "statistically_qualified"
            if statistically_qualified
            else "structurally_valid"
        ),
        "structurally_valid": True,
        "statistically_qualified": statistically_qualified,
        "campaign_uuid": campaign_uuid,
        "collision_state_source": metadata["collision_state_source"],
        "final_state_source": metadata["final_state_source"],
        "local_deposit": metadata["local_deposit"],
        "target_selection": "target-first; target-specific cross sections are required at runtime",
        "depth_conditioned_production": False,
        "scalar_energy_scaling": False,
        "nearest_fill": False,
        "minimum_events_per_bin": minimum_events_per_bin,
        "unsupported_product_count": unsupported_product_count,
        "unsupported_product_energy_MeV": unsupported_product_energy_MeV,
        "captured_incident_energy_MeV": captured_incident_energy_MeV,
        "unsupported_product_energy_fraction_of_captured_incident": (
            unsupported_product_energy_fraction
        ),
        "unsupported_products": {
            "runtime_disposition": "ledger_untracked",
            "count": unsupported_product_count,
            "kinetic_energy_MeV": unsupported_product_energy_MeV,
            "captured_incident_energy_MeV": captured_incident_energy_MeV,
            "energy_fraction_of_captured_incident": unsupported_product_energy_fraction,
            "maximum_energy_fraction": maximum_unsupported_product_energy_fraction,
            "by_pdg": dict(sorted(unsupported_by_pdg.items(), key=lambda item: int(item[0]))),
        },
        "runtime_supported_ledger_normalisation": {
            "runtime_disposition": "direct_secondary_transport",
            "count": supported_ledger_normalisation["count"],
            "kinetic_energy_MeV": supported_ledger_normalisation["kinetic_energy_MeV"],
            "by_pdg": dict(
                sorted(
                    supported_ledger_normalisation["by_pdg"].items(),
                    key=lambda item: int(item[0]),
                )
            ),
        },
        "isomer_ground_state_normalisation": {
            "runtime_disposition": "same-Z/A ground-state transport",
            "projectile_interactions": {
                "count": interaction_isomer_normalisation["count"],
                "excitation_energy_MeV": interaction_isomer_normalisation[
                    "excitation_energy_MeV"
                ],
                "by_pdg": dict(
                    sorted(
                        interaction_isomer_normalisation["by_pdg"].items(),
                        key=lambda item: int(item[0]),
                    )
                ),
            },
            "count": isomer_normalisation["count"],
            "kinetic_energy_MeV": isomer_normalisation["kinetic_energy_MeV"],
            "excitation_energy_MeV": isomer_normalisation["excitation_energy_MeV"],
            "by_pdg": dict(
                sorted(
                    isomer_normalisation["by_pdg"].items(),
                    key=lambda item: int(item[0]),
                )
            ),
        },
        "qualification": {
            "minimum_events_per_bin": minimum_events_per_bin,
            "structurally_valid": True,
            "statistically_qualified": statistically_qualified,
            "report": (
                {
                    "format": QUALIFICATION_SCHEMA,
                    "path": str(qualification_report_path),
                    "sha256": hashlib.sha256(qualification_report_path.read_bytes()).hexdigest(),
                    "qualified": bool(qualification_report["qualified"]),
                    "thresholds": qualification_report.get("thresholds", {}),
                }
                if qualification_report is not None
                else None
            ),
            "cells": qualification_cells,
            "energy_gaps": qualification_energy_gaps,
            "unsupported_product_energy_fraction_gate": {
                "maximum": maximum_unsupported_product_energy_fraction,
                "value": unsupported_product_energy_fraction,
                "passed": True,
            },
        },
        "energy_grid": {"minimum_MeV_per_u": energy_min_mevu, "width_MeV_per_u": energy_bin_width_mevu},
        "global_energy_index": {
            "persisted": True,
            "node_count": len(global_node_payload_parts),
            "offset_count": len(global_event_offsets),
            "event_index_count": len(global_event_indices),
            "event_index_width_bytes": PACKAGE_EVENT_INDEX.size,
        },
        "records": {"cells": len(index_rows), "interactions": len(ordered_records), "products": sum(len(products) for _, products in ordered_records)},
        "raw_files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in raw_paths],
        "output": {"path": str(output_path), "bytes": file_size, "sha256": digest},
        "provenance": metadata.get("provenance", {}),
    }
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.write_text(json.dumps(compiled, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return compiled


def make_metadata(**overrides: Any) -> dict[str, Any]:
    """Return a strict starter sidecar for a TOPAS capture campaign."""
    metadata: dict[str, Any] = {
        "schema": {"name": "CINEL02", "version": RAW_VERSION},
        "collision_state_source": {
            "authoritative": True,
            "object": "G4Track passed to wrapped PostStepDoIt",
            "pre_step_authoritative": False,
        },
        "final_state_source": {
            "object": "G4VParticleChange returned by the same delegated PostStepDoIt",
            "delayed_secondary_reconstruction": False,
        },
        "local_deposit": {"definition": "G4VParticleChange::GetLocalEnergyDeposit", "runtime_safe": True},
        "depth_conditioned_production": False,
        "scalar_energy_scaling": False,
        "nearest_fill": False,
        "first_step_products": False,
        "whole_step_local_deposit": False,
        "campaign": {"uuid": "00000000-0000-4000-8000-000000000001"},
        "provenance": {"maigo_sha": "5f6f06f37bc9281faaa6606375c51a50fd0df04d"},
    }
    for key, value in overrides.items():
        metadata[key] = value
    return metadata

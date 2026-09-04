#!/usr/bin/env python3
"""Fail-mode tests for tools/verify_schneider_v2_1_data.py.

Builds a miniature fake repo (tiny binaries with real magic/version
headers, manifest, bundle, metadata, channels) and asserts the verifier:
passes clean, and fails on (1) one-byte metadata tamper, (2) bundle SHA
tamper, (3) bundle pin swapped to another existing file, (4) missing
binary, (5) v1-magic content at a v2 path.
"""
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_v2_1", REPO / "tools" / "verify_schneider_v2_1_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fake_bin(magic: str, version: int, payload: bytes = b"\x00" * 64) -> bytes:
    return magic.encode("ascii") + struct.pack("<I", version) + payload


REGISTRY = [{"z": 6, "a": 12}]
TARGETS = [1, 6, 12]


def make_fake_repo(root: Path):
    d = root / "data" / "schneider"
    d.mkdir(parents=True)
    rate = fake_bin("SCHN2RAT", 3)
    (d / "secondary_inelastic_rates_v2_1.bin").write_bytes(rate)
    channels = {"channels": [
        {"projectile_z": 6, "projectile_a": 12, "target_element_z": t}
        for t in TARGETS]}
    (d / "cinel03_c12_targets_v2_1.channels.json").write_text(json.dumps(channels))
    binref = {"file": "data/schneider/secondary_inelastic_rates_v2_1.bin",
              "sha256": sha(rate)}
    bundle = {
        "primary_rate": dict(binref),
        "primary_package": dict(binref, **{
            "channels_file": "data/schneider/cinel03_c12_targets_v2_1.channels.json",
            "channels_sha256": sha(json.dumps(channels).encode())}),
        "secondary_rate": dict(binref),
        "secondary_package": dict(binref, **{
            "channels_file": "data/schneider/cinel03_c12_targets_v2_1.channels.json",
            "channels_sha256": sha(json.dumps(channels).encode())}),
        "stopping_table": dict(binref),
        "projectile_registry": REGISTRY, "target_order": TARGETS}
    (d / "schneider_physics_bundle_v2_1.json").write_text(json.dumps(bundle))
    meta = {"data_filename": "secondary_inelastic_rates_v2_1.bin",
            "data_sha256": sha(rate),
            "file_size_bytes": len(rate)}
    (d / "secondary_inelastic_rates_v2_1.metadata.json").write_text(json.dumps(meta))
    manifest = {
        "schema_version": 2,
        "bundle_registry_sha256": sha(json.dumps(
            REGISTRY, sort_keys=True).encode()),
        "bundle_target_order_sha256": sha(json.dumps(TARGETS).encode()),
        "files": [
            {"path": "data/schneider/secondary_inelastic_rates_v2_1.bin",
             "bytes": len(rate), "sha256": sha(rate),
             "magic": "SCHN2RAT", "version": 3,
             "generator": "fake", "raw_source": [], "in_git": False},
            {"path": "data/schneider/schneider_physics_bundle_v2_1.json",
             "bytes": (d / "schneider_physics_bundle_v2_1.json").stat().st_size,
             "sha256": sha((d / "schneider_physics_bundle_v2_1.json").read_bytes()),
             "generator": "fake", "raw_source": [], "in_git": True},
            {"path": "data/schneider/cinel03_c12_targets_v2_1.channels.json",
             "bytes": (d / "cinel03_c12_targets_v2_1.channels.json").stat().st_size,
             "sha256": sha((d / "cinel03_c12_targets_v2_1.channels.json").read_bytes()),
             "generator": "fake", "raw_source": [], "in_git": True},
            {"path": "data/schneider/secondary_inelastic_rates_v2_1.metadata.json",
             "bytes": (d / "secondary_inelastic_rates_v2_1.metadata.json").stat().st_size,
             "sha256": sha((d / "secondary_inelastic_rates_v2_1.metadata.json").read_bytes()),
             "generator": "fake", "raw_source": [], "in_git": True},
        ]}
    (d / "v2_1_data_manifest.json").write_text(json.dumps(manifest))
    return manifest


class TestStrictVerifier(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_fake_repo(self.root)
        self.ver = load_verifier()

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, argv_repo=True):
        import io
        from contextlib import redirect_stdout
        import sys
        buf = io.StringIO()
        old = sys.argv
        sys.argv = ["verify", "--repo", str(self.root)]
        try:
            with redirect_stdout(buf):
                code = self.ver.main()
        finally:
            sys.argv = old
        return code, buf.getvalue()

    def test_clean_passes(self):
        code, out = self.check()
        self.assertEqual(code, 0, out)
        self.assertIn("OK", out)

    def test_metadata_byte_tamper_fails(self):
        p = self.root / "data/schneider/secondary_inelastic_rates_v2_1.metadata.json"
        doc = json.loads(p.read_text())
        doc["file_size_bytes"] = doc["file_size_bytes"] + 1
        p.write_text(json.dumps(doc))
        code, out = self.check()
        self.assertNotEqual(code, 0, out)

    def test_bundle_sha_tamper_fails(self):
        p = self.root / "data/schneider/schneider_physics_bundle_v2_1.json"
        doc = json.loads(p.read_text())
        doc["secondary_rate"]["sha256"] = "0" * 64
        p.write_text(json.dumps(doc))
        code, out = self.check()
        self.assertNotEqual(code, 0, out)

    def test_bundle_pin_swapped_file_fails(self):
        other = self.root / "data/schneider/other.bin"
        other.write_bytes(fake_bin("SCHN2RAT", 3, b"\x01" * 64))
        p = self.root / "data/schneider/schneider_physics_bundle_v2_1.json"
        doc = json.loads(p.read_text())
        doc["secondary_rate"]["file"] = "data/schneider/other.bin"
        p.write_text(json.dumps(doc))
        code, out = self.check()
        self.assertNotEqual(code, 0, out)

    def test_missing_binary_fails(self):
        (self.root / "data/schneider/secondary_inelastic_rates_v2_1.bin").unlink()
        code, out = self.check()
        self.assertNotEqual(code, 0, out)
        self.assertIn("MISSING", out)

    def test_v1_magic_at_v2_path_fails(self):
        p = self.root / "data/schneider/secondary_inelastic_rates_v2_1.bin"
        v1 = fake_bin("SCHN2RAT", 1, b"\x00" * 64)
        p.write_bytes(v1)
        mpath = self.root / "data/schneider/v2_1_data_manifest.json"
        man = json.loads(mpath.read_text())
        for e in man["files"]:
            if e["path"].endswith(".bin"):
                e["bytes"] = len(v1)
                e["sha256"] = sha(v1)
        mpath.write_text(json.dumps(man))
        code, out = self.check()
        self.assertNotEqual(code, 0, out)
        self.assertIn("MAGIC-MISMATCH", out)


if __name__ == "__main__":
    unittest.main()

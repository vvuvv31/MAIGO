#!/usr/bin/env python3
"""Verify the pinned research-only C12 water joint EM package (not CT data)."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = {
    'joint_nodes.csv': '163cad049def757201d75d419c11c5af8a348ccbf280e83ee5d8b8d702d10d96',
    'raw_vectors.csv': 'f09cb8f0dd601f8853d973a08c4eff3f366ddbcaedb00043183e367dd4e18101',
}

def verify(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['schema'] != 'water_joint_em_v1' or manifest['model'] != 'g4_joint_water_v1':
        raise ValueError('Wrong joint EM schema/model')
    if manifest['validation_status'] != 'research_candidate':
        raise ValueError('This verifier does not authorize a production package')
    for name, expected in EXPECTED.items():
        payload = (directory / name).read_bytes()
        entry = manifest['files'][name]
        if hashlib.sha256(payload).hexdigest() != expected or entry['sha256'] != expected or entry['bytes'] != len(payload):
            raise ValueError('Joint EM hash/size mismatch: ' + name)
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=Path(__file__).resolve().parents[1] / 'data/water_joint_em_v1')
    args = parser.parse_args()
    verify(args.directory)
    print('Verified water_joint_em_v1: research only, C12 primary, unit-density homogeneous water, local delta deposition')

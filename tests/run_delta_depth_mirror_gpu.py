"""Opt-in local CUDA regression; keeps all outputs in a new directory.
Usage: python3 tests/run_delta_depth_mirror_gpu.py --binary ... --config ... --out ...
The input must be the frozen 1k homogeneous section-0 controlled smoke config.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    binary, source, root = a.binary.resolve(), a.config.resolve(), a.out.resolve()
    values = dict(line.split(":", 1) for line in source.read_text().splitlines()
                  if ":" in line and not line.lstrip().startswith("#"))
    values = {k.strip(): v.strip() for k, v in values.items()}
    assert values["number_of_histories"] == "1000"
    assert values["run_mode"] == "smoke"
    assert values["enable_inelastic"] == "false"
    assert values["enable_energy_straggling"] == "false"
    assert values["enable_multiple_scattering"] == "false"
    assert values["beam_energy_spread"] == "0"
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(["python3", str(repo/"tools/verify_schneider_v2_1_data.py")],
                   check=True, cwd=repo)
    root.mkdir(parents=True, exist_ok=False)
    results = {}
    env = dict(os.environ)
    env["ONEAPI_DEVICE_SELECTOR"] = "cuda:*"
    env["LD_LIBRARY_PATH"] = "/home/wuwei/sycl_workspace/llvm/build/install/lib:" + env.get("LD_LIBRARY_PATH", "")
    for name, energy, candidate, overrides in [
        ("base200", 200, False, {}),
        ("long200", 200, True, {}),
        ("base120", 120, False, {}),
        ("long120", 120, True, {}),
        ("oblique200", 200, False, {"beam_uz_x": "0.6", "beam_uz_z": "0.8"}),
        ("edge200", 200, False, {"source_origin_x_mm": "99.9"}),
    ]:
        case = root/name
        case.mkdir()
        cfg = dict(values)
        cfg.update(initial_energy_MeVu=str(energy))
        cfg.update(overrides)
        cfg.pop("ct_schneider_delta_longitudinal_file", None)
        cfg.pop("ct_schneider_delta_longitudinal_scale", None)
        if candidate:
            cfg["ct_schneider_delta_longitudinal_file"] = str(repo/"data/schneider/schneider_section0_c12_delta_longitudinal_v1.csv")
            cfg["ct_schneider_delta_longitudinal_scale"] = "1"
        for key in ("output_file", "dose_output_file", "let_output_file",
                    "voxel_dose_Gy_output_file", "charged_origin_voxel_output_file",
                    "neutral_origin_voxel_output_file", "charged_origin_voxel_dose_Gy_output_file"):
            cfg[key] = ""
        # Keep the strict sparse writer enabled: its per-plane assertion is the regression gate.
        cfg["voxel_dose_output_file"] = str(case/"voxel.csv")
        cfg["voxel_dose_mhd_output_file"] = str(case/"dose.mhd")
        config = case/"run.yaml"
        config.write_text("".join(f"{k}: {v}\n" for k, v in cfg.items()))
        run = subprocess.run([str(binary), "--config", str(config), "--device", "cuda"],
                             cwd=case, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (case/"run.log").write_text(run.stdout)
        q = json.loads((case/"out/run/quality_report.json").read_text())
        codes = [f["code"] for f in q["failures"]]
        expected = ["unvalidated_longitudinal_candidate"] if candidate else []
        complete = (case/"dose.mhd").exists() and (case/"dose.raw").exists()
        passed = codes == expected and complete and "Voxel dose does not close" not in run.stdout
        passed = passed and ((run.returncode != 0) if candidate else (run.returncode == 0))
        results[name] = dict(passed=passed, returncode=run.returncode,
                            failures=codes, dose_complete=complete, config_sha256=sha(config),
                            raw_sha256=sha(case/"dose.raw") if complete else None)
        print(name, results[name], flush=True)
    equal = results["base120"]["raw_sha256"] is not None and results["base120"]["raw_sha256"] == results["long120"]["raw_sha256"]
    report = dict(binary_sha256=sha(binary), source_config_sha256=sha(source),
                  cases=results, out_of_domain_raw_equal=equal,
                  passed=all(r["passed"] for r in results.values()) and equal)
    (root/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())

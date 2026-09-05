"""No real sbatch in these executor failure-path tests."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import execute_electron_response_campaign as ex
from run_electron_response_diagnostic import collect_config_inputs, sha256_file, estimated_shard_bytes


class ExecutorTest(unittest.TestCase):
    def test_measured_density_and_history_gate(self):
        identity = {"density_g_cm3": 0.0113160652}
        valid = {"record_schema_version":3, "parent_step_binding_verified":True,
                 "events_observed":1, "histories_requested":1,
                 "material_density_g_cm3":identity["density_g_cm3"], "dose3d_over_steps":1.0}
        ex.validate_report(valid, identity, 1)
        for key, value in (("material_density_g_cm3",None),
                           ("material_density_g_cm3",float("nan")),
                           ("material_density_g_cm3",0.0393234522),
                           ("events_observed",2), ("histories_requested",2),
                           ("record_schema_version",2), ("parent_step_binding_verified",False),
                           ("dose3d_over_steps",1.01), ("dose3d_over_steps",float("nan"))):
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                ex.validate_report(dict(valid, **{key:value}), identity, 1)

    def test_resource_units(self):
        self.assertEqual(estimated_shard_bytes(1),192*1024**2)
        self.assertEqual(estimated_shard_bytes(12),896*1024**2)
        with self.assertRaises(ValueError):
            estimated_shard_bytes(13)
        self.assertEqual(ex.queue_usage("42|8|1|4Gc\n43|2|2|3Gn"), (10,38*1024**3))
        self.assertEqual(ex.queue_usage(""), (0,0))
        for value in ("0", "N/A", "0G", "unlimited"):
            with self.assertRaises(ValueError):
                ex.memory_bytes(value, 2, 1)

    def test_include_graph_and_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp)
            (b/"child.txt").write_text("sv:Ph/Default/Modules = 1 a\n")
            (b/"base.txt").write_text("includeFile = child.txt\nd:So/Beam/BeamEnergy = 2400 MeV\n")
            pins, effective=collect_config_inputs(b/"base.txt")
            self.assertEqual(len(pins), 2)
            self.assertIn("Modules", effective)
            (b/"child.txt").write_text("includeFile = base.txt\n")
            with self.assertRaisesRegex(ValueError, "Cyclic"):
                collect_config_inputs(b/"base.txt")

    def fixture(self, b):
        c=b/"campaign"; c.mkdir()
        d=c/"shard"; d.mkdir()
        script=d/"run.slurm"; script.write_text("topas")
        analyzer=Path(ex.__file__).parent/"analyze_electron_deposit_steps.py"
        manifest={"record_schema_version":3, "input_files":{str(script):sha256_file(script),
                      str(analyzer):sha256_file(analyzer)},
            "effective_config":"""
s:Ge/Slab/Type = "TsBox"
s:Ge/Slab/Material = "PatientTissueFromHUNegative1000"
d:Ge/Slab/HLX = 100 mm
d:Ge/Slab/HLY = 100 mm
d:Ge/Slab/HLZ = 110 mm
d:Ge/Slab/TransZ = 110 mm
s:Sc/Dose3D/Component = "Slab"
s:Sc/Dose3D/Quantity = "DoseToMedium"
i:Sc/Dose3D/XBins = 100
i:Sc/Dose3D/YBins = 100
i:Sc/Dose3D/ZBins = 440
""",
            "disk_budget_bytes":1024**3, "cpus_per_shard":2,"mem_gb_per_shard":4,
            "histories_per_shard":1,"shards":[{"dir":str(d),"case_id":"test"}]}
        (d/"metadata.json").write_text(json.dumps({"slurm_job_id":None}))
        f=c/"campaign_manifest.json"; f.write_text(json.dumps(manifest))
        return f,d

    def test_disk_breach_cancels_only_own_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp); f,d=self.fixture(b)
            def command(args):
                if args[0]=="squeue": return ""
                if args[0]=="sbatch": return "999"
                raise AssertionError(args)
            with patch.object(ex,"DATA_ROOT",b), patch.object(ex,"command",side_effect=command), \
                 patch.object(ex,"disk_bytes",side_effect=[0,2*1024**3]), \
                 patch.object(ex.subprocess,"run",return_value=SimpleNamespace(returncode=0)) as run:
                with self.assertRaisesRegex(ValueError,"budget breach"):
                    ex.execute(f,-1000,poll=1)
                run.assert_called_once_with(["scancel","999"],capture_output=True,timeout=30)
            state=json.loads((f.parent/"execution.json").read_text())
            self.assertEqual(state["status"],"failed")
            self.assertEqual(json.loads((d/"metadata.json").read_text())["slurm_job_id"],"999")
            self.assertTrue((d/"run.slurm").exists())

    def test_failed_job_never_analyzed(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp); f,d=self.fixture(b)
            def command(args):
                return {"squeue":"", "sbatch":"998", "sacct":"998|FAILED|1:0"}[args[0]]
            with patch.object(ex,"DATA_ROOT",b), patch.object(ex,"command",side_effect=command), \
                 patch.object(ex.subprocess,"run") as run:
                with self.assertRaisesRegex(ValueError,"Slurm job failed"):
                    ex.execute(f,-1000,poll=1)
                run.assert_not_called()
            self.assertFalse((d/"analysis.json").exists())

    def test_mutated_input_refused_before_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp); f,d=self.fixture(b)
            (d/"run.slurm").write_text("changed")
            with patch.object(ex,"DATA_ROOT",b), patch.object(ex,"command") as command:
                with self.assertRaisesRegex(ValueError,"SHA mismatch"):
                    ex.execute(f,-1000)
                command.assert_not_called()

    def test_completed_job_requires_valid_analysis(self):
        for correct_density in (True, False):
            with self.subTest(correct_density=correct_density), tempfile.TemporaryDirectory() as tmp:
                b=Path(tmp); f,d=self.fixture(b)
                def command(args):
                    return {"squeue":"", "sbatch":"997", "sacct":"997|COMPLETED|0:0"}[args[0]]
                def analyze(args, **kwargs):
                    self.assertIn("--steps", args)
                    report={"record_schema_version":3,"parent_step_binding_verified":True,
                            "events_observed":1,"histories_requested":1,"dose3d_over_steps":1.0,
                            "material_density_g_cm3":.0113160652 if correct_density else .0393}
                    for key in ("steps", "header", "dose"):
                        raw=d/("fake_"+key); raw.write_bytes(b"fixture")
                        report["raw_"+key]=str(raw)
                        report[key+"_sha256"]=sha256_file(raw)
                    Path(args[args.index("--output")+1]).write_text(json.dumps(report))
                    return SimpleNamespace(returncode=0)
                with patch.object(ex,"DATA_ROOT",b), patch.object(ex,"command",side_effect=command), \
                     patch.object(ex.subprocess,"run",side_effect=analyze) as run:
                    if correct_density:
                        ex.execute(f,-1000,poll=1)
                    else:
                        with self.assertRaisesRegex(ValueError,"Measured material density"):
                            ex.execute(f,-1000,poll=1)
                    self.assertEqual(run.call_count,1)  # completed job must not be cancelled
                meta=json.loads((d/"metadata.json").read_text())
                state=json.loads((f.parent/"execution.json").read_text())
                self.assertEqual(meta["completion_status"],"completed")
                self.assertEqual(meta["analysis_status"],"validated" if correct_density else "failed")
                self.assertEqual(state["status"],"validated" if correct_density else "failed")
                if correct_density:
                    self.assertEqual(meta["histories_actual"],1)
                    self.assertEqual(meta["report_sha256"],sha256_file(d/"analysis.json"))


if __name__=="__main__":
    unittest.main()

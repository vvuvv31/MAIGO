import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from bootstrap_stratum_radial import source_moments,summarize,audit,RHO,sha

class StratumRadialTests(unittest.TestCase):
    def test_exact_endpoint_and_sha(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.npz'
            np.savez(p,vectors=np.array([[.001,0,0],[-.001,0,.01],[.003,.004,0]]),
                     offsets=np.array([0,2,3]),bins=np.array([34,34]),weights=np.array([1.,3.]))
            m=source_moments(p,-1000,sha(p))
            self.assertEqual(m[34,0],4.)
            self.assertAlmostEqual(m[34,1],3*.005**2*(10/RHO[-1000])**2)
            with self.assertRaisesRegex(ValueError,'SHA'):source_moments(p,-1000,'0'*64)

    def test_bad_payload_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.npz'
            for w in (float('nan'),float('inf'),-1.):
                np.savez(p,vectors=np.ones((1,3)),offsets=np.array([0,1]),
                         bins=np.array([34]),weights=np.array([w]))
                with self.assertRaises(ValueError):source_moments(p,100,sha(p))
            np.savez(p,vectors=np.ones((1,3)),offsets=np.array([0,2]),bins=np.array([34]),weights=np.ones(1))
            with self.assertRaises(ValueError):source_moments(p,100,sha(p))

    def test_weighting_missing_bins_and_determinism(self):
        a=np.zeros((37,2));b=a.copy();a[34]=[1,1];b[34]=[3,27]
        out=summarize([a,b],100,7)
        self.assertEqual(out,summarize([a,b],100,7))
        self.assertAlmostEqual(out[34]['rms_mm'],np.sqrt(7))
        self.assertEqual(out[33]['missing_exposure_replicates'],100)
        self.assertIsNone(out[33]['bootstrap_percentile_95'])
        for count in (0,-1,1):
            with self.assertRaises(ValueError):summarize([a],count,7)

    def test_manifest_chain_and_group_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r3=root/'r3.json';idx=root/'index.json'
            raw=[];sources={};index={}
            for i,hu in enumerate((-1000,100)):
                name=f'source_{i:03d}.npz';p=root/name
                np.savez(p,vectors=np.ones((1,3)),offsets=np.array([0,1]),bins=np.array([34]),weights=np.ones(1))
                rawpath=f'/campaign/mat{i}/seed/steps.phsp';rawsha=str(i)*64
                raw.append(dict(path=rawpath,hu=hu,sha256=rawsha))
                sources[name]=dict(raw_path=rawpath,hu=hu,raw_sha256=rawsha,payload_sha256=sha(p))
                group=root/f'group{i}';group.mkdir()
                index[str(i)]=dict(hu=hu,campaign='campaign',n_varied=1,n_total=2,dir=str(group))
            r3.write_text(json.dumps(dict(sources=raw)))
            manifest=dict(sources=sources,source_metadata_sha256=sha(r3))
            (root/'manifest.json').write_text(json.dumps(manifest));idx.write_text(json.dumps(index))
            for i in range(2):(root/f'group{i}/manifest.json').write_text(json.dumps(manifest))
            self.assertFalse(audit(root,r3,idx,10)['interface_gate_passed'])
            altered=dict(manifest,sources={})
            (root/'group0/manifest.json').write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,'Stratum manifest'):audit(root,r3,idx,10)
            (root/'group0/manifest.json').write_text(json.dumps(manifest))
            r3.write_text(r3.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'Metadata SHA'):audit(root,r3,idx,10)

if __name__=='__main__':unittest.main()

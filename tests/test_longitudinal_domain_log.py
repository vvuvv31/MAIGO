import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from analyze_longitudinal_domain_log import summarize
class DomainLogTests(unittest.TestCase):
    def fixture(self):
        return dict(longitudinal_domain_log=dict(truncated=False,records=[
            dict(history=1,step=40,initial_energy_MeVu=175,query_energy_MeVu=149.9,
                 step_start_z_mm=200,step_length_mm=.5,retained_MeV=.1234567)]),
            longitudinal_candidate=dict(out_of_domain_queries=1,out_of_domain_local_energy_MeV=.123456))
    def test_valid(self):
        r=summarize(self.fixture())
        self.assertEqual(r["classification"],{"below":1})
        self.assertTrue(r["all_initially_in_domain"])
    def test_refusals(self):
        for change in ("truncated","count","energy","nan","duplicate","covered"):
            d=self.fixture()
            if change=="truncated":d["longitudinal_domain_log"]["truncated"]=True
            if change=="count":d["longitudinal_candidate"]["out_of_domain_queries"]=2
            if change=="energy":d["longitudinal_candidate"]["out_of_domain_local_energy_MeV"]=0
            if change=="nan":d["longitudinal_domain_log"]["records"][0]["query_energy_MeVu"]=float("nan")
            if change=="covered":d["longitudinal_domain_log"]["records"][0]["query_energy_MeVu"]=175
            if change=="duplicate":
                d["longitudinal_domain_log"]["records"]*=2
                d["longitudinal_candidate"]["out_of_domain_queries"]=2
            with self.subTest(change=change),self.assertRaises(ValueError):summarize(d)
if __name__=="__main__":unittest.main()

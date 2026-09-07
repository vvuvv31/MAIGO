import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_ct_electron_gamma_probe import validate_experiment_exit


class ExitGuardTest(unittest.TestCase):
    def test_expected_quality_refusal(self):
        validate_experiment_exit(1,'carbon_mc: run quality fail (mode=smoke, failures=1, approximations=1): expected\n',True)
        validate_experiment_exit(0,'Completed\n',False)

    def test_io_exception_is_not_expected_quality(self):
        for code,log in [(1,'carbon_mc: Charged-origin depth category size does not match configuration'),
                         (-9,''),(1,''),(0,'carbon_mc: another error')]:
            with self.subTest(code=code,log=log):
                with self.assertRaises(RuntimeError):validate_experiment_exit(code,log,True)

    def test_multiple_errors_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_experiment_exit(1,'carbon_mc: run quality fail (mode=smoke, failures=1, approximations=1): expected\ncarbon_mc: I/O failed',True)


if __name__=='__main__':unittest.main()

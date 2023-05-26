import unittest

import COPASI
import sys
import os
_PATH = os.path.abspath(os.path.dirname(__file__))
_BENCHMARK_DIR = os.path.join(_PATH, '..', 'benchmarks',
                              'hackathon_contributions_new_data_format')
sys.path.append(_PATH)
from copasi_petab_importer import convert_petab
import glob

try:
    import petabtests
    petab_tests_available = True
except ImportError:
    petab_tests_available = False

class TestConverter(unittest.TestCase):
    def test_copasi_version(self):
        self.assertGreater(int(COPASI.CVersion.VERSION.getVersionDevel()),  214,
                           "Need newer COPASI version")

    @unittest.skipIf(not os.path.exists(_BENCHMARK_DIR), 'benchmarks not available')
    def test_petab_problem(self):
        problem = convert_petab.PEtabProblem.from_yaml(os.path.join(_BENCHMARK_DIR,
                                        'Bruno_JExpBio2016/'
                                        'Bruno_JExpBio2016.yaml'))
        self.assertTrue(os.path.exists(problem.measurement_file))
        self.assertTrue(os.path.exists(problem.model_file))
        self.assertTrue(os.path.exists(problem.observable_file))
        self.assertTrue(os.path.exists(problem.condition_file))
        self.assertTrue(os.path.exists(problem.parameter_file))

    @unittest.skipIf(not os.path.exists(_BENCHMARK_DIR), 'benchmarks not available')
    def test_import_bruno(self):
        petab_dir = os.path.join(_BENCHMARK_DIR, 'Bruno_JExpBio2016')
        self.assertTrue(os.path.exists(petab_dir))
        model_name = 'Bruno_JExpBio2016'
        out_dir = os.path.join(_PATH, 'out')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        converter = convert_petab.PEtabConverter(petab_dir, model_name, out_dir)
        converter.convert()

        self.assertTrue(os.path.exists(converter.experimental_data_file))
    @unittest.skipIf(not os.path.exists(_BENCHMARK_DIR), 'benchmarks not available')
    def test_import_bachmann(self):
        petab_dir = os.path.join(_BENCHMARK_DIR, 'Bachmann_MSB2011')
        self.assertTrue(os.path.exists(petab_dir))
        model_name = 'Bachmann_MSB2011'
        out_dir = os.path.join(_PATH, 'out')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        converter = convert_petab.PEtabConverter(petab_dir, model_name, out_dir)
        converter.convert()

        self.assertTrue(os.path.exists(converter.experimental_data_file))
    @unittest.skipUnless(petab_tests_available, 'petabtests need to be installed to run this')
    def test_petabtest_import(self):
        cases = [c for c in glob.glob(str(petabtests.CASES_DIR) + '/*/*.yaml') if '0' in c and 'solution' not in c]
        for case in cases:
            if '17' not in case:
                continue
            converter = convert_petab.PEtabConverter.from_yaml(case)
            converter.convert()


if __name__ == '__main__':
    unittest.main()

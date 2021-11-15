import unittest

import COPASI
import sys
import os
_PATH = os.path.abspath("./")
sys.path.append(_PATH)
from copasi_petab_importer import convert_petab
import glob

try:
    import petabtests
    petab_tests_available = True
except ImportError:
    petab_tests_available = False


def test_copasi_version():
    assert int(COPASI.CVersion.VERSION.getVersionDevel()) >= 214, \
        "Need newer COPASI version"


def test_petab_problem():
    problem = convert_petab.PEtabProblem.from_yaml(os.path.join(_PATH, 'benchmarks/'
                                    'hackathon_contributions_new_data_format/'
                                    'Bruno_JExpBio2016/'
                                    'Bruno_JExpBio2016.yaml'))
    assert os.path.exists(problem.measurement_file)
    assert os.path.exists(problem.model_file)
    assert os.path.exists(problem.observable_file)
    assert os.path.exists(problem.condition_file)
    assert os.path.exists(problem.parameter_file)


def test_import():
    petab_dir = os.path.join(_PATH, 'benchmarks/'
                                    'hackathon_contributions_new_data_format/'
                                    'Bruno_JExpBio2016')
    assert os.path.exists(petab_dir)
    model_name = 'Bruno_JExpBio2016'
    out_dir = os.path.join(_PATH, 'out')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    converter = convert_petab.PEtabConverter(petab_dir, model_name, out_dir)
    converter.convert()

    assert os.path.exists(converter.experimental_data_file)


@unittest.skipUnless(petab_tests_available, 'petabtests need to be installed to run this')
def test_petabtest_import():
    cases = [c for c in glob.glob(petabtests.CASES_DIR + '/*/*.yaml') if '0' in c and 'solution' not in c]
    for case in cases:
        if '17' not in case:
            continue
        converter = convert_petab.PEtabConverter.from_yaml(case)
        converter.convert()

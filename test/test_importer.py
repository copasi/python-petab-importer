import COPASI
import sys
import os
_PATH = os.path.abspath("./")
sys.path.append(_PATH)
import convert_petab


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

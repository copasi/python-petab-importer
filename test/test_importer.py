import COPASI
import sys
import os
_PATH = os.path.abspath("./")
sys.path.append(_PATH)
import convert_petab


def test_copasi_version():
    assert int(COPASI.CVersion.VERSION.getVersionDevel()) >= 214, \
        "Need newer COPASI version"


def test_import():
    petab_dir = os.path.join(_PATH, 'benchmarks/'
                                    'hackathon_contributions_new_data_format/'
                                    'Becker_Science2010')
    assert os.path.exists(petab_dir)
    model_name = 'Becker_Science2010__BaF3_Exp'
    out_dir = os.path.join(_PATH, 'out')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    converter = convert_petab.PEtabConverter(petab_dir, model_name, out_dir)
    converter.convert()

    assert os.path.exists(converter.experimental_data_file)

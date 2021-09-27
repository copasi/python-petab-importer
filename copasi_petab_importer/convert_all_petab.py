from copasi_petab_importer import convert_petab
import sys
import os


def convert_all_benchmarks(base_dir, out_dir):
    dirs = None
    for (dirpath, dirnames, filenames) in os.walk(base_dir):
        dirs = sorted(dirnames)
        break

    for directory in dirs:
        current_dir = os.path.join(base_dir, directory)
        models = get_all_models(current_dir)
        num_models = len(models)
        if num_models == 0:
            print("No models for %s ... skipping" % directory)
            continue

        for model in models:
            if num_models == 1:
                print("  Converting model {0}".format(model))
            else:
                print("  Converting model {0} of {1}".format(model, directory))

            convert_model(current_dir, model, out_dir)

    pass


def get_all_models(current_dir):
    models = []
    for (dirpath, dirnames, filenames) in os.walk(current_dir):
        for name in filenames:
            assert (isinstance(name, str))
            if name.endswith('.xml'):
                name = name[:-4]
                if name.startswith('model_'):
                    name = name[6:]
                models.append(name)
        break
    models = sorted(models)
    return models


def convert_model(current_dir, model, out_dir):
    try:
        worker = convert_petab.PEtabConverter(current_dir,
                                              model, out_dir)
        worker.convert()
    except BaseException:
        import traceback
        print("Couldn't convert {0} due to\n\n{1}".format(
            model, traceback.format_exc()))


if __name__ == "__main__":
    num_args = len(sys.argv)
    if num_args > 2:
        base_dir = sys.argv[1]
        out_dir = sys.argv[2]
    else:
        base_dir = '../benchmarks/hackathon_contributions_new_data_format'
        out_dir = '../out'

    convert_all_benchmarks(base_dir, out_dir)

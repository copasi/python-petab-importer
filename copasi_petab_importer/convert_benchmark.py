import COPASI
import sys
import os
import pandas as pd

dm = COPASI.CRootContainer.addDatamodel()
assert (isinstance(dm, COPASI.CDataModel))
print("using COPASI: %s" % COPASI.CVersion.VERSION.getVersion())


def get_parameters_from_info_file(info_file):
    df = pd.read_excel(info_file, sheet_name="Parameters")
    return df


def get_experimental_data(data_file):
    df = pd.read_excel(data_file, sheet_name="Exp Data")
    return df


def create_experimental_data_mapping(data, exp_file_name):
    task = dm.getTask('Parameter Estimation')
    problem = task.getProblem()
    exp_set = problem.getExperimentSet()
    exp = COPASI.CExperiment(dm)
    exp = exp_set.addExperiment(exp)
    info = COPASI.CExperimentFileInfo(exp_set)
    info.setFileName(exp_file_name)
    exp.setFirstRow(1)
    exp.setLastRow(info.countLines()-1)
    exp.setHeaderRow(1)
    exp.setFileName(os.path.basename(exp_file_name))
    exp.setExperimentType(COPASI.CTaskEnum.Task_timeCourse)
    exp.setSeparator(',')
    info.sync()
    cols = data.columns.to_list()
    exp.setNumColumns(len(cols))

    # now do the mapping
    obj_map = exp.getObjectMap()
    obj_map.setNumCols(len(cols))
    for i in range(len(cols)):
        exp_type = COPASI.CExperiment.ignore
        if cols[i] == 'time':
            exp_type = COPASI.CExperiment.time
        else:
            obj = dm.findObjectByDisplayName('Values[' + cols[i] + ']')
            if obj is not None:
                cn = obj.getValueReference().getCN()
                exp_type = COPASI.CExperiment.dependent
                obj_map.setRole(i, exp_type)
                obj_map.setObjectCN(i, cn)
                exp.calculateWeights()
                continue
        obj_map.setRole(i, exp_type)


def add_fit_items(parameters):
    task = dm.getTask('Parameter Estimation')
    problem = task.getProblem()

    for i in range(parameters.shape[0]):
        current = parameters.iloc[i]
        name = current.parameter
        if 'log10(' in name:
            name = name[name.index('(') + 1: name.rindex(')')]

        obj = dm.findObjectByDisplayName(str('Values[' + name + ']'))

        if obj is None:
            continue

        cn = obj.getInitialValueReference().getCN()

        lower = pow(10.0, float(current['lower boundary']))
        upper = pow(10.0, float(current['upper boundary']))
        value = pow(10.0, float(current['value']))

        # if we found it, get its internal identifier and create the item
        item = problem.addOptItem(cn)
        item.setLowerBound(COPASI.CCommonName(str(lower)))   # set the lower
        item.setUpperBound(COPASI.CCommonName(str(upper)))   # and upper bound
        item.setStartValue(value)                # as well as the initial value


def convert_benchmark(sbml_file, data_file, info_file, output_dir, out_name):
    parameters = get_parameters_from_info_file(info_file)
    data = get_experimental_data(data_file)
    exp_file_name = os.path.join(output_dir,
                                 os.path.splitext(
                                     os.path.basename(out_name))[0] + '.txt')
    data.to_csv(exp_file_name, index=False)

    if not dm.importSBML(sbml_file):
        raise ValueError(COPASI.CCopasiMessage.getAllMessageText())

    dm.saveModel(os.path.join(output_dir, out_name), True)

    # map the experimental data
    create_experimental_data_mapping(data, exp_file_name)

    # now add fit items
    add_fit_items(parameters)

    dm.saveModel(os.path.join(output_dir, out_name), True)
    dm.loadModel(os.path.join(output_dir, out_name))

    # add plot for progress & result
    task = dm.getTask('Parameter Estimation')

    COPASI.COutputAssistant.getListOfDefaultOutputDescriptions(task)

    COPASI.COutputAssistant.createDefaultOutput(913, task, dm)
    COPASI.COutputAssistant.createDefaultOutput(910, task, dm)

    dm.saveModel(os.path.join(output_dir, out_name), True)


if __name__ == "__main__":
    num_args = len(sys.argv)
    if num_args > 5:
        model_file = sys.argv[1]
        data_file = sys.argv[2]
        info_file = sys.argv[3]
        output_dir = sys.argv[4]
        output_name = sys.argv[5]
    else:
        bench_dir = '../benchmarks/Benchmark-Models/Bachmann_MSB2011'
        model_file = bench_dir + '/SBML/model1_data1_l2v4.xml'
        data_file = bench_dir + '/Data/model1_data1.xlsx'
        info_file = bench_dir + '/General_info.xlsx'
        output_dir = '..'
        output_name = 'out_bach1.cps'

    convert_benchmark(model_file, data_file, info_file,
                      output_dir, output_name)

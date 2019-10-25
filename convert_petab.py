import COPASI
import sys
import os
import pandas as pd

dm = COPASI.CRootContainer.addDatamodel()
assert (isinstance(dm, COPASI.CDataModel))
print("using COPASI: %s" % COPASI.CVersion.VERSION.getVersion())


class PEtabProblem:
    def __init__(self, dirname, model_name):
        self.directory = dirname
        self.model_name = model_name

        self.condition_file = self._get_condition_file()
        self.measurement_file = self._get_measurement_file()
        self.parameter_file = self._get_parameter_file()
        self.simulation_file = self._get_simulation_file()
        self.model_file = self._get_model_file()

        self.measurement_data = pd.read_csv(self.measurement_file, sep='\t')
        self.condition_data = pd.read_csv(self.condition_file, sep='\t')
        self.parameter_data = pd.read_csv(self.parameter_file, sep='\t')
        self.simulation_data = None if self.simulation_file is None else pd.read_csv(self.simulation_file, sep='\t')

    def _get_file_from_folder(self, prefix, suffix):
        file_name = str(os.path.join(self.directory, '{0}_{1}.{2}'.format(prefix, self.model_name, suffix)))
        if not os.path.exists(file_name):
            file_name = str(os.path.join(self.directory, '{0}_{1}.{2}'
                                     .format(prefix, os.path.basename(self.directory), suffix)))
            if not os.path.exists(file_name):
                raise ValueError('{0} file missing'.format(prefix))
        return file_name

    def _get_condition_file(self):
        return self._get_file_from_folder('experimentalCondition', 'tsv')

    def _get_measurement_file(self):
        return self._get_file_from_folder('measurementData', 'tsv')

    def _get_parameter_file(self):
        return self._get_file_from_folder('parameters', 'tsv')

    def _get_model_file(self):
        return self._get_file_from_folder('model', 'xml')

    def _get_simulation_file(self):
        try:
            return self._get_file_from_folder('simulationData', 'tsv')
        except:
            return None


class PEtabConverter:

    def __init__(self, petab_dir, model_name, out_dir='.', out_name=None):
        self.petab_dir = petab_dir
        self.model_name = model_name
        self.petab = PEtabProblem(petab_dir, model_name)
        self.experiments = {}
        self.out_dir = out_dir
        self.out_name = out_name
        if out_name is None:
            self.out_name = model_name
        self.experimental_data_file = None

    def get_columns(self, experiments):
        result = []
        for cond in experiments.keys():
            for obs in experiments[cond]['columns'].keys():
                if not obs in result:
                    result.append(obs)
        return result

    def write_headers(self, output, experiments):
        output.write('#time')
        cols = self.get_columns(experiments)
        for col in cols:
            output.write('\t')
            output.write(col)

    def get_num_rows(self, experiment):
        for obs in experiment['columns'].keys():
            return len(experiment['columns'][obs])
        return 0

    def write_experiments(self, experiments, experimental_data_file):
        # type:({}, str) -> None

        # potentially it could be important to write several experiment files, if the time points would not
        # match, but i hope we can avoid it.  the function below will fail in case time is different for the experiments
        # write_all_times = are_times_equal(experiments)

        with open(experimental_data_file, 'w') as output:
            # write header
            self.write_headers(output, experiments)
            line = 0
            for cond in experiments.keys():
                output.write('\n')
                line += 1
                experiments[cond]['offset'] = line
                rows = self.get_num_rows(experiments[cond])
                for i in range(rows):
                    output.write(str(experiments[cond]['time'][i]))
                    output.write('\t')
                    cols = experiments[cond]['columns']
                    col_keys = cols.keys()
                    for col_no in range(len(cols)):
                        if cols[col_keys[col_no]][i][0] != experiments[cond]['time'][i]:
                            raise ValueError('times are different, look into this')
                        output.write(str(cols[col_keys[col_no]][i][1]))
                        if col_no+1 < len(cols):
                            output.write('\t')
                    output.write('\n')
                    line += 1

    def create_mapping(self, experiments, data_file):
        task = dm.getTask('Parameter Estimation')
        problem = task.getProblem()
        exp_set = problem.getExperimentSet()

        for cond in experiments.keys():
            exp = COPASI.CExperiment(dm)
            exp = exp_set.addExperiment(exp)
            info = COPASI.CExperimentFileInfo(exp_set)
            info.setFileName(str(data_file))
            exp.setObjectName(cond)
            exp.setFirstRow(1 if experiments[cond]['offset'] == 1 else experiments[cond]['offset'] + 1 )
            cols = experiments[cond]['columns'].keys()
            exp.setLastRow(experiments[cond]['offset'] + len(experiments[cond]['columns'][cols[0]]) )
            exp.setHeaderRow(1)
            exp.setFileName(str(os.path.basename(data_file)))
            exp.setExperimentType(COPASI.CTaskEnum.Task_timeCourse)
            exp.setSeparator('\t')
            info.sync()
            cols.insert(0, 'time')
            exp.setNumColumns(len(cols))

            # now do the mapping
            map = exp.getObjectMap()
            map.setNumCols(len(cols))
            for i in range(len(cols)):
                type = COPASI.CExperiment.ignore
                if cols[i] == 'time':
                    type = COPASI.CExperiment.time
                else:
                    obj = dm.findObjectByDisplayName('Values[observable_' + cols[i] + ']')
                    if obj is not None:
                        cn = obj.getValueReference().getCN()
                        type = COPASI.CExperiment.dependent
                        map.setRole(i, type)
                        map.setObjectCN(i, cn)
                        exp.calculateWeights()
                        continue
                map.setRole(i, type)

    def generate_copasi_data(self, petab, experimental_data_file):
        print(petab.measurement_file)
        data = petab.measurement_data

        experiments = self.get_experiments(data)
        self.write_experiments(experiments, experimental_data_file)
        self.create_mapping(experiments, experimental_data_file)

    def get_experiments(self, data):
        experiments = {}
        for i in range(data.shape[0]):

            obs = data.observableId[i]
            cond = data.simulationConditionId[i]
            time = data.time[i]
            value = data.measurement[i]
            params = data.observableParameters[i]
            transformation = data.observableTransformation[i] if 'observableTransformation' in data else 'lin'

            if cond not in experiments.keys():
                experiments[cond] = {'name': cond, 'columns': {}, 'time': [], 'offset': 0}

            if obs not in experiments[cond]['columns'].keys():
                experiments[cond]['columns'][obs] = []

            experiments[cond]['columns'][obs].append((time, value, params, transformation))
            experiments[cond]['time'].append(time)
        return experiments

    def add_fit_items(self, parameters):
        task = dm.getTask('Parameter Estimation')
        problem = task.getProblem()

        for i in range(parameters.shape[0]):
            current = parameters.iloc[i]
            name = current.parameterId

            pObj = dm.findObjectByDisplayName(str('Values[' + name + ']'))

            if pObj is None:
                continue

            cn = pObj.getInitialValueReference().getCN()

            if current.parameterScale == 'log10':
                lower = pow(10.0, float(current['lowerBound']))
                upper = pow(10.0, float(current['upperBound']))
                value = pow(10.0, float(current['nominalValue']))
            elif current.parameterScale == 'log':
                import math
                lower = math.exp(float(current['lowerBound']))
                upper = math.exp(float(current['upperBound']))
                value = math.exp(float(current['nominalValue']))
            else:
                lower = float(current['lowerBound'])
                upper = float(current['upperBound'])
                value = float(current['nominalValue'])

            item = problem.addOptItem(cn)  # if we found it, we can get its internal identifier and create the item
            item.setLowerBound(COPASI.CCommonName(str(lower)))  # set the lower
            item.setUpperBound(COPASI.CCommonName(str(upper)))  # and upper bound
            item.setStartValue(value)  # as well as the initial value

    def convert_petab_fromdir(self, petab_dir, model_name, out_dir, out_name):
        petab = PEtabProblem(petab_dir, model_name)
        self.generate_copasi_file(petab, out_dir, out_name)

    def generate_copasi_file(self, petab, out_dir, out_name):
        output_model = str(os.path.join(out_dir, out_name + '.cps'))
        if not dm.importSBML(petab.model_file):
            raise ValueError(COPASI.CCopasiMessage.getAllMessageText())
        dm.saveModel(output_model, True)
        self.experimental_data_file = os.path.join(out_dir, out_name + '.txt')
        self.generate_copasi_data(petab, self.experimental_data_file)
        # now add fit items
        self.add_fit_items(petab.parameter_data)
        dm.saveModel(output_model, True)
        dm.loadModel(output_model)
        # add plot for progress & result
        task = dm.getTask('Parameter Estimation')
        task.setMethodType(COPASI.CTaskEnum.Method_Statistics)
        COPASI.COutputAssistant.getListOfDefaultOutputDescriptions(task)
        COPASI.COutputAssistant.createDefaultOutput(913, task, dm)
        COPASI.COutputAssistant.createDefaultOutput(910, task, dm)
        dm.saveModel(output_model, True)

    def convert(self):
        self.generate_copasi_file(self.petab, self.out_dir, self.out_name)


if __name__ == "__main__":

    num_args = len(sys.argv)

    if num_args > 3:
        petab_dir = sys.argv[1]
        model_name = sys.argv[2]
        out_dir = sys.argv[3]
    else:
        petab_dir = r'E:\Development\Benchmark-Models\hackathon_contributions_new_data_format\Becker_Science2010'
        model_name = r'Becker_Science2010__BaF3_Exp'
        out_dir = '.'

    converter = PEtabConverter(petab_dir, model_name, out_dir)
    converter.convert()

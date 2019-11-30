import COPASI
import sys
import os
import math
import pandas as pd
import numpy as np
import logging

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
        self.experiment_time_points = \
            self._get_time_points(self.measurement_data)
        self.condition_data = pd.read_csv(self.condition_file, sep='\t')
        self.independent_columns =\
            self._get_independent_columns(self.condition_data)
        self.parameter_data = pd.read_csv(self.parameter_file, sep='\t')
        self.simulation_data = None if self.simulation_file is None \
            else pd.read_csv(self.simulation_file, sep='\t')

    @staticmethod
    def _get_time_points(data_set):
        # type: (pd.DataFrame) -> {}
        result = {}
        for i in range(data_set.shape[0]):
            current = data_set.iloc[i]
            exp = current.simulationConditionId
            if exp not in result:
                result[exp] = {}
            if current.time not in result[exp]:
                result[exp][current.time] = {}
            obs = current.observableId
            if obs not in result[exp][current.time]:
                result[exp][current.time][obs] = 0
            result[exp][current.time][obs] += 1
        for key in result:
            current = result[key]
            times = []
            for point in current:
                replicates = max(1, max(current[point].values()))
                for i in range(replicates):
                    times.append(point)
            result[key] = sorted(times)
        return result

    @staticmethod
    def _get_independent_columns(data_set):
        # type: (pd.DataFrame) -> []
        result = []
        for col in data_set.columns:
            if col == 'conditionId' or col == 'conditionName':
                continue
            result.append(col)
        return result

    def _get_file_from_folder(self, prefix, suffix):
        file_name = str(os.path.join(self.directory,
                                     '{0}_{1}.{2}'.format(
                                         prefix, self.model_name, suffix)))
        if not os.path.exists(file_name):
            try1 = file_name
            basename = os.path.basename(self.directory)
            file_name = str(os.path.join(self.directory, '{0}_{1}.{2}'
                                         .format(prefix, basename, suffix)))
            if not os.path.exists(file_name):
                raise ValueError('{0} file missing, looked for\n{1} and \n{2}'.
                                 format(prefix, try1, file_name))
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
        except ValueError:
            return None


class PEtabConverter:

    def __init__(self, petab_dir, model_name, out_dir='.', out_name=None):
        self.petab_dir = petab_dir
        self.model_name = model_name
        self.petab = PEtabProblem(petab_dir, model_name)
        self.experiments = {}
        self.out_dir = out_dir
        self.out_name = out_name
        self.transform_data = False
        self.ignore_independent = {}
        self.experiment_to_key = {}
        if out_name is None:
            self.out_name = model_name
        self.experimental_data_file = None

    @staticmethod
    def get_columns(experiments):
        # type: ({}) -> []
        result = []
        for cond in experiments.keys():
            current = experiments[cond]
            for obs in current['columns'].keys():
                if obs not in result:
                    result.append(obs)
        return result

    def write_headers(self, output, experiments):
        output.write('#time')
        cols = self.get_columns(experiments)
        for col in cols:
            output.write('\t')
            output.write(col)

    @staticmethod
    def get_num_rows(experiment):
        for obs in experiment['columns'].keys():
            return len(experiment['columns'][obs])
        return 0

    def write_experiments(self, experiments, petab):
        # type:({}, PEtabProblem) -> None

        with open(self.experimental_data_file, 'w') as output:
            # write header
            self.write_headers(output, experiments)
            # append independent columns
            for col in petab.independent_columns:
                output.write('\t%s' % col)

            all_cols = self.get_columns(experiments)
            line = 0
            for cond in experiments.keys():
                output.write('\n')
                line += 1
                current_exp = experiments[cond]
                current_exp['offset'] = line
                times = petab.experiment_time_points[cond]
                rows = len(times)  # self.get_num_rows(current_exp)
                for i in range(rows):
                    output.write(str(times[i]))
                    output.write('\t')
                    cols = current_exp['columns']
                    for col_no in range(len(all_cols)):
                        if not all_cols[col_no] in cols:
                            if col_no + 1 < len(all_cols):
                                output.write('\t')
                            continue
                        vals = cols[all_cols[col_no]]
                        cur_val = vals[i]
                        if cur_val[0] == current_exp['time'][i]:
                            output.write(str(cur_val[1]))
                        if col_no + 1 < len(all_cols):
                            output.write('\t')

                    # append independents for first row
                    if i == 0 and 'conditionId' in petab.condition_data:
                        conditions = petab.condition_data[
                            petab.condition_data.conditionId ==
                            current_exp['condition']]
                        for col in petab.independent_columns:
                            output.write('\t')
                            try:
                                output.write(str(float(conditions[col])))
                            except ValueError:
                                # output.write(str(conditions[col]))
                                if not current_exp['condition'] in self.ignore_independent:
                                    self.ignore_independent[current_exp['condition']] = {}
                                self.ignore_independent[current_exp['condition']][col] =\
                                    conditions[col].values[0]
                    output.write('\n')
                    line += 1

    def create_mapping(self, experiments, petab):
        task = dm.getTask('Parameter Estimation')
        # mark task as executable, so it can be run by copasi se
        task.setScheduled(True)
        problem = task.getProblem()
        # disable statistics at the end of the runs
        problem.setCalculateStatistics(False)
        exp_set = problem.getExperimentSet()

        all_cols = self.get_columns(experiments)

        for cond in experiments.keys():
            cur_exp = experiments[cond]
            if len(cur_exp['time']) == 0:
                continue  # no data return

            exp = COPASI.CExperiment(dm)
            exp = exp_set.addExperiment(exp)
            info = COPASI.CExperimentFileInfo(exp_set)
            info.setFileName(str(self.experimental_data_file))
            exp.setObjectName(cond)
            self.experiment_to_key[cond] = exp.getKey()
            exp.setFirstRow(1 if cur_exp['offset'] == 1
                            else cur_exp['offset'] + 1)
            cols = all_cols
            times = petab.experiment_time_points[cur_exp['condition']]
            exp.setLastRow(cur_exp['offset'] + len(times))
            exp.setHeaderRow(1)
            exp.setFileName(str(os.path.basename(self.experimental_data_file)))
            is_tc = True
            if np.isinf(times[0]):
                exp.setExperimentType(COPASI.CTaskEnum.Task_steadyState)
                is_tc = False
            else:
                exp.setExperimentType(COPASI.CTaskEnum.Task_timeCourse)
            exp.setSeparator('\t')
            info.sync()
            if 'time' not in cols:
                cols.insert(0, 'time')

            cond_cols = petab.independent_columns
            num_conditions = len(cond_cols)

            num_cols = len(cols)
            exp.setNumColumns(num_cols + num_conditions)

            # now do the mapping
            obj_map = exp.getObjectMap()
            obj_map.setNumCols(num_cols + num_conditions)
            for i in range(num_cols):
                role = COPASI.CExperiment.ignore
                if cols[i] == 'time' and is_tc:
                    role = COPASI.CExperiment.time
                else:
                    if all_cols[i] in cur_exp['columns']:
                        obj = dm.findObjectByDisplayName(
                            'Values[observable_' + all_cols[i] + ']')
                        if obj is None:
                            obj = dm.findObjectByDisplayName(
                                'Values[' + all_cols[i] + ']')
                        if obj is not None:
                            cn = obj.getValueReference().getCN()
                            role = COPASI.CExperiment.dependent
                            obj_map.setRole(i, role)
                            obj_map.setObjectCN(i, cn)
                            exp.calculateWeights()
                            continue
                obj_map.setRole(i, role)

            for i in range(num_conditions):
                role = COPASI.CExperiment.ignore
                obj = dm.findObjectByDisplayName(
                    'Values[' + cond_cols[i] + ']')

                if cond in self.ignore_independent and cond_cols[i] in self.ignore_independent[cond]:
                    # ignore mapping to parameter / fit item
                    obj = None

                if obj is not None:
                    cn = obj.getInitialValueReference().getCN()
                    role = COPASI.CExperiment.independent
                    obj_map.setRole(num_cols + i, role)
                    obj_map.setObjectCN(num_cols + i, cn)
                    continue
                obj_map.setRole(num_cols + i, role)

    def generate_copasi_data(self, petab):

        self.experiments = self.get_experiments(petab)
        self.write_experiments(self.experiments, petab)
        self.create_mapping(self.experiments, petab)

    def get_experiments(self, petab):
        data = petab.measurement_data
        experiments = {}
        for i in range(data.shape[0]):

            obs = data.observableId[i]
            cond = data.simulationConditionId[i]
            time = data.time[i]
            value = data.measurement[i]
            params = data.observableParameters[i] \
                if 'observableParameters' in data else None
            transformation = data.observableTransformation[i] \
                if 'observableTransformation' in data else 'lin'
            condition = data.simulationConditionId[i] \
                if 'simulationConditionId' in data else None

            if self.transform_data and transformation == 'log10':
                value = math.pow(10.0, float(value))
            elif self.transform_data and transformation == 'log':
                value = math.exp(float(value))

            if cond not in experiments.keys():
                experiments[cond] = {'name': cond,
                                     'columns': {},
                                     'time': [],
                                     'offset': 0,
                                     'condition': condition}

            cur_exp = experiments[cond]
            if obs not in cur_exp['columns'].keys():
                cur_exp['columns'][obs] = []
                # add transformations only once
                self.add_transformation_for_params(obs, params)

            cur_exp['columns'][obs].\
                append((time, value, params, transformation))

        for cond in experiments:
            cur_exp = experiments[cond]
            times = petab.experiment_time_points[cond]
            cur_exp['time'] = times
            for obs in cur_exp['columns']:
                cur_obs = cur_exp['columns'][obs]
                cur_obs = self.sort_tuples(cur_obs)
                padded_obs = self.pad_data(times, cur_obs)
                experiments[cond]['columns'][obs] = padded_obs

        # pad columns
        return experiments

    @staticmethod
    def sort_tuples(data):
        data.sort(key=lambda tup: tup[0])
        return data

    @staticmethod
    def pad_data(times, data):
        # type: ([],[()]) -> []
        result = []
        index = 0
        num_entries = len(data)
        for time in times:
            if index >= num_entries:
                dummy = data[0]
                result.append((time, np.nan, dummy[2], dummy[3]))
                continue
            current = data[index]
            cur_time = current[0]
            if cur_time == time:
                result.append(current)
                index = index + 1
            else:
                dummy = data[0]
                result.append((time, np.nan, dummy[2], dummy[3]))
                pass
        return result

    def add_fit_items(self, parameters):
        task = dm.getTask('Parameter Estimation')
        problem = task.getProblem()

        for i in range(parameters.shape[0]):
            current = parameters.iloc[i]
            name = current.parameterId
            estimate = 1 if 'estimate' not in current else current['estimate']

            if current.parameterScale == 'log10':
                lower = pow(10.0, float(current['lowerBound']))
                upper = pow(10.0, float(current['upperBound']))
                value = pow(10.0, float(current['nominalValue']))
            elif current.parameterScale == 'log':
                lower = math.exp(float(current['lowerBound']))
                upper = math.exp(float(current['upperBound']))
                value = math.exp(float(current['nominalValue']))
            else:
                lower = float(current['lowerBound'])
                upper = float(current['upperBound'])
                value = float(current['nominalValue'])

            obj = dm.findObjectByDisplayName(str('Values[' + name + ']'))
            if estimate == 0:
                # update the initial value but don't create an item for it
                if obj is not None:
                    obj.setInitialValue(value)
                continue

            if obj is None:
                # if there is no such parameter in the model, we might have 
                # to create it first
                model = dm.getModel()
                obj = model.createModelValue(name, value)
                logging.debug('created model value {0} for fit item'.format(name))

            # update the initial value
            obj.setInitialValue(value)

            cn = obj.getInitialValueReference().getCN()

            # if we found it, we can get its internal identifier and create
            # the item
            item = problem.addOptItem(cn)
            if np.isnan(lower):
                lower = math.pow(10, -6)
            item.setLowerBound(COPASI.CCommonName(str(lower)))
            if np.isnan(upper):
                upper = math.pow(10, 6)
            item.setUpperBound(COPASI.CCommonName(str(upper)))
            item.setStartValue(value)  # as well as the initial value

        # create experiment specific fit items
        for condition in self.ignore_independent:
            for name in self.ignore_independent[condition]:
                parameterId = self.ignore_independent[condition][name]
                obj = dm.findObjectByDisplayName(str('Values[' + name + ']'))
                if obj is None:
                    logging.warning(
                        'No model value for {0} to create fit item for'.
                            format(name))
                    continue

                current = parameters[parameters.parameterId == parameterId]
                if current.shape[0] == 0:
                    # this should not be happening
                    logging.warning('No entry for {0} in parameter table'.
                                    format(parameterId))
                    continue
                current = current.iloc[0]
                if current.parameterScale == 'log10':
                    lower = pow(10.0, float(current['lowerBound']))
                    upper = pow(10.0, float(current['upperBound']))
                    value = pow(10.0, float(current['nominalValue']))
                elif current.parameterScale == 'log':
                    lower = math.exp(float(current['lowerBound']))
                    upper = math.exp(float(current['upperBound']))
                    value = math.exp(float(current['nominalValue']))
                else:
                    lower = float(current['lowerBound'])
                    upper = float(current['upperBound'])
                    value = float(current['nominalValue'])

                cn = obj.getInitialValueReference().getCN()

                # if we found it, we can get its internal identifier and create
                # the item
                item = problem.addFitItem(cn)
                if np.isnan(lower):
                    lower = math.pow(10, -6)
                item.setLowerBound(COPASI.CCommonName(str(lower)))
                if np.isnan(upper):
                    upper = math.pow(10, 6)
                item.setUpperBound(COPASI.CCommonName(str(upper)))
                item.setStartValue(value)  # as well as the initial value
                item.addExperiment(self.experiment_to_key[condition])

    def convert_petab_fromdir(self, petab_dir, model_name, out_dir, out_name):
        petab = PEtabProblem(petab_dir, model_name)
        self.generate_copasi_file(petab, out_dir, out_name)

    def generate_copasi_file(self, petab, out_dir, out_name):
        output_model = str(os.path.join(out_dir, out_name + '.cps'))
        if not dm.importSBML(petab.model_file):
            raise ValueError(COPASI.CCopasiMessage.getAllMessageText())
        dm.saveModel(output_model, True)
        self.experimental_data_file = os.path.join(out_dir, out_name + '.txt')
        self.generate_copasi_data(petab)
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
        dm.exportCombineArchive(str(os.path.join(out_dir, out_name + '.omex')),
                                True, False, True, False, True)

    def convert(self):
        self.generate_copasi_file(self.petab, self.out_dir, self.out_name)

    def add_transformation_for_params(self, obs, params):
        # type: (str, str) -> None
        """ add assignment rules to observable parameters """
        count = 0

        if np.isreal(params):
            if not not np.isnan(params):
                self.add_value_transform(params, 1, obs)
            return

        for param in params.split(';'):
            count += 1
            obj = dm.findObjectByDisplayName('Values[' + param + ']')
            if obj is None:
                # it could be a value
                if self.add_value_transform(param, count, obs):
                    continue
                # otherwise add it as model value and try mapping
                obj = dm.getModel().createModelValue(param, 1.0)
                logging.debug('created model value {0} for transformation'.
                              format(param))

            obs_param = dm.findObjectByDisplayName(
                'Values[observableParameter{0}_{1}]'.format(count, obs))
            if obs_param is None or \
                    not isinstance(obs_param, COPASI.CModelValue):
                continue
            obs_param.setStatus(COPASI.CModelValue.Status_ASSIGNMENT)
            obs_param.setExpression('<{0}>'.format(obj.getCN()))

    @staticmethod
    def add_value_transform(value, index, obs):
        if np.isreal(value):
            obs_param = dm.findObjectByDisplayName(
                'Values[observableParameter{0}_{1}]'.format(index, obs))
            if obs_param is None or \
                    not isinstance(obs_param, COPASI.CModelValue):
                return False
            obs_param.setValue(value)
            return True
        return False


if __name__ == "__main__":

    num_args = len(sys.argv)

    if num_args > 3:
        petab_dir = sys.argv[1]
        model_name = sys.argv[2]
        out_dir = sys.argv[3]
    else:
        petab_dir = './benchmarks/hackathon_contributions_new_data_format/' \
                    'Becker_Science2010'
        model_name = 'Becker_Science2010__BaF3_Exp'
        out_dir = '.'

    converter = PEtabConverter(petab_dir, model_name, out_dir)
    converter.convert()

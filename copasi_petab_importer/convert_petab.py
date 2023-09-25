import COPASI
import sys
import os
import math
import pandas as pd
import numpy as np
import logging
import libsbml

dm = COPASI.CRootContainer.addDatamodel()
assert (isinstance(dm, COPASI.CDataModel))

logger = logging.getLogger(__name__)
logger.info("using COPASI: %s" % COPASI.CVersion.VERSION.getVersion())


class PEtabProblem:
    def __init__(self, dirname=None, model_name=None, condition_file=None, measurement_file=None,
                 parameter_file=None, simulation_file=None, model_file=None, observable_file=None, yaml_file=None):
        self.directory = dirname
        self.model_name = model_name

        self.condition_file = condition_file
        self.measurement_file = measurement_file
        self.parameter_file = parameter_file
        self.simulation_file = simulation_file
        self.model_file = model_file
        self.observable_file = observable_file
        self.yaml_file = yaml_file

        if self.yaml_file is None:
            self.yaml_file = self._get_yaml_file()

        if self.yaml_file is not None and os.path.exists(self.yaml_file):
            self._init_from_yaml()

        if self.model_name is None and self.directory is not None:
            self.model_name = os.path.basename(self.directory)

        if self.condition_file is None:
            self.condition_file = self._get_condition_file()
        if self.measurement_file is None:
            self.measurement_file = self._get_measurement_file()
        if self.parameter_file is None:
            self.parameter_file = self._get_parameter_file()
        if self.simulation_file is None:
            self.simulation_file = self._get_simulation_file()
        if self.model_file is None:
            self.model_file = self._get_model_file()
        if self.observable_file is None:
            self.observable_file = self._get_observable_file()

        self.transformed_sbml = None
        self._init_from_files()

    @staticmethod
    def from_folder(dirname, model_name=None):
        if model_name is None:
            model_name = os.path.basename(dirname)
        return PEtabProblem(dirname, model_name)

    @staticmethod
    def from_yaml(filename):
        return PEtabProblem(yaml_file=filename)

    def _init_from_files(self):
        self.measurement_data = pd.read_csv(self.measurement_file, sep='\t')
        self.experiment_time_points = \
            self._get_time_points(self.measurement_data)
        self.condition_data = pd.read_csv(self.condition_file, sep='\t')
        self.independent_columns = \
            self._get_independent_columns(self.condition_data)
        self.parameter_data = pd.read_csv(self.parameter_file, sep='\t')
        self.simulation_data = None if self.simulation_file is None \
            else pd.read_csv(self.simulation_file, sep='\t')
        self.observable_data = None if self.observable_file is None \
            else pd.read_csv(self.observable_file, sep='\t')
        self.transformed_sbml = None
        self.transform_model(self.observable_data)

    def transform_model(self, observable_data):
        if observable_data is None:
            # just read file and leave it as is
            with open(self.model_file, 'r') as sbml:
                self.transformed_sbml = sbml.read()
            return

        doc = libsbml.readSBMLFromFile(self.model_file)
        model = doc.getModel()

        all_ids = [x.getId() for x in doc.getListOfAllElements() if x.isSetId()]

        for i in range(observable_data.shape[0]):
            current = observable_data.iloc[i]
            id = current.observableId

            # ignore invalid ids
            if not libsbml.SyntaxChecker.isValidSBMLSId(id):
                logger.warning(
                    'Invalid observableId {0} in observable table'.
                    format(id))
                continue

            name = id

            # if we have one already, don't add again
            # something ought to be wrong with the table here
            if id in all_ids:
                logger.warning(
                    'observableId {0} appears already in the model'.
                    format(id))
                continue

            formula = current.observableFormula
            if not isinstance(formula, str):
                logger.warning(
                    'Invalid observableFormula for observableId {0}'.
                    format(id))
                continue

            if 'observableTransformation' in current:
                transformation = current.observableTransformation
                if not isinstance(transformation, str):
                    logger.warning(
                        'Invalid observableTransformation for observableId {0} default to lin'.
                        format(id))
                    transformation = 'lin'
            else:
                transformation = 'lin'

            if transformation == 'log':
                formula = 'ln({0})'.format(formula)
            elif transformation == 'log10':
                formula = 'log({0})'.format(formula)

            math = libsbml.parseL3Formula(formula)
            if math is None:
                logger.warning(
                    'Invalid observableFormula for observableId {0}'.
                    format(id))
                continue

            self.add_missing_params(model, math)

            obs = model.createParameter()
            obs.setId(id)
            obs.setName(name)
            obs.setValue(0)

            assignment = model.createAssignmentRule()
            assignment.setVariable(current.observableId)
            assignment.setMath(math)

            all_ids.append(id)

        self.transformed_sbml = libsbml.writeSBMLToString(doc)
        del doc

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
            if col == 'conditionId' or col == 'conditionName' or \
                    col == 'conditionID':
                continue
            result.append(col)
        return result

    def _get_file_from_folder(self, prefix, suffix):
        if self.directory is None:
            return None

        if self.model_name is None:
            return None

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

    def _get_observable_file(self):
        try:
            return self._get_file_from_folder('observables', 'tsv')
        except ValueError:
            return None

    def _get_yaml_file(self):
        if self.directory is None:
            return None
        if self.model_name is None:
            for filename in os.listdir(self.directory):
                if filename.endswith('.yaml'):
                    return os.path.join(self.directory, filename)
            return None
        filename = os.path.join(self.directory, self.model_name + '.yaml')
        if os.path.exists(filename):
            return filename
        return None

    def add_missing_params(self, model, math):
        if model is None or math is None:
            return
        if math.getType() == libsbml.AST_NAME:
            id = math.getName()
            param = model.getElementBySId(id)
            if param is None:
                param = model.createParameter()
                param.setId(id)
                param.setValue(1)
        for i in range(math.getNumChildren()):
            self.add_missing_params(model, math.getChild(i))

    def _init_from_yaml(self):
        import yaml
        with open(self.yaml_file) as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        self.directory = os.path.abspath(os.path.dirname(os.path.abspath(self.yaml_file)))
        self.model_name = os.path.splitext(os.path.basename(self.yaml_file))[0]
        self.parameter_file = self._get_file(data['parameter_file'])
        problem = data['problems'][0]
        self.condition_file = self._get_file(problem['condition_files'][0])
        self.measurement_file = self._get_file(problem['measurement_files'][0])
        self.observable_file = self._get_file(problem['observable_files'][0])
        self.model_file = self._get_file(problem['sbml_files'][0])

    def _get_file(self, filename):
        if not os.path.exists(filename):
            return os.path.join(self.directory, filename)
        return filename


class PEtabConverter:

    def __init__(self, petab_dir, model_name, out_dir='.', out_name=None):
        self.petab_dir = petab_dir
        self.model_name = model_name
        if model_name.endswith('.yaml') or model_name.endswith('.yml'):
            self.petab = PEtabProblem.from_yaml(os.path.join(petab_dir, model_name))
            self.model_name = os.path.splitext(os.path.basename(model_name))[0]
        else:
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

        self.show_progress_of_fit = True
        self.show_result = True
        self.show_result_per_experiment = False
        self.show_result_per_dependent = False

        self.save_report = True
        self.copasi_file = None
        self.copasi_file_omex = None

    @staticmethod
    def from_yaml(filename, out_dir='.', out_name=None):
        """Initializes a converter object from the given yaml file

        :param filename: the PEtab yaml file
        :type filename: str

        :param out_dir: optional output directory (defaults to '.')
        :type out_dir: str
        :param out_name: optional name of the output copasi file. Defaults to the basename of the filename + .cps
               in the output directory
        :type out_name: str or None
        :return: the initialized converter
        :rtype: PEtabConverter
        """
        petab_dir = os.path.dirname(filename)
        model_name = os.path.basename(filename)
        return PEtabConverter(petab_dir, model_name, out_dir, out_name)

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
                                output.write(str(float(conditions[col].iloc[0])))
                            except ValueError:
                                # output.write(str(conditions[col]))
                                if not current_exp['condition'] in self.ignore_independent:
                                    self.ignore_independent[current_exp['condition']] = {}
                                self.ignore_independent[current_exp['condition']][col] = \
                                    conditions[col].values[0]
                    output.write('\n')
                    line += 1

    def assign_weight(self, petab, observable, obj_map, i):
        """ retrieves the weight from the petab problem for the given column
        """

        noise_formula = petab.observable_data.query('observableId == "' + observable + '"').noiseFormula.values[0]

        # check if it is a number if so use it
        try:
            value = float(noise_formula)
            obj_map.setScale(i, value)
            return True
        except ValueError:
            pass

        # otherwise we search for it in the parameter table
        parameter = petab.parameter_data.query('parameterId == "' + noise_formula + '"')

        if len(parameter) == 0:
            # check whether the noise formula is a parameter in the model
            mv = dm.getModel().getModelValue(noise_formula)
            if mv is not None:
                # if so, we use it
                obj_map.setScale(i, mv.getInitialValue())
                return True

            # if the noise formula starts with 'noiseParameter' and a number, we should
            # be taking the value from the noiseParameters column in the measurement
            # table
            if noise_formula.startswith('noiseParameter'):
                # strip the noiseParameter part
                noise_formula = noise_formula[14:]
                # find the index until the next '_'
                index = noise_formula.find('_')
                if index > 0:
                    # get the number
                    number = int(noise_formula[0:index])
                    # get the observable name
                    observable_name = noise_formula[index + 1:]
                    # get the noiseParameters
                    noise_parameter = petab.measurement_data.query('observableId == "' + observable_name + '"') \
                                      .noiseParameters.values
                    if len(noise_parameter) == 0:
                        return False

                    try:
                        noise_parameter = noise_parameter[0].split(';')
                    except AttributeError:
                        try:
                            noise_parameter = float(noise_parameter[0])
                            noise_parameter = [ noise_parameter]
                        except ValueError:
                            return False

                    if len(noise_parameter) >= number:
                        # get the value from the column
                        noise_parameter = noise_parameter[number - 1]
                        # now that we have the parameter check whether it is a number
                        try:
                            value = float(noise_parameter)
                            obj_map.setScale(i, value)
                            return True
                        except ValueError:
                            pass
                        # otherwise look for it in the parameter table
                        parameter = petab.parameter_data.query('parameterId == "' + noise_parameter + '"')
                        return self._assign_weight_from_parameter(parameter, noise_parameter, obj_map, i)
            logger.debug('Unsupported noise formula %s. Ignoring it.' % noise_formula)
            return False

        return self._assign_weight_from_parameter(parameter, noise_formula, obj_map, i)

    def _assign_weight_from_parameter(self, parameter, noise_formula, obj_map, i):
        if len(parameter) == 0:
            return False
        if parameter.iloc[0].estimate == 1:
            logger.debug('COPASI cannot estimate the noise parameter %s. Ignoring it.' % noise_formula)
            return False
        # take the nominal value
        try:
            value = float(parameter.iloc[0].nominalValue)
            obj_map.setScale(i, value)
            return True
        except ValueError:
            pass
        return False

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
                tc = dm.getTask('Time-Course')
                assert (isinstance(tc, COPASI.CTrajectoryTask))
                p = tc.getProblem()
                assert (isinstance(p, COPASI.CTrajectoryProblem))
                pre_equilib = cur_exp['preequilibrationCondition']
                if pre_equilib is not None:
                    is_empty = np.isreal(pre_equilib) and np.isnan(pre_equilib)
                    p.setStartInSteadyState(not is_empty)
                else:
                    p.setStartInSteadyState(False)
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
                            if not self.assign_weight(petab, all_cols[i], obj_map, i):
                                exp.calculateWeights()
                            # exp.calculateWeights()
                            continue
                obj_map.setRole(i, role)

            for i in range(num_conditions):
                role = COPASI.CExperiment.ignore
                obj = dm.findObjectByDisplayName(
                    'Values[' + cond_cols[i] + ']')
                if obj is None:
                    obj = dm.findObjectByDisplayName(cond_cols[i])

                if cond in self.ignore_independent and cond_cols[i] in self.ignore_independent[cond]:
                    # ignore mapping to parameter / fit item
                    obj = None

                if obj is not None:
                    if isinstance(obj, COPASI.CMetab):
                        cn = obj.getInitialConcentrationReference().getCN()
                    else:
                        cn = obj.getInitialValueReference().getCN()

                    # if value is set to NaN, it is not supposed to be used as condition, but actually to be estimated
                    cond_value = float(petab.condition_data.loc[petab.condition_data.conditionId == cond][cond_cols[i]].iloc[0])
                    if not np.isnan(cond_value):
                        # now map to independent, as the value is not NaN
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

            try:
                transformation = \
                petab.observable_data.loc[petab.observable_data.observableId == obs]['observableTransformation'].values[
                    0]
                if transformation == 'log':
                    value = math.log(float(value))
                elif transformation == 'log10':
                    value = math.log10(float(value))
            except IndexError:
                transformation = data.observableTransformation[i] \
                    if 'observableTransformation' in data else 'lin'
                pass
            except KeyError:
                transformation = data.observableTransformation[i] \
                    if 'observableTransformation' in data else 'lin'
                pass

            condition = data.simulationConditionId[i] \
                if 'simulationConditionId' in data else None
            preequilibrationCondition = data.preequilibrationConditionId[i] \
                if 'preequilibrationConditionId' in data else None

            if cond not in experiments.keys():
                experiments[cond] = {'name': cond,
                                     'columns': {},
                                     'time': [],
                                     'offset': 0,
                                     'condition': condition,
                                     'preequilibrationCondition': preequilibrationCondition}

            cur_exp = experiments[cond]
            if obs not in cur_exp['columns'].keys():
                cur_exp['columns'][obs] = []
                # add transformations only once
                self.add_transformation_for_params(obs, params)

            cur_exp['columns'][obs]. \
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

            # if current.parameterScale == 'log10':
            #     lower = pow(10.0, float(current['lowerBound']))
            #     upper = pow(10.0, float(current['upperBound']))
            #     value = pow(10.0, float(current['nominalValue']))
            # elif current.parameterScale == 'log':
            #     lower = math.exp(float(current['lowerBound']))
            #     upper = math.exp(float(current['upperBound']))
            #     value = math.exp(float(current['nominalValue']))
            # else:
            lower = float(current['lowerBound'])
            upper = float(current['upperBound'])
            value = float(current['nominalValue'])

            obj = dm.findObjectByDisplayName(str('Values[' + name + ']'))
            if obj is None and 'parameterName' in current:
                obj = dm.findObjectByDisplayName(str('Values[' + current.parameterName + ']'))
            if estimate == 0:
                # update the initial value but don't create an item for it
                if obj is not None:
                    obj.setInitialValue(value)
                else:
                    # check whether this is used in a noiseFormula
                    if not self.petab.observable_data.noiseFormula.str.contains(name).any():
                        # otherwise raise warning since we could not resolve object
                        logger.warning('could not resolve object {0} for fit item'.format(name))
                continue

            if obj is None:
                # if there is no such parameter in the model, we might have 
                # to create it first
                model = dm.getModel()
                obj = model.createModelValue(name, value)
                logger.debug('created model value {0} for fit item'.format(name))

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
                    obj = dm.findObjectByDisplayName(name)
                if obj is None:
                    logger.warning(
                        'No model value for {0} to create fit item for'.
                        format(name))
                    continue

                current = parameters[parameters.parameterId == parameterId]
                if current.shape[0] == 0:
                    # this should not be happening
                    logger.warning('No entry for {0} in parameter table'.
                                   format(parameterId))
                    continue
                current = current.iloc[0]
                # if current.parameterScale == 'log10':
                #     lower = pow(10.0, float(current['lowerBound']))
                #     upper = pow(10.0, float(current['upperBound']))
                #     value = pow(10.0, float(current['nominalValue']))
                # elif current.parameterScale == 'log':
                #     lower = math.exp(float(current['lowerBound']))
                #     upper = math.exp(float(current['upperBound']))
                #     value = math.exp(float(current['nominalValue']))
                # else:
                lower = float(current['lowerBound'])
                upper = float(current['upperBound'])
                value = float(current['nominalValue'])

                if isinstance(obj, COPASI.CMetab):
                    cn = obj.getInitialConcentrationReference().getCN()
                else:
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
        try:
            if not dm.importSBMLFromString(petab.transformed_sbml):
                raise ValueError(COPASI.CCopasiMessage.getAllMessageText())
        except COPASI.CCopasiException:
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
        if self.show_result:
            COPASI.COutputAssistant.createDefaultOutput(910, task, dm)
        if self.show_result_per_experiment:
            COPASI.COutputAssistant.createDefaultOutput(911, task, dm)
        if self.show_result_per_dependent:
            COPASI.COutputAssistant.createDefaultOutput(912, task, dm)
        if self.show_progress_of_fit:
            COPASI.COutputAssistant.createDefaultOutput(913, task, dm)

        if self.save_report:
            report = task.getReport()
            assert (isinstance(report, COPASI.CReport))
            report.setConfirmOverwrite(False)
            report.setAppend(False)
            report.setTarget(str(out_name + '_report.txt'))

        dm.saveModel(output_model, True)
        dm.exportCombineArchive(str(os.path.join(out_dir, out_name + '.omex')),
                                True, False, True, False, True)
        self.copasi_file = output_model
        self.copasi_file_omex = str(os.path.join(out_dir, out_name + '.omex'))

    def convert(self):
        """performs the conversion

        assumes, that the problem `petab`, `out_dir` and `out_name` has already been set

        :return: None
        """

        self.generate_copasi_file(self.petab, self.out_dir, self.out_name)

    def add_transformation_for_params(self, obs, params, transformation='lin'):
        # type: (str, str, str) -> None
        """ add assignment rules to observable parameters """
        count = 0

        if params is None:
            return

        if np.isreal(params):
            if not np.isnan(params):
                self.add_value_transform(params, 1, obs, transformation)
            return

        for param in params.split(';'):
            count += 1
            obj = dm.findObjectByDisplayName('Values[' + param + ']')
            if obj is None:
                # it could be a value
                if self.add_value_transform(param, count, obs, transformation):
                    continue
                # otherwise add it as model value and try mapping
                obj = dm.getModel().createModelValue(param, 1.0)
                logger.debug('created model value {0} for transformation'.
                             format(param))

            obs_param = dm.findObjectByDisplayName(
                'Values[observableParameter{0}_{1}]'.format(count, obs))
            if obs_param is None or \
                    not isinstance(obs_param, COPASI.CModelValue):
                continue
            obs_param.setStatus(COPASI.CModelValue.Status_ASSIGNMENT)
            if transformation == 'log':
                obs_param.setExpression('ln(<{0}>)'.format(obj.getCN()))
            elif transformation == 'log10':
                obs_param.setExpression('log(<{0}>)'.format(obj.getCN()))
            else:
                obs_param.setExpression('<{0}>'.format(obj.getCN()))

    @staticmethod
    def add_value_transform(value, index, obs, transformation='lin'):
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return False
        if np.isreal(value):
            obs_param = dm.findObjectByDisplayName(
                'Values[observableParameter{0}_{1}]'.format(index, obs))
            if obs_param is None or \
                    not isinstance(obs_param, COPASI.CModelValue):
                return False
            if transformation == 'log':
                try:
                    value = math.log(float(value))
                except ValueError:
                    logger.warning('encountered value {0} for log transform for measurement of observable {1}'.format(
                        value, obs))
                    value = 0
            elif transformation == 'log10':
                try:
                    value = math.log10(float(value))
                except ValueError:
                    logger.warning('encountered value {0} for log transform for measurement of observable {1}'.format(
                        value, obs))
                    value = 0
            else:
                value = float(value)
            obs_param.setValue(value)
            obs_param.setInitialValue(value)
            return True
        return False


def main():
    num_args = len(sys.argv)

    if num_args > 2:
        filename = os.path.abspath(sys.argv[1])
        if not (filename.endswith('yml') or filename.endswith('yaml')):
            print('Only yml files supported, or directory / model name')
            sys.exit(1)
        out_dir = sys.argv[2]
        converter = PEtabConverter.from_yaml(filename, out_dir)
    elif num_args > 3:
        petab_dir = sys.argv[1]
        model_name = sys.argv[2]
        out_dir = sys.argv[3]
        converter = PEtabConverter(petab_dir, model_name, out_dir)
    else:
        print('usage: copasi_petab_import [<petab_dir>]  <model_name> <output_dir>')
        # petab_dir = './benchmarks/hackathon_contributions_new_data_format/' \
        #             'Becker_Science2010'
        # model_name = 'Becker_Science2010__BaF3_Exp'
        # out_dir = '.'
        sys.exit(1)
        return

    converter.convert()


if __name__ == "__main__":
    main()

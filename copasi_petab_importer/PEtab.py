import os
import sys
import traceback
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication, \
    QFileDialog, QMessageBox
from PyQt5 import uic
from PyQt5.QtCore import QUrl, QSettings, Qt
from PyQt5.QtGui import QDesktopServices

from copasi_petab_importer import convert_petab


class PETabGui(QMainWindow):

    def __init__(self):
        # super().__init__()
        QMainWindow.__init__(self)

        self.dir = None
        self.model_dir = None
        self.model = None
        self.out_dir = None
        self.show_progress = True
        self.show_result = True
        self.show_result_per_experiment = False
        self.show_result_per_dependent = False
        self.write_report = False

        self.ui = uic.loadUi(os.path.join(os.path.dirname(__file__), 'petab.ui'), self)

        self.center()
        self.load_settings()
        self.load_model_dirs()
        self.show()

    def closeEvent(self, event):
        self.save_settings()

    @staticmethod
    def _get_user_dir():
        home = os.getenv("HOME")
        if home is not None:
            return home
        from pathlib import Path
        return Path.home()

    def load_settings(self):
        settings = QSettings(os.path.join(PETabGui._get_user_dir(), ".petab.ini"), QSettings.IniFormat)

        benchmark_dir = '../benchmarks/hackathon_contributions_new_data_format'
        self.dir = settings.value("dir", benchmark_dir)
        self.model_dir = settings.value("model_dir", r'Becker_Science2010')
        self.model = settings.value("model", 'Becker_Science2010__BaF3_Exp')
        self.out_dir = settings.value("out_dir", './out')
        self.show_progress = settings.value("show_progress", True, type=bool)
        self.show_result = settings.value("show_result", True, type=bool)
        self.show_result_per_experiment = settings.value("show_result_per_experiment", False, type=bool)
        self.show_result_per_dependent = settings.value("show_result_per_dependent", False, type=bool)
        self.write_report = settings.value("write_report", False, type=bool)

        self.ui.txtDir.setText(self.dir)
        self.ui.txtOutDir.setText(self.out_dir)
        self.ui.chkPlotProgressOfFit.setChecked(self.show_progress)
        self.ui.chkPlotResult.setChecked(self.show_result)
        self.ui.chkPlotResultPerExperiment.setChecked(self.show_result_per_experiment)
        self.ui.chkPlotResultPerDependent.setChecked(self.show_result_per_dependent)
        self.ui.chkWriteReport.setChecked(self.write_report)

    def save_settings(self):
        settings = QSettings(os.path.join(PETabGui._get_user_dir(), ".petab.ini"), QSettings.IniFormat)
        settings.setValue("dir", self.dir)
        settings.setValue("model_dir", self.model_dir)
        settings.setValue("model", self.model)
        settings.setValue("out_dir", self.out_dir)
        settings.setValue("show_progress", self.show_progress)
        settings.setValue("show_result", self.show_result)
        settings.setValue("show_result_per_experiment", self.show_result_per_experiment)
        settings.setValue("show_result_per_dependent", self.show_result_per_dependent)
        settings.setValue("write_report", self.write_report)

    def slotOpenModelDir(self):
        url = QUrl.fromLocalFile(os.path.join(self.dir, self.model_dir))
        QDesktopServices.openUrl(url)

    def slotOpenInCOPASI(self):
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(
                os.path.join(self.out_dir, os.path.splitext(self.model)[0] + '.cps')))

    def slotSetBenchmarkDir(self, dir):
        if not os.path.exists(dir):
            return
        self.dir = dir
        self.ui.txtDir.setText(dir)
        self.load_model_dirs()

    def slotSetModelDir(self, model_dir):
        self.model_dir = model_dir
        items = self.ui.lstModelDirs.findItems(self.model_dir,
                                               Qt.MatchFixedString)
        if len(items) > 0:
            self.ui.lstModelDirs.setCurrentItem(items[0])
        self.load_models()

    def slotSetModel(self, model):
        self.model = model
        self.setWindowFilePath(model)
        items = self.ui.lstModels.findItems(self.model, Qt.MatchFixedString)
        if len(items) > 0:
            self.ui.lstModels.setCurrentItem(items[0])

    def slotSetOutputDir(self, out_dir):
        self.out_dir = out_dir
        self.ui.txtOutDir.setText(out_dir)

    def slotBrowseBenchmarkDir(self):
        result = QFileDialog.getExistingDirectory(self,
                                                  'Select Benchmark dir',
                                                  self.dir)
        if result is None:
            return
        self.slotSetBenchmarkDir(result)

    def slotModelDirSelected(self):
        selected = self.ui.lstModelDirs.currentItem()
        if selected is None:
            self.model_dir = None
        else:
            self.model_dir = selected.text()
        self.slotSetModelDir(self.model_dir)

    def load_model_dirs(self):
        self.ui.lstModelDirs.clear()
        if self.dir is None or not os.path.exists(self.dir):
            return
        for (dirpath, dirnames, filenames) in os.walk(self.dir):
            self.ui.lstModelDirs.addItems(sorted(dirnames))
            break  # only from top level
        if self.model_dir is not None:
            self.slotSetModelDir(self.model_dir)

    def load_models(self):
        self.ui.lstModels.clear()
        self.model = None
        full_dir = os.path.join(self.dir, self.model_dir)
        if self.model_dir is None or not os.path.exists(full_dir):
            self.ui.wdgDetail.setEnabled(False)
            return
        self.ui.wdgDetail.setEnabled(True)
        yaml = None
        for (dirpath, dirnames, filenames) in os.walk(full_dir):
            for file in sorted(filenames):
                if file.startswith('model_'):
                    file = file[6:]
                if file.endswith('.xml'):
                    file = file[:-4]
                    self.ui.lstModels.addItem(file)
                    self.model = file
                if file.endswith('.yaml') and not '_solution' in file:
                    self.ui.lstModels.addItem(file)
                    yaml = file
            break  # skip other dirs
        if yaml is not None:
            self.model = yaml
            self.slotSetModel(yaml)
        elif self.model is not None:
            self.slotSetModel(self.model)

    def slotModelSelected(self):
        selected = self.ui.lstModels.currentItem()
        if selected is None:
            return
        self.slotSetModel(selected.text())

    def slotBrowseOutputDir(self):
        result = QFileDialog.getExistingDirectory(self,
                                                  'Select Output Folder',
                                                  self.out_dir)
        if result is None:
            return
        self.slotSetOutputDir(result)

    def slotConvert(self):
        self.ui.cmdConvert.setEnabled(False)
        self.ui.cmdOpenInCOPASI.setEnabled(False)
        QApplication.processEvents()
        try:
            self.out_dir = self.ui.txtOutDir.text()
            self.show_progress = self.ui.chkPlotProgressOfFit.isChecked()
            self.show_result = self.ui.chkPlotResult.isChecked()
            self.show_result_per_experiment = self.ui.chkPlotResultPerExperiment.isChecked()
            self.show_result_per_dependent = self.ui.chkPlotResultPerDependent.isChecked()
            self.write_report = self.ui.chkWriteReport.isChecked()

            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir, exist_ok=True)
            full_dir = os.path.join(self.dir, self.model_dir)
            converter = convert_petab.PEtabConverter(full_dir, self.model,
                                                     self.out_dir, os.path.splitext(self.model)[0])
            converter.transform_data = self.ui.chkTransformData.isChecked()
            converter.show_progress_of_fit = self.show_progress
            converter.show_result = self.show_result
            converter.show_result_per_experiment = self.show_result_per_experiment
            converter.show_result_per_dependent = self.show_result_per_dependent
            converter.save_report = self.write_report
            converter.convert()
            if converter.experimental_data_file is not None:
                with open(converter.experimental_data_file, 'r') as data:
                    text = data.read()
                    self.ui.txtData.document().setPlainText(text)
        except BaseException:
            msg = traceback.format_exc()
            QMessageBox.critical(self, 'Error converting', msg)
        self.ui.cmdConvert.setEnabled(True)
        self.ui.cmdOpenInCOPASI.setEnabled(True)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


def petab_gui():
    app = QApplication(sys.argv)
    widget = PETabGui()
    sys.exit(app.exec_())


if __name__ == '__main__':
    petab_gui()

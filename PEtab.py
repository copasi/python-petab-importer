import os
import sys
import traceback
from PyQt5.QtWidgets import QMainWindow, QWidget, QDesktopWidget, QApplication, QFileDialog, QDirModel, QMessageBox, QListWidget
from PyQt5 import uic
from PyQt5.QtCore import qDebug, QUrl, QSettings, Qt
from PyQt5.QtGui import QDesktopServices

import convert_petab

class PETabGui(QMainWindow):

    def __init__(self):
        #super().__init__()
        QMainWindow.__init__(self)

        self.dir = None
        self.model_dir = None
        self.model = None
        self.out_dir = None

        self.ui = uic.loadUi('petab.ui', self)

        self.center()
        self.load_settings()
        self.load_model_dirs()
        self.show()

    def closeEvent(self, event):
        self.save_settings()

    def load_settings(self):
        settings = QSettings("copasi", "PEtabImporter")

        self.dir = settings.value("dir", r'E:\Development\Benchmark-Models\hackathon_contributions_new_data_format')
        self.model_dir = settings.value("model_dir", r'Becker_Science2010')
        self.model = settings.value("model", 'Becker_Science2010__BaF3_Exp')
        self.out_dir = settings.value("out_dir", './out')

        self.ui.txtDir.setText(self.dir)
        self.ui.txtOutDir.setText(self.out_dir)

    def save_settings(self):
        settings = QSettings("copasi", "PEtabImporter")
        settings.setValue("dir", self.dir)
        settings.setValue("model_dir", self.model_dir)
        settings.setValue("model", self.model)
        settings.setValue("out_dir", self.out_dir)

    def slotOpenModelDir(self):
        url = QUrl.fromLocalFile(os.path.join(self.dir, self.model_dir))
        QDesktopServices.openUrl(url)

    def slotOpenInCOPASI(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.join(self.out_dir, self.model + '.cps')))

    def slotSetBenchmarkDir(self, dir):
        if not os.path.exists(dir):
            return
        self.dir = dir
        self.ui.txtDir.setText(dir)
        self.load_model_dirs()

    def slotSetModelDir(self, model_dir):
        self.model_dir = model_dir
        items = self.ui.lstModelDirs.findItems(self.model_dir, Qt.MatchFixedString)
        if len(items) > 0:
            self.ui.lstModelDirs.setCurrentItem(items[0])
        self.load_models()

    def slotSetModel(self, model):
        self.model = model
        items = self.ui.lstModels.findItems(self.model, Qt.MatchFixedString)
        if len(items) > 0:
            self.ui.lstModels.setCurrentItem(items[0])

    def slotSetOutputDir(self, out_dir):
        self.out_dir = out_dir
        self.ui.txtOutDir.setText(out_dir)

    def slotBrowseBenchmarkDir(self):
        result = QFileDialog.getExistingDirectory(self, 'Select Benchmark dir', self.dir)
        if result is None:
            return
        self.slotSetBenchmarkDir(result)

    def slotModelDirSelected(self):
        selected = self.ui.lstModelDirs.currentItem()
        if selected is None:
            self.model_dir = None
        else:
            self.model_dir = os.path.join(self.dir, selected.text())
        self.slotSetModelDir(self.model_dir)

    def load_model_dirs(self):
        self.ui.lstModelDirs.clear()
        if self.dir is None or not os.path.exists(self.dir):
            return
        for (dirpath, dirnames, filenames) in os.walk(self.dir):
            self.ui.lstModelDirs.addItems(dirnames)
        if self.model_dir is not None:
            self.slotSetModelDir(self.model_dir)

    def load_models(self):
        self.ui.lstModels.clear()
        if self.model_dir is None or not os.path.exists(os.path.join(self.dir, self.model_dir)):
            self.ui.wdgDetail.setEnabled(False)
            return
        self.ui.wdgDetail.setEnabled(True)
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(self.dir, self.model_dir)):
            for file in filenames:
                if file.startswith('model_'):
                    file = file[6:]
                if file.endswith('.xml'):
                    file = file[:-4]
                    self.ui.lstModels.addItem(file)
            if self.model is not None:
                self.slotSetModel(self.model)

    def slotModelSelected(self):
        selected = self.ui.lstModels.currentItem()
        if selected is None:
            return
        self.slotSetModel(selected.text())

    def slotBrowseOutputDir(self):
        result = QFileDialog.getExistingDirectory(self, 'Select Output Folder', self.out_dir)
        if result is None:
            return
        self.slotSetOutputDir(result)

    def slotConvert(self):
        try:
            self.out_dir = self.ui.txtOutDir.text()
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir, exist_ok=True)
            converter = convert_petab.PEtabConverter(self.model_dir, self.model, self.out_dir, self.model)
            converter.convert()
            if converter.experimental_data_file is not None:
                with open(converter.experimental_data_file, 'r') as data:
                    text = data.read()
                    self.ui.txtData.document().setPlainText(text)
        except:
            msg = traceback.format_exc()
            QMessageBox.critical(self, 'Error converting', msg)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = PETabGui()
    sys.exit(app.exec_())
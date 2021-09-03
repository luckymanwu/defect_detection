from PyQt5 import QtWidgets
import  TopBar
from TrainMainWin import Ui_TrainMainWindow


class TrainWin(QtWidgets.QMainWindow,Ui_TrainMainWindow):
    def __init__(self):
        super(TrainWin, self).__init__()
        self.setupUi(self)
        self.actionDetection.triggered.connect(self.change_mode)
        self.actionMark.triggered.connect(self.switchMark)
    def change_mode(self):
        TopBar.switchDetection(self)

    def switchMark(self):
        TopBar.switchMark(self)


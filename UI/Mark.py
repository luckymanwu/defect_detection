from PyQt5 import QtWidgets
import  TopBar
from MarkMainWin import  Ui_MarkMainWindow
class MarkWin(QtWidgets.QMainWindow,Ui_MarkMainWindow):
    def __init__(self):
        super(MarkWin, self).__init__()
        self.setupUi(self)
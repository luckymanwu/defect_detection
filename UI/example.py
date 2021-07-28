import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QTextEdit, QAction, QFileDialog
from PyQt5.QtGui import QIcon


class Example(QMainWindow):
    def __init__(self):
        super(Example, self).__init__()
        self.initUI()

    def initUI(self):
        self.textEdit = QTextEdit()
        self.setCentralWidget(self.textEdit)
        self.statusBar()

        openfile = QAction(QIcon(r'C:\Users\Administrator\PycharmProjects\QT\picture\文件.jpg'), 'open', self)
        openfile.setShortcut("Ctrl + 0")
        openfile.setStatusTip('open new file')
        openfile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        filemune = menubar.addMenu('$File')
        filemune.addAction(openfile)

        self.setGeometry(300, 300, 300, 300)
        self.setWindowTitle('FIEL dialog')
        self.show()

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'open file', '/')
        if fname[0]:
            try:
                f = open(fname[0], 'r')
                with f:
                    data = f.read()
                    self.textEdit.setText(data)
            except:
                self.textEdit.setText("打开文件失败，可能是文件内型错误")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

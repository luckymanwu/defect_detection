from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from model.MyLabel import MyLabel

class QClickableImage(QWidget):
    image_id = ''

    def __init__(self, width=0, height=0, pixmap=None, image_id=''):
        QWidget.__init__(self)

        self.width = width
        self.height = height
        self.pixmap = pixmap

        self.layout = QVBoxLayout(self)
        self.lable2 = QLabel()
        self.lable2.setObjectName('label2')

        if self.width and self.height:
            self.resize(self.width, self.height)
        if self.pixmap and image_id:
            pixmap = self.pixmap.scaled(QSize(self.width, self.height), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label1 = MyLabel(pixmap, image_id)
            self.label1.setObjectName('label1')
            # self.label1.connect(self.mouseressevent())
            self.layout.addWidget(self.label1)

        if image_id:
            self.image_id = image_id
            self.lable2.setText(image_id.split('\\')[-1])
            self.lable2.setAlignment(Qt.AlignCenter)
            ###让文字自适应大小
            self.lable2.adjustSize()
            self.layout.addWidget(self.lable2)
        self.setLayout(self.layout)


    def imageId(self):
        return self.image_id

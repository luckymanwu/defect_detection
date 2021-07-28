from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel


class MyLabel(QLabel):
    global NOP_value, NOP_dict

    def __init__(self, pixmap=None, image_id=None):
        QLabel.__init__(self)
        self.pixmap = pixmap
        self.image_id = image_id
        self.setPixmap(pixmap)

        self.setAlignment(Qt.AlignCenter)
        self.setContextMenuPolicy(Qt.CustomContextMenu)

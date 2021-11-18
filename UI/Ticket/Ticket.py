import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
from utils.CommonHelper import CommonHelper
from UI.Ticket.TicketWin import TicketWin
class Ticket(TicketWin):
    ticket_signal = pyqtSignal(object, object)
    def __init__(self, parent=None):
        super(Ticket, self).__init__(parent)
        self.setupUi(self)
        styleFile = 'E:\defect_detection-main/resource/Ticket.qss'
        # 换肤时进行全局修改，只需要修改不同的QSS文件即可
        style = CommonHelper.readQss(styleFile)
        self.setStyleSheet(style)
        self.confirm.clicked.connect(self.submitTicket)
        self.cancel.clicked.connect(self.close)


    def submitTicket(self):
        operator = self.operator_lineEdit.text()
        productName = self.productName_lineEdit.text()
        productId = self.productId_lineEdit.text()
        self.ticket_signal.emit(productName,productId)
        self.close()




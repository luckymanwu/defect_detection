# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TicketWin.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5.QtGui import  QPalette, QColor,QIcon,QPixmap
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal, QPoint, Qt
class TicketWin(QDialog):
    def setupUi(self, Ticket):
        Ticket.setObjectName("Ticket")
        Ticket.resize(569, 407)
        self.Title = QtWidgets.QLabel(Ticket)
        self.Title.setGeometry(QtCore.QRect(240, 12, 130, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.Title.setFont(font)
        self.Title.setObjectName("Title")
        self.widget = QtWidgets.QWidget(Ticket)
        self.widget.setGeometry(QtCore.QRect(110, 60, 371, 231))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.operator_2 = QtWidgets.QLabel(self.widget)
        self.operator_2.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.operator_2.setFont(font)
        self.operator_2.setObjectName("operator_2")
        self.gridLayout.addWidget(self.operator_2, 0, 0, 1, 1)
        self.operator_lineEdit = QtWidgets.QLineEdit(self.widget)
        self.operator_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.operator_lineEdit.setObjectName("operator_lineEdit")
        self.gridLayout.addWidget(self.operator_lineEdit, 0, 1, 1, 1)
        self.productName = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.productName.setFont(font)
        self.productName.setObjectName("productName")
        self.gridLayout.addWidget(self.productName, 1, 0, 1, 1)
        self.productName_lineEdit = QtWidgets.QLineEdit(self.widget)
        self.productName_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.productName_lineEdit.setObjectName("productName_lineEdit")
        self.gridLayout.addWidget(self.productName_lineEdit, 1, 1, 1, 1)
        self.product_id = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.product_id.setFont(font)
        self.product_id.setObjectName("product_id")
        self.gridLayout.addWidget(self.product_id, 2, 0, 1, 1)
        self.productId_lineEdit = QtWidgets.QLineEdit(self.widget)
        self.productId_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.productId_lineEdit.setObjectName("productId_lineEdit")
        self.gridLayout.addWidget(self.productId_lineEdit, 2, 1, 1, 1)
        self.widget1 = QtWidgets.QWidget(Ticket)
        self.widget1.setGeometry(QtCore.QRect(280, 321, 201, 51))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.confirm = QtWidgets.QPushButton(self.widget1)
        self.confirm.setMinimumSize(QtCore.QSize(50, 40))
        self.confirm.setObjectName("confirm")
        self.horizontalLayout.addWidget(self.confirm)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.cancel = QtWidgets.QPushButton(self.widget1)
        self.cancel.setMinimumSize(QtCore.QSize(50, 40))
        self.cancel.setObjectName("cancel")
        self.horizontalLayout.addWidget(self.cancel)

        self.retranslateUi(Ticket)
        QtCore.QMetaObject.connectSlotsByName(Ticket)

    def retranslateUi(self, Ticket):
        _translate = QtCore.QCoreApplication.translate
        Ticket.setWindowTitle(_translate("Ticket", "Dialog"))
        self.Title.setText(_translate("Ticket", "新建工单"))
        self.operator_2.setText(_translate("Ticket", "操作人："))
        self.productName.setText(_translate("Ticket", "产品名称："))
        self.product_id.setText(_translate("Ticket", "产品批次："))
        self.confirm.setText(_translate("Ticket", "confirm"))
        self.cancel.setText(_translate("Ticket", "cancel"))
        pe = QPalette()
        Ticket.setAutoFillBackground(True)
        pe.setColor(QPalette.Window, QColor("#5A5A5A"))  # 设置背景色
        Ticket.setPalette(pe)
        # self.setWindowFlags(Qt.FramelessWindowHint)  # 去边框
        self.setWindowTitle("新建工单")
        icon = QIcon()
        icon.addPixmap(QPixmap("../resource/logo3.ico"))
        self.setWindowIcon(icon)



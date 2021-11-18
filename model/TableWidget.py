import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal
import cv2
OUT_PATH = "../output/Image/"
class TableWidget(QWidget):
    control_signal = pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super(TableWidget, self).__init__(*args, **kwargs)
        self.__init_ui()

    def __init_ui(self):
        style_sheet = """
            QTableWidget {
                border: none;
                background-color:rgb(240,240,240)
            }
            QPushButton{
                max-width: 18ex;
                max-height: 6ex;
                font-size: 11px;
            }
            QLineEdit{
                max-width: 30px
            }
        """
        column_name  = [
            '编号',
            '缺陷类型',
        ]
        self.table = QTableWidget(10, 2)  # 5 行 2 列的表格
        self.table.setHorizontalHeaderLabels(column_name)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 自适应宽度
        self.__layout = QVBoxLayout()
        self.__layout.addWidget(self.table)
        self.setLayout(self.__layout)
        self.setStyleSheet(style_sheet)
        self.table.doubleClicked.connect(self.table_double_clicked)

    def table_double_clicked(self,index):
        table_column = index.column()
        table_row = index.row()
        current_item = self.table.item(table_row, table_column).text()
        image_path = OUT_PATH+current_item
        print(image_path)
        image = cv2.imread(image_path)
        cv2.imshow(current_item,image)
        cv2.waitKey(0)

    def setPageController(self, page):
        """自定义页码控制器"""
        control_layout = QHBoxLayout()
        homePage = QPushButton("首页")
        prePage = QPushButton("<上一页")
        self.curPage = QLabel("1")
        nextPage = QPushButton("下一页>")
        finalPage = QPushButton("尾页")
        self.totalPage = QLabel("共" + str(page) + "页")
        skipLable_0 = QLabel("跳到")
        self.skipPage = QLineEdit()
        skipLabel_1 = QLabel("页")
        confirmSkip = QPushButton("确定")
        homePage.clicked.connect(self.__home_page)
        prePage.clicked.connect(self.__pre_page)
        nextPage.clicked.connect(self.__next_page)
        finalPage.clicked.connect(self.__final_page)
        confirmSkip.clicked.connect(self.__confirm_skip)
        control_layout.addStretch(1)
        control_layout.addWidget(homePage)
        control_layout.addWidget(prePage)
        control_layout.addWidget(self.curPage)
        control_layout.addWidget(nextPage)
        control_layout.addWidget(finalPage)
        control_layout.addWidget(self.totalPage)
        control_layout.addWidget(skipLable_0)
        control_layout.addWidget(self.skipPage)
        control_layout.addWidget(skipLabel_1)
        control_layout.addWidget(confirmSkip)
        control_layout.addStretch(1)
        self.__layout.addLayout(control_layout)

    def __home_page(self):
        """点击首页信号"""
        self.control_signal.emit(["home", self.curPage.text()])

    def __pre_page(self):
        """点击上一页信号"""
        self.control_signal.emit(["pre", self.curPage.text()])

    def __next_page(self):
        """点击下一页信号"""
        self.control_signal.emit(["next", self.curPage.text()])

    def __final_page(self):
        """尾页点击信号"""
        self.control_signal.emit(["final", self.curPage.text()])

    def __confirm_skip(self):
        """跳转页码确定"""
        self.control_signal.emit(["confirm", self.skipPage.text()])

    def showTotalPage(self):
        """返回当前总页数"""
        return int(self.totalPage.text()[1:-1])

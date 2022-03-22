from PyQt5.QtCore import pyqtSignal, QPoint, Qt
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QSpacerItem, QSizePolicy, QPushButton
from utils.CommonHelper import CommonHelper
from UI.Ticket.Ticket import Ticket
class TitleBar(QWidget):
    # 窗口最小化信号
    windowMinimumed = pyqtSignal()
    # 窗口最大化信号
    windowMaximumed = pyqtSignal()
    # 窗口还原信号
    windowNormaled = pyqtSignal()
    # 窗口关闭信号
    windowClosed = pyqtSignal()
    # 窗口移动
    windowMoved = pyqtSignal(QPoint)
    def __init__(self, *args, **kwargs):
          super(TitleBar, self).__init__(*args, **kwargs)
          styleFile = '../../resource/TitleBar.qss'
          # 换肤时进行全局修改，只需要修改不同的QSS文件即可
          style = CommonHelper.readQss(styleFile)
          self.setStyleSheet(style)
          # 支持qss设置背景
          self.setAttribute(Qt.WA_StyledBackground, True)
          self.mPos = None
          self.iconSize = 40  # 图标的默认大小
          # 布局
          layout = QHBoxLayout(self, spacing=0)
          layout.setContentsMargins(5, 5, 5, 5)
          # 窗口图标
          self.iconLabel = QLabel(self)
          # self.iconLabel.setScaledContents(True)
          layout.addWidget(self.iconLabel)
          # 窗口标题
          self.titleLabel = QLabel(self)
          self.titleLabel.setMargin(10)
          layout.addWidget(self.titleLabel)
          layout.addSpacerItem(QSpacerItem(30, 20))

          # 中间伸缩条
          layout.addSpacerItem(QSpacerItem(
              40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
         # 利用Webdings字体来显示图标
          font = self.font() or QFont()
          font.setFamily('Webdings')
          # 最小化按钮
          self.buttonMinimum = QPushButton(
              '0', self, clicked=self.windowMinimumed.emit, font=font, objectName='buttonMinimum')
          layout.addWidget(self.buttonMinimum)
          # 最大化/还原按钮
          self.buttonMaximum = QPushButton(
              '1', self, clicked=self.showMaximized, font=font, objectName='buttonMaximum')
          layout.addWidget(self.buttonMaximum)
          # 关闭按钮
          self.buttonClose = QPushButton(
              'r', self, clicked=self.windowClosed.emit, font=font, objectName='buttonClose')
          layout.addWidget(self.buttonClose)
          # 初始高度
          self.setHeight()



    def showMaximized(self):
         if self.buttonMaximum.text() == '1':
             # 最大化
             self.buttonMaximum.setText('2')
             self.windowMaximumed.emit()
         else:  # 还原
             self.buttonMaximum.setText('1')
             self.windowNormaled.emit()

    def setHeight(self, height=50):
         """设置标题栏高度"""
         self.setMinimumHeight(height)
         self.setMaximumHeight(height)
         # 设置右边按钮的大小
         self.buttonMinimum.setMinimumSize(height, height)
         self.buttonMinimum.setMaximumSize(height, height)
         self.buttonMaximum.setMinimumSize(height, height)
         self.buttonMaximum.setMaximumSize(height, height)
         self.buttonClose.setMinimumSize(height, height)
         self.buttonClose.setMaximumSize(height, height)

    def setTitle(self, title):
         """设置标题"""
         self.titleLabel.setText(title)

    def setIcon(self, icon):
        """设置图标"""
        self.iconLabel.setPixmap(icon.pixmap(self.iconSize, self.iconSize))

    def setIconSize(self, size):
         """设置图标大小"""
         self.iconSize = size

    def enterEvent(self, event):
         self.setCursor(Qt.ArrowCursor)
         super(TitleBar, self).enterEvent(event)

    def mouseDoubleClickEvent(self, event):
         super(TitleBar, self).mouseDoubleClickEvent(event)
         self.showMaximized()

    def mousePressEvent(self, event):
         """鼠标点击事件"""
         if event.button() == Qt.LeftButton:
             self.mPos = event.pos()
         event.accept()

    def mouseReleaseEvent(self, event):
        '''鼠标弹起事件'''
        self.mPos = None
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.mPos:
            self.windowMoved.emit(self.mapToGlobal(event.pos() - self.mPos))
        event.accept()

Left, Top, Right, Bottom, LeftTop, RightTop, LeftBottom, RightBottom = range(8)
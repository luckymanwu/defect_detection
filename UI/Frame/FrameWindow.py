import sys
import qdarkstyle
from PyQt5.QtCore import QSize, QSettings
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt,QPoint
from PyQt5.QtGui import QEnterEvent, QPainter, QColor, QPen, QPixmap
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout,QDesktopWidget
from UI.Frame.LeftTabWidget import LeftTabWidget
from UI.Frame.TitleBar import TitleBar
from UI.Setting.Setting import Setting
Left, Top, Right, Bottom, LeftTop, RightTop, LeftBottom, RightBottom = range(8)
from PyQt5.QtWidgets import QDesktopWidget


class FramelessWindow(QWidget):
     # 四周边距
     Margins = 5
     def __init__(self, *args, **kwargs):
         super(FramelessWindow, self).__init__(*args, **kwargs)
         # styleFile = '../resource/Frame.qss'
         # style = CommonHelper.readQss(styleFile)
         # self.setStyleSheet(style)
         self._pressed = False
         self.Direction = None
         self.setAttribute(Qt.WA_TranslucentBackground, True)
         self.setWindowFlags(Qt.FramelessWindowHint)  # 隐藏边框
         # 鼠标跟踪
         self.setMouseTracking(True)
         # 布局
         layout = QVBoxLayout(self, spacing=0)
         # 预留边界用于实现无边框窗口调整大小
         layout.setContentsMargins(
             self.Margins, self.Margins, self.Margins, self.Margins)
         # 标题栏
         self.titleBar = TitleBar(self)
         layout.addWidget(self.titleBar)
         # 信号槽
         self.titleBar.windowMinimumed.connect(self.showMinimized)
         self.titleBar.windowMaximumed.connect(self.showMaximized)
         self.titleBar.windowNormaled.connect(self.showNormal)
         self.titleBar.windowClosed.connect(self.close)
         self.titleBar.windowMoved.connect(self.move)
         self.windowTitleChanged.connect(self.titleBar.setTitle)
         self.windowIconChanged.connect(self.titleBar.setIcon)
     def setTitleBarHeight(self, height=35):
         """设置标题栏高度"""
         self.titleBar.setHeight(height)


     def setIconSize(self, size):
         """设置图标的大小"""
         self.titleBar.setIconSize(size)

     def setWidget(self, widget):
         """设置自己的控件"""
         if hasattr(self, '_widget'):
             return
         self._widget = widget
         # 设置默认背景颜色,否则由于受到父窗口的影响导致透明
         self._widget.installEventFilter(self)
         self.layout().addWidget(self._widget)

     def move(self, pos):
         if self.windowState() == Qt.WindowMaximized or self.windowState() == Qt.WindowFullScreen:
             # 最大化或者全屏则不允许移动
             return
         super(FramelessWindow, self).move(pos)

     def showMaximized(self):
         """最大化,要去除上下左右边界,如果不去除则边框地方会有空隙"""
         super(FramelessWindow, self).showMaximized()
         self.layout().setContentsMargins(0, 0, 0, 0)

     def showNormal(self):
         """还原,要保留上下左右边界,否则没有边框无法调整"""
         super(FramelessWindow, self).showNormal()
         self.layout().setContentsMargins(
             self.Margins, self.Margins, self.Margins, self.Margins)

     def eventFilter(self, obj, event):
         """事件过滤器,用于解决鼠标进入其它控件后还原为标准鼠标样式"""
         if isinstance(event, QEnterEvent):
             self.setCursor(Qt.ArrowCursor)
         return super(FramelessWindow, self).eventFilter(obj, event)

     def paintEvent(self, event):
         """由于是全透明背景窗口,重绘事件中绘制透明度为1的难以发现的边框,用于调整窗口大小"""
         super(FramelessWindow, self).paintEvent(event)
         painter = QPainter(self)
         painter.setPen(QPen(QColor(255, 255, 255, 1), 2 * self.Margins))
         painter.drawRect(self.rect())

     def mousePressEvent(self, event):
         """鼠标点击事件"""
         super(FramelessWindow, self).mousePressEvent(event)
         if event.button() == Qt.LeftButton:
             self._mpos = event.pos()
             self._pressed = True

     def mouseReleaseEvent(self, event):
         '''鼠标弹起事件'''
         super(FramelessWindow, self).mouseReleaseEvent(event)
         self._pressed = False
         self.Direction = None

     def mouseMoveEvent(self, event):
         """鼠标移动事件"""
         super(FramelessWindow, self).mouseMoveEvent(event)
         pos = event.pos()
         xPos, yPos = pos.x(), pos.y()
         wm, hm = self.width() - self.Margins, self.height() - self.Margins
         if self.isMaximized() or self.isFullScreen():
             self.Direction = None
             self.setCursor(Qt.ArrowCursor)
             return
         if event.buttons() == Qt.LeftButton and self._pressed:
             self._resizeWidget(pos)
             return
         if xPos <= self.Margins and yPos <= self.Margins:
             # 左上角
             self.Direction = LeftTop
             self.setCursor(Qt.SizeFDiagCursor)
         elif wm <= xPos <= self.width() and hm <= yPos <= self.height():
             # 右下角
             self.Direction = RightBottom
             self.setCursor(Qt.SizeFDiagCursor)
         elif wm <= xPos and yPos <= self.Margins:
             # 右上角
             self.Direction = RightTop
             self.setCursor(Qt.SizeBDiagCursor)
         elif xPos <= self.Margins and hm <= yPos:
             # 左下角
             self.Direction = LeftBottom
             self.setCursor(Qt.SizeBDiagCursor)
         elif 0 <= xPos <= self.Margins and self.Margins <= yPos <= hm:
             # 左边
             self.Direction = Left
             self.setCursor(Qt.SizeHorCursor)
         elif wm <= xPos <= self.width() and self.Margins <= yPos <= hm:
             # 右边
             self.Direction = Right
             self.setCursor(Qt.SizeHorCursor)
         elif self.Margins <= xPos <= wm and 0 <= yPos <= self.Margins:
             # 上面
             self.Direction = Top
             self.setCursor(Qt.SizeVerCursor)
         elif self.Margins <= xPos <= wm and hm <= yPos <= self.height():
             # 下面
             self.Direction = Bottom
             self.setCursor(Qt.SizeVerCursor)

     def _resizeWidget(self, pos):
         """调整窗口大小"""
         if self.Direction == None:
             return
         mpos = pos - self._mpos
         xPos, yPos = mpos.x(), mpos.y()
         geometry = self.geometry()
         x, y, w, h = geometry.x(), geometry.y(), geometry.width(), geometry.height()
         if self.Direction == LeftTop:  # 左上角
             if w - xPos > self.minimumWidth():
                 x += xPos
                 w -= xPos
             if h - yPos > self.minimumHeight():
                 y += yPos
                 h -= yPos
         elif self.Direction == RightBottom:  # 右下角
             if w + xPos > self.minimumWidth():
                 w += xPos
                 self._mpos = pos
             if h + yPos > self.minimumHeight():
                 h += yPos
                 self._mpos = pos
         elif self.Direction == RightTop:  # 右上角
             if h - yPos > self.minimumHeight():
                 y += yPos
                 h -= yPos
             if w + xPos > self.minimumWidth():
                 w += xPos
                 self._mpos.setX(pos.x())
         elif self.Direction == LeftBottom:  # 左下角
             if w - xPos > self.minimumWidth():
                 x += xPos
                 w -= xPos
             if h + yPos > self.minimumHeight():
                 h += yPos
                 self._mpos.setY(pos.y())
         elif self.Direction == Left:  # 左边
             if w - xPos > self.minimumWidth():
                 x += xPos
                 w -= xPos
             else:
                 return
         elif self.Direction == Right:  # 右边
             if w + xPos > self.minimumWidth():
                 w += xPos
                 self._mpos = pos
             else:
                 return
         elif self.Direction == Top:  # 上面
             if h - yPos > self.minimumHeight():
                 y += yPos
                 h -= yPos
             else:
                 return
         elif self.Direction == Bottom:  # 下面
             if h + yPos > self.minimumHeight():
                 h += yPos
                 self._mpos = pos
             else:
                 return
         self.setGeometry(x, y, w, h)

     def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(QPoint((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2))

     # def initSetting(self):


class MainWindow(QWidget):

     def __init__(self, *args, **kwargs):
         super(MainWindow, self).__init__(*args, **kwargs)
         layout = QVBoxLayout(self, spacing=0)
         layout.setContentsMargins(0, 0, 0, 0)
         configuration = QSettings('config.ini', QSettings.IniFormat)
         configuration.setIniCodec('UTF-8')  # 设置ini文件编码为 UTF-8
         self.left_tag = LeftTabWidget(configuration)
         layout.addWidget(self.left_tag)


if __name__ == '__main__':

     app = QApplication(sys.argv)
     mainWnd = FramelessWindow()
     icon = QIcon()
     icon.addPixmap(QPixmap("../resource/logo3.ico"))
     mainWnd.setWindowIcon(icon)
     mainWnd.setWindowTitle('SMMD')
     mainWnd.resize(QSize(1400,950))

     mainWnd.center()
     mainWnd.setWidget(MainWindow())  # 把自己的窗口添加进来
     mainWnd.show()
     sys.exit(app.exec_())
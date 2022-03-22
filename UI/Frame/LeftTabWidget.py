from PyQt5.QtWidgets import QListWidget,QStackedWidget
from PyQt5.QtWidgets import QListWidgetItem
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtCore import QSize, Qt
from UI.Detection.Detection import Detection
from UI.Setting.Setting import Setting
from UI.PictureDetection.PictureDetection import PictureDetection
from UI.Annotation.Annotation import Annotation
from UI.Train.Train import Train


class LeftTabWidget(QWidget):
     '''左侧选项栏'''
     def __init__(self,configuration):
         super(LeftTabWidget, self).__init__()
         self.setObjectName('LeftTabWidget')
         self.setWindowTitle('LeftTabWidget')
         with open('../../resource/leftTab.qss', 'r', encoding='utf-8') as f:   #导入QListWidget的qss样式
             self.list_style = f.read()
         self.main_layout = QHBoxLayout(self, spacing=0)     #窗口的整体布局
         self.main_layout.setContentsMargins(0,0,0,0)
         self.left_widget = QListWidget()     #左侧选项列表
         self.left_widget.setStyleSheet(self.list_style)
         self.main_layout.addWidget(self.left_widget)
         self.right_widget = QStackedWidget()
         self.main_layout.addWidget(self.right_widget)
         self.showWin = None
         self.configuration = configuration
         self.Detection = Detection(self.configuration)
         self.PictureDetection=PictureDetection(self.configuration)
         self.Annotation = Annotation(self.configuration)
         self.Train = Train(self.configuration)
         self.Setting = Setting(self.configuration)
         self.Setting.train_modal_signal.connect(self.update_train)
         self._setup_ui()

     def update_train(self,title):
         self.Train.chart.setTitle("训练模式: "+title)
     def _setup_ui(self):
         '''加载界面ui'''
         self.left_widget.currentRowChanged.connect(self.right_widget.setCurrentIndex)   #list和右侧窗口的index对应绑定
         self.left_widget.setFrameShape(QListWidget.NoFrame)    #去掉边框
         self.left_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  #隐藏滚动条
         self.left_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
         list_str = ['实时检测','图片检测','图片标注','模型训练','设置']
         win_list=[self.Detection,self.PictureDetection,self.Annotation,self.Train,self.Setting]
         for i in range(5):
             self.item = QListWidgetItem(list_str[i],self.left_widget)   #左侧选项的添加
             self.item.setSizeHint(QSize(30,60))
             self.item.setTextAlignment(Qt.AlignCenter)                  #居中显示
             self.right_widget.addWidget(win_list[i])




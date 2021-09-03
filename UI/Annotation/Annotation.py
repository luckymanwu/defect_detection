from PyQt5 import QtWidgets
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QRect, Qt, QSettings
from UI.ZoomWidget import ZoomWidget
from UI.Canvas import Canvas
from project.utils import *
from UI.Annotation.AnnotationWin1 import AnnotationWin
from utils.CommonHelper import CommonHelper
import os
import numpy
import cv2

class Annotation(AnnotationWin):
    FIT_WINDOW, MANUAL_ZOOM = list(range(2))
    def __init__(self,configuration):
        super(Annotation, self).__init__()
        self.setupUi(self)
        styleFile = '../resource/Annotation.qss'
        # 换肤时进行全局修改，只需要修改不同的QSS文件即可
        style = CommonHelper.readQss(styleFile)
        self.setStyleSheet(style)

        self.zoom_widget.valueChanged.connect(self.paint_canvas)
        self.zoom_in.clicked.connect(lambda :self.addZoom(10))
        self.zoom_out.clicked.connect(lambda :self.addZoom(-10))
        self.fit_window.clicked.connect(lambda :self.set_fit_window(True))
        self.createBox.clicked.connect(self.create_shape)

        self.scalers = {
            self.FIT_WINDOW: self.scale_fit_window,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }
        self._beginner = True
        self.configuration = configuration
        self.open.clicked.connect(self.open_file)
        self.openDir.clicked.connect(self.open_dir)
        self.previous.clicked.connect(self.open_previous)
        self.next.clicked.connect(self.open_next)
        self.save.clicked.connect(self.save_img)
        self.add_button.clicked.connect(self.add_label)
        self.label_input.textChanged.connect(self.set_text)
        self.label_list.itemClicked.connect(self.select_type)
        self.filename = None
        self.current = 0
        self.cwd = os.getcwd()  # 获取当前程序文件位置
        self.imageDir =''
        self.imageList = []
        self.imageTotal = 0
        self.imageDirPathBuffer = ''
        self.text=None
        self.selectType=None
        self.save_path='../output/txt'
        self.save_xml_path = '../../output/xml'
        self.plotDist = {}
        self.typeDist={}

    def open_file(self):
        self.filename = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if not self.filename:
           return None
        self.imageTotal =1
        self.load_image(self.filename[0])

    def load_image(self,file):
        self.canvas.setEnabled(False)
        self.img = cv2.imread(file)
        self.image = QtGui.QImage( self.img.data,  self.img.shape[1],  self.img.shape[0],
                                    QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式

        self.canvas.load_pixmap(QtGui.QPixmap.fromImage(self.image))
        self.canvas.setEnabled(True)
        # self.adjust_scale(initial=True)
        self.paint_canvas()

        self.show_label.setText(str(self.current+1) + "/" + str(self.imageTotal))
        self.info.clear()
        if(self.imageList[self.current] in self.plotDist):
            coordinate=self.plotDist[self.imageList[self.current]]
            type = self.typeDist[self.imageList[self.current]]
            s = '%s (%g,%g)===>(%g,%g)' % (
            type,coordinate[0] , coordinate[1], coordinate[2], coordinate[3])
            self.info.addItem(s)
        return self.img

    def open_dir(self):
        self.imageDir = QFileDialog.getExistingDirectory(self, "选取图片", self.cwd)
        if not self.imageDir:
            return None
        self.imageList = os.listdir(self.imageDir)
        self.imageList = sorted(self.imageList)
        self.imageTotal = len(self.imageList)
        self.filename = None
        self.imageDirPathBuffer = self.imageDir
        self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.current])


    def open_next(self, event=None):
        if self.current < len(self.imageList)-1:
           self.current += 1
           im0 = self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.current])
           # self.img_label.x0 = 0
           # self.img_label.x1 = 0
           # self.img_label.y0 = 0
           # self.img_label.y1 = 0
           # if self.imageList[self.current] in self.plotDist:
           #     cv2_img = cv2.cvtColor(numpy.asarray(im0), cv2.COLOR_RGB2BGR)
           #     plot_one_box(self.plotDist[self.imageList[self.current]], cv2_img, color=(255, 0, 0))
           #     qt_img = QtGui.QImage(cv2_img.data,  # 数据源
           #                           cv2_img.shape[1],  # 宽度
           #                           cv2_img.shape[0],  # 高度
           #                           cv2_img.shape[1] * 3,  # 行字节数
           #                           QtGui.QImage.Format_RGB888)
           #     # label 控件显示图片
           #     self.img_label.setPixmap(QtGui.QPixmap.fromImage(qt_img))


    def open_previous(self, event=None):
         if self.current >= 1:
             self.current -= 1
             im0=self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.current])
             if self.imageList[self.current] in self.plotDist:
                 cv2_img = cv2.cvtColor(numpy.asarray(im0), cv2.COLOR_RGB2BGR)
                 plot_one_box(self.plotDist[self.imageList[self.current]], cv2_img, color=(255, 0, 0))
                 qt_img = QtGui.QImage(cv2_img.data,  # 数据源
                                       cv2_img.shape[1],  # 宽度
                                       cv2_img.shape[0],  # 高度
                                       cv2_img.shape[1] * 3,  # 行字节数
                                       QtGui.QImage.Format_RGB888)
                 # label 控件显示图片
                 self.img_label.setPixmap(QtGui.QPixmap.fromImage(qt_img))


    def add_label(self):
        self.label_list.addItem(self.text)
        self.label_input.clear()

    def set_text(self):
      self.text=self.label_input.text()

    def select_type(self):
      print(self.selectType)
      self.selectType = self.label_list.selectedItems()[0].text()


    def save_img(self):

        if  self.selectType and (self.img_label.x0 !=0):
            s='%s (%g,%g)=====>(%g,%g)' %(self.selectType,self.img_label.x0,self.img_label.y0,self.img_label.x1,self.img_label.y1)
            self.info.addItem(s)
            plot =[self.img_label.x0,self.img_label.y0,self.img_label.x1,self.img_label.y1]
            self.plotDist[self.imageList[self.current]] = plot
            self.typeDist[self.imageList[self.current]] = self.selectType
            img_tag_info  = {"img_name":self.imageList[self.current],"img_type":self.selectType,"x0":self.img_label.x0,"y0": self.img_label.y0,"x1": self.img_label.x1,"y1":self.img_label.y1}
            # savePath = self.save_path + '/'+self.imageList[self.current]
            # with open(savePath[:savePath.rfind('.')] + '.txt', 'a') as file:
            # #     file.write(('%s %s %g %g %g %g ' + '\n') % (self.imageList[self.current],self.selectType, self.img_label.x0, self.img_label.y0, self.img_label.x1, self.img_label.y1))
            # self.txtToXml(os.path.splitext(savePath)[0] + ".txt",self.save_xml_path,self.imageList[self.current].split('.')[0])
            self.saveXml(img_tag_info)
        else:
            QMessageBox.information(self,'提示','未选中标签或目标')

        self.selectType =None
        self.img_label.x0 = 0
        self.img_label.y0 = 0
        self.img_label.x1 = 0
        self.img_label.y1 = 0


    def saveXml(self,img_tag_info):
        src_xml_dir = self.configuration.value("SAVE_XML_PATH")
        img = img_tag_info["img_name"].split('.')[0]
        xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(self.width) + '</width>\n')
        xml_file.write('        <height>' + str(self.height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')
        # write the region of image on xml file
        # spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + img_tag_info["img_name"] + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' +  str(img_tag_info["x0"]) + '</xmin>\n')
        xml_file.write('            <ymin>' +  str(img_tag_info["y0"]) + '</ymin>\n')
        xml_file.write('            <xmax>' +  str(img_tag_info["x1"]) + '</xmax>\n')
        xml_file.write('            <ymax>' +  str(img_tag_info["y1"]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')
        xml_file.write('</annotation>')

    def paint_canvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoom_widget.value()
        self.canvas.label_font_size = int(0.02 * max(self.image.width(), self.image.height()))
        self.canvas.adjustSize()
        self.canvas.update()

    def addZoom(self,increment=10):
        self.zoom_mode = self.MANUAL_ZOOM
        self.zoom_widget.setValue(self.zoom_widget.value() + increment)

    def set_fit_window(self, value=True):
        # if value:
        #     self.actions.fitWidth.setChecked(False)
        self.zoom_mode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        print(self.zoom_mode)

        self.adjust_scale()

    def adjust_scale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoom_mode]()
        # value = self.scale_fit_window()
        self.zoom_widget.setValue(int(100 * value))

    def scale_fit_window(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.parent().width() - e
        h1 = self.parent().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def create_shape(self):
        assert self.beginner()
        self.canvas.set_editing(False)

    def beginner(self):
        return self._beginner

    def new_shape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if not self.use_default_label_checkbox.isChecked() or not self.default_label_text_line.text():
            if len(self.label_hist) > 0:
                self.label_dialog = LabelDialog(
                    parent=self, list_item=self.label_hist)

            # Sync single class mode from PR#106
            if self.single_class_mode.isChecked() and self.lastLabel:
                text = self.lastLabel
            else:
                text = self.label_dialog.pop_up(text=self.prev_label_text)
                self.lastLabel = text
        else:
            text = self.default_label_text_line.text()

        # Add Chris
        self.diffc_button.setChecked(False)
        if text is not None:
            self.prev_label_text = text
            generate_color = generate_color_by_text(text)
            shape = self.canvas.set_last_label(text, generate_color, generate_color)
            self.add_label(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.set_editing(True)
                self.actions.create.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.set_dirty()

            if text not in self.label_hist:
                self.label_hist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.reset_all_lines()






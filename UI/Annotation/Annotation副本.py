from PyQt5 import QtWidgets
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QRect, Qt, QSettings
from project.utils import *
from UI.Annotation.AnnotationWin1 import AnnotationWin
from UI.Canvas import Canvas
from utils.CommonHelper import CommonHelper
import os
import numpy
import cv2

class Annotation(AnnotationWin):
    def __init__(self,configuration):
        super(Annotation, self).__init__()
        self.setupUi(self)
        styleFile = '../resource/Annotation.qss'
        # 换肤时进行全局修改，只需要修改不同的QSS文件即可
        style = CommonHelper.readQss(styleFile)
        self.setStyleSheet(style)
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
        self.width=None
        self.height=None
    def open_file(self):
        self.filename = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if not self.filename:
           return None
        self.imageTotal =1
        self.load_image(self.filename[0])

    def load_image(self,file):
        self.img = cv2.imread(file)
        self.image = QtGui.QImage( self.img.data,  self.img.shape[1],  self.img.shape[0],
                                    QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.canvas.load_pixmap(QtGui.QPixmap.fromImage(self.image))
        # self.img_label.setPixmap()
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
           self.img_label.x0 = 0
           self.img_label.x1 = 0
           self.img_label.y0 = 0
           self.img_label.y1 = 0
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











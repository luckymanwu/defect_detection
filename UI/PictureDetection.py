import os
import time
from PIL import Image, ImageQt
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PictureDetectionMainWin import Ui_PicDetectionMainWindow
from PyQt5 import QtCore, QtWidgets, QtGui
import TopBar
import UIdetect

class PictureDetectionWin(QtWidgets.QMainWindow,Ui_PicDetectionMainWindow):
    def __init__(self):
        super(PictureDetectionWin, self).__init__()
        self.setupUi(self)
        self.actionOpen.triggered.connect(self.open_img)
        self.actionrun.triggered.connect(self.start_detection)
        self.actionreal_time_detection.triggered.connect(self.change_detection_mode)
        self.actionpocket_mouth.triggered.connect(self.pocket_mouth_mode)
        self.actionimport.triggered.connect(self.change_weight)
        self.actionmagentic_core.triggered.connect(self.magentic_core_mode)
        self.actionbatch_detection.triggered.connect(self.changeToBatchDetectionMode)
        self.actiondefault_mode.triggered.connect(self.default_mode)
        self.weights='../weights/best.pt'
        self.names='../data/rbc.names'
        self.cfg='cfg/yolov3-tiny.cfg'
    def change_weight(self):
       self.wegiht = QFileDialog.getOpenFileName(self, "加载模型", "", "*.pt")
       if(self.weight =='../weights/best.pt'):
          QMessageBox.information(self, "提示", "模型导入失败", QMessageBox.Yes)
       else:
          QMessageBox.information(self, "提示", "模型导入成功", QMessageBox.Yes)
    def magentic_core_mode(self):
        self.weights='../weights/best.pt'
        QMessageBox.information(self, "提示", "磁芯检测模式", QMessageBox.Yes)

    def default_mode(self):
        self.weights = '../weights/yolov3-tiny.weights'
        self.names='../data/coco.names'
        self.cfg ='../cfg/yolov3-tiny-original.cfg'
        QMessageBox.information(self, "提示", "coco数据集模式", QMessageBox.Yes)

    def pocket_mouth_mode(self):
        self.weights=''
        QMessageBox.information(self, "提示", "袋口检测模式", QMessageBox.Yes)
    def open_img(self):
        self.imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        self.origin = QtGui.QPixmap(self.imgName)
        jpg = self.origin.scaled(self.original_img.width(), self.original_img.height())
        self.original_img.setPixmap(jpg)


    def start_detection(self):
        qIamge =  self.origin.toImage()
        image = ImageQt.fromqimage(qIamge)
        pre_savename = '../data/samples'
        img_path = self.imgName.split("/")
        img_name = img_path[len(img_path)-1]
        savename = os.path.join(pre_savename, img_name)
        image.save(savename)
        defect_type = UIdetect.startDetect(savename,self.weights,self.names,self.cfg)
        pre_dirpath = '../output/'
        dirpath = os.path.join(pre_dirpath, img_name)
        img_Out = Image.open(dirpath)  # 读取数据
        self.detection_img.setPixmap(QtGui.QPixmap.fromImage(ImageQt.ImageQt(img_Out)).scaled(self.detection_img.width(), self.detection_img.height()))
        self.detection_result.setText(defect_type[img_name])

    def change_detection_mode(self):
        TopBar.changeToDetectionMode(self)

    def changeToBatchDetectionMode(self):
        TopBar.changeToBatchDetectionMode(self)
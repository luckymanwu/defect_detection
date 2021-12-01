import os
from model.Opt import Opt
import cv2
from PyQt5.QtGui import QStandardItem
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication
from PyQt5 import QtGui
from Service.hkDetect import hkDetect
import glob
from utils.CommonHelper import CommonHelper
from UI.PictureDetection.PictureDetectionWin import PictureDetectionWin
from model.QclickableImage import QClickableImage
from PyQt5.QtCore import QThread, pyqtSignal


class PictureDetection(PictureDetectionWin):
    def __init__(self,configuration):
        super(PictureDetection, self).__init__()
        self.setupUi(self)
        styleFile = '../resource/PictureDetection.qss'
        # 换肤时进行全局修改，只需要修改不同的QSS文件即可
        style = CommonHelper.readQss(styleFile)
        self.imgName=[]
        self.configuration =configuration
        self.setStyleSheet(style)
        self.cwd = os.getcwd()  # 获取当前程序文件位置
        self.opt = Opt()
        self.opt.cfg = self.configuration.value('CFG_PATH')
        self.opt.output = self.configuration.value('SAVE_IMG_PATH')
        self.opt.weights = self.configuration.value('WEIGHTS_PATH')
        self.opt.names = self.configuration.value('SAVE_DATASET_PATH')+"\\rbc.names"
        self.opt.confidence = self.configuration.value('CONFIDENCE')
        self.thread = PictureDetectThread(self.opt)
        self.open.clicked.connect(self.opne_file)
        self.start.clicked.connect(self.start_batch_detection)
        self.resultView.doubleClicked.connect(self.file_item_double_clicked)
        self.thread.show_picture_signal.connect(self.showPicture)
        self.thread.addItem_signal.connect(self.addItem)
        self.thread.show_total_signal.connect(self.show_total)
        self.start.setEnabled(False)
        self.stop.setEnabled(False)
        self.badNum=0


    def opne_file(self):
        file_path = QFileDialog.getExistingDirectory(self, "选取文件夹", self.cwd)
        for i in range(self.gridLayout.count()):
            self.gridLayout.itemAt(i).widget().deleteLater()
        if file_path:
             self.images_path = glob.glob(os.path.join(file_path + '/*.*'))
             photo_num = len(self.images_path)
             if photo_num != 0:
                 for i in range(photo_num):
                    image_id = self.images_path[i]
                    split_img_path = image_id.split("/")
                    img_name = split_img_path[len(split_img_path)-1]
                    self.imgName.append(img_name)
                    pixmap = QtGui.QPixmap(image_id)
                    clickable_image = QClickableImage(440, 300, pixmap,img_name)
                    self.gridLayout1.addWidget(clickable_image, i , 1)
                    QApplication.processEvents()
                 self.model.clear()
             else:
                QMessageBox.information(self, '提示', '该文件夹图片为空')
        else:
            QMessageBox.information(self, '提示', '请先选择根文件夹')

        self.start.setEnabled(True)


    def start_batch_detection(self):
        self.stop.setEnabled(True)
        self.thread.images_path = self.images_path
        self.thread.imgName = self.imgName
        self.thread.start()


    def showPicture(self,num,result_image):
        img_name = self.imgName[num]
        result_image = QtGui.QImage(result_image.data, result_image.shape[1], result_image.shape[0],
                                    QtGui.QImage.Format_RGB888)
        result_image = QtGui.QPixmap.fromImage(result_image)
        clickable_image = QClickableImage(440, 300, result_image, img_name)
        self.gridLayout.addWidget(clickable_image, num, 1)

    def addItem(self,img_name,defect_type):
        self.model.appendRow([QStandardItem(img_name), QStandardItem(defect_type)])

    def show_total(self,imgs,badNum):
        self.model.appendRow([QStandardItem("total: " + str(len(imgs))), QStandardItem(
            "瑕疵品: " + str(self.badNum) + "  瑕疵率: " + str((badNum / len(imgs)) * 100) + "%")])



    def file_item_double_clicked(self,item=None):
        self.cur_img_idx = self.resultView.currentIndex()
        self.cur_img_idx = self.cur_img_idx.row()
        filename = self.imgName[self.cur_img_idx]
        if filename:
            height = 360*self.cur_img_idx
            self.origin_scrollArea.verticalScrollBar().setSliderPosition(height)
            self.detection_scrollArea.verticalScrollBar().setSliderPosition(height)


class PictureDetectThread(QThread):
    show_picture_signal = pyqtSignal(object,object)
    addItem_signal = pyqtSignal(str,str)
    show_total_signal = pyqtSignal(object,int)
    def __init__(self,opt=None):
        super(PictureDetectThread, self).__init__()
        self.images_path = None
        self.imgName = None
        self.opt = opt
    def run(self):
        self.hkDetect = hkDetect(self.opt)
        self.badNum = 0
        imgs = []
        for image_path in self.images_path:
            print(image_path)
            img = cv2.imread(image_path)
            imgs.append(img)
        for i in range(len(imgs)):
            img = imgs[i]
            result_image, defect_type = self.hkDetect.detect(img, self.opt)
            img_name = self.imgName[i]
            self.show_picture_signal.emit(i,result_image)
            if ("good" not in defect_type):
                self.badNum += 1
            self.addItem_signal.emit(img_name,defect_type)
        self.show_total_signal.emit(imgs,self.badNum)




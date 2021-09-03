import os
import time
from Opt import Opt
import cv2
import numpy as np
from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QStandardItem, QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene, QGraphicsPixmapItem, QApplication
from PyQt5 import QtCore, QtWidgets, QtGui
from hkDetect import hkDetect
import glob
from utils.CommonHelper import CommonHelper
from UI.PictureDetection.PictureDetectionWin import PictureDetectionWin
from UI.QclickableImage import QClickableImage

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
        self.open.clicked.connect(self.opne_file)
        self.start.clicked.connect(self.start_batch_detection)
        self.badNum=0


    def opne_file(self):
        file_path = QFileDialog.getExistingDirectory(self, "选取文件夹", self.cwd)
        for i in range(self.gridLayout.count()):
            self.gridLayout.itemAt(i).widget().deleteLater()
        if file_path:
             self.images_path = glob.glob(os.path.join(file_path + '/*.jpg'))
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
             else:
                QMessageBox.information(self, '提示', '该文件夹图片为空')
        else:
            QMessageBox.information(self, '提示', '请先选择根文件夹')



    def start_batch_detection(self):
        self.badNum = 0
        self.opt = Opt()
        self.opt.cfg = self.configuration.value('CFG_PATH')
        self.opt.output = self.configuration.value('SAVE_IMG_PATH')
        self.opt.weights = self.configuration.value('WEIGHTS_PATH')
        self.hkDetect = hkDetect(self.opt)
        imgs = []
        for image_path in self.images_path:
            img = cv2.imread(image_path)
            imgs.append(img)
        # 将List转化为numpy数组，也可使用dtype=np.float32
        # imgs = np.array(imgs, dtype=float)

        for i in range(len(imgs)):
            img = imgs[i]
            result_image, defect_type = self.hkDetect.detect(img, self.opt)
            img_name = self.imgName[i]
            result_image = QtGui.QImage(result_image.data, result_image.shape[1], result_image.shape[0], QtGui.QImage.Format_RGB888)
            result_image = QtGui.QPixmap.fromImage(result_image)
            clickable_image = QClickableImage(440, 300, result_image, img_name)
            self.gridLayout.addWidget(clickable_image, i, 1)
            if ("ok" not  in defect_type):
                self.badNum+=1
            self.model.appendRow([QStandardItem(img_name), QStandardItem(defect_type)])
        self.model.appendRow([QStandardItem("total: " + str(len(imgs))), QStandardItem(
                "瑕疵品: " + str(self.badNum) + "  瑕疵率: " + str((self.badNum / len(imgs)) * 100) + "%")])





    # def change_detection_mode(self):


        # def addImage(self, pixmap, image_id):
        #     ##获取图片列数
        #     nr_of_columns = self.get_nr_of_image_columns()
        #     nr_of_widgets = self.gridLayout.count()
        #     self.max_columns = nr_of_columns
        #     if self.col < self.max_columns:
        #         self.col += 1
        #     else:
        #         self.col = 0
        #         self.row += 1
        #     clickable_image = QClickableImage(self.display_image_size, self.display_image_size, pixmap, image_id)
        #     clickable_image.clicked.connect(self.on_left_clicked)
        #     clickable_image.rightClicked.connect(self.on_right_clicked)
        #     self.gridLayout.addWidget(clickable_image, self.row, self.col)


# def get_nr_of_image_columns(self):
#     # 展示图片的区域，计算每排显示图片数。返回的列数-1是因为我不想频率拖动左右滚动条，影响数据筛选效率
#     scroll_area_images_width = int(0.68 * self.width)
#     if scroll_area_images_width > self.display_image_size:
#
#         pic_of_columns = scroll_area_images_width // self.display_image_size  # 计算出一行几列；
#     else:
#         pic_of_columns = 1
#
#     return pic_of_columns - 1

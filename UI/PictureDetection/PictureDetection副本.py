import os
import time

import cv2
from PIL import Image, ImageQt
from PyQt5.QtGui import QStandardItem, QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene, QGraphicsPixmapItem, QApplication
from PyQt5 import QtCore, QtWidgets, QtGui
import UIdetect
import glob
from utils.CommonHelper import CommonHelper
from UI.PictureDetection.PictureDetectionWin import PictureDetectionWin
from UI.QclickableImage import QClickableImage

class PictureDetection(PictureDetectionWin):
    def __init__(self):
        super(PictureDetection, self).__init__()
        self.setupUi(self)
        styleFile = '../resource/PictureDetection.qss'
        # 换肤时进行全局修改，只需要修改不同的QSS文件即可
        style = CommonHelper.readQss(styleFile)
        self.setStyleSheet(style)
        # self.actionOpen.triggered.connect(self.opne_file)
        # self.actionrun.triggered.connect(self.start_batch_detection)
        self.cwd = os.getcwd()  # 获取当前程序文件位置
        self.wegiht = '../weights/best.pt'
        # self.actionreal_time_detection.triggered.connect(self.change_realtime_mode)
        # self.actionpicture_detection.triggered.connect(self.change_pic_mode)
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
                    img_name = split_img_path[len(split_img_path) - 1]
                    pixmap = QtGui.QPixmap(image_id)
                    clickable_image = QClickableImage(440, 260, pixmap,img_name)
                    self.gridLayout1.addWidget(clickable_image, i , 1)
                    QApplication.processEvents()
             else:
                QMessageBox.information(self, '提示', '该文件夹图片为空')
        else:
            QMessageBox.information(self, '提示', '请先选择根文件夹')

        # self.images_path = glob.glob(os.path.join( file_path + '/*.jpg'))  # 所有图片路径
        # QMessageBox.information(self, "提示", "数据导入成功", QMessageBox.Yes )
        # for image_path in self.images_path:
        #     img = cv2.imread(image_path)  # 读取图像
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        #     x, y = img.shape[:2]
        #     size = (int(x), int(y * 0.45))
        #     dst = cv2.resize(img, size,interpolation=cv2.INTER_AREA)
        #     frame = QImage(dst, x, y * 0.45, QImage.Format_RGB888)
        #     pix = QPixmap.fromImage(frame)
        #     self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        #     self.scene.addItem(self.item)
        # self.originalView.setScene(self.scene)
    def start_batch_detection(self):
        self.bad_num = 0
        i=0
        for image_path in self.images_path:
            i+=1
            self.total = len(self.images_path)
            defect = UIdetect.startDetect(image_path,self.wegiht)
            split_img_path = image_path.split("/")
            img_name = split_img_path[len(split_img_path) - 1]
            pre_dirpath = '../output/'
            dirpath = os.path.join(pre_dirpath, img_name)
            # img_Out = Image.open(dirpath)  # 读取数据
            # self.show_label.setPixmap(QtGui.QPixmap.fromImage(ImageQt.ImageQt(img_Out)).scaled(self.show_label.width(), self.show_label.height()))
            pixmap = QtGui.QPixmap(dirpath)
            clickable_image = QClickableImage(440, 260, pixmap, img_name)
            self.gridLayout.addWidget(clickable_image, i, 1)
            split_defect_type =  defect[img_name].split("_")
            defect_type=split_defect_type[len(split_defect_type)-1]
            if(defect_type!="ok"):
                self.bad_num+=1
            self.model.appendRow([QStandardItem(img_name), QStandardItem(defect[img_name])])
        self.model.appendRow([QStandardItem("total: "+str(self.total)), QStandardItem("瑕疵品: "+str(self.bad_num)+"  瑕疵率: "+str((self.bad_num/self.total)*100)+"%")])

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

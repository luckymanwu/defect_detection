from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QIcon, QPixmap
from MainWin import Ui_MainWindow
from PIL import Image, ImageQt
import TopBar
import os
import UIdetect
import sys
import cv2
import time
class DetectionWin(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(DetectionWin,self).__init__()
        self.setupUi(self)
        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()  # 视频流
        self.originalshowImage=None
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头
        self.timer_camera.timeout.connect(self.show_camera) #若定时器结束，则调用show_camera()
        self.checkBox.toggled.connect(self.checkBoxStatus)
        self.checkBox_2.toggled.connect(self.checkBoxStatus)
        # self.actionDetection.triggered.connect(self.DetectionWinShow)
        self.actionTrain.triggered.connect(self.TrainWinShow)
        self.actionMark.triggered.connect(self.MarkWinShow)
        self.actionpicture_detection.triggered.connect(self.changeDetectionMode)
        self.actionreal_time_detection.triggered.connect(self.changeDetectionMode)
        self.actionbatch_detection.triggered.connect(self.changeToBatchDetectionMode)
    def changeDetectionMode(self):
        TopBar.changeToPicDetectionMode(self)

    def changeToBatchDetectionMode(self):
        TopBar.changeToBatchDetectionMode(self)
    def TrainWinShow(self):
       TopBar.switchTrain(self)

    def MarkWinShow(self):
        TopBar.switchAnnotation(self)

    def checkBoxStatus(self):
       if(self.checkBox.isChecked() | self.checkBox_2.isChecked()):
           self.carmerOpen()
       else:
           self.carmerClose()


    def carmerOpen(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(300)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示

    def carmerClose(self):
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label.clear()  # 清空视频显示区域
            self.label_2.clear()  # 清空视频显示区域

    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读取
        originalshow = cv2.resize( self.image, (400,360))
        originalshow = cv2.cvtColor(originalshow, cv2.COLOR_BGR2RGB)
        self.originalshowImage = QtGui.QImage(originalshow.data, originalshow.shape[1], originalshow.shape[0],
                                QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.originalshowImage))  # 往显示视频的Label里 显示QImage
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(self.originalshowImage))  # 往显示视频的Label里 显示QImage
        # #嵌入
        originalshowImage=ImageQt.fromqimage( self.originalshowImage)
        pre_savename = '../data/samples'
        now_time = str(int(time.time()))+".jpg"
        savename = os.path.join(pre_savename, now_time)
        originalshowImage.save(savename)
        defect_type=UIdetect.startDetect(savename)
        pre_dirpath='../output/'
        dirpath=os.path.join(pre_dirpath, now_time)
        img_Out = Image.open(dirpath)  # 读取数据
        self.carmer_detection_label.setPixmap(QtGui.QPixmap.fromImage(ImageQt.ImageQt(img_Out)))
        self.carmer_detection_label1.setPixmap(QtGui.QPixmap.fromImage(ImageQt.ImageQt(img_Out)))
        self.show_defect_type_label.setText(defect_type[now_time])
        self.show_defect_type_label1.setText(defect_type[now_time])

if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    window=DetectionWin()
    icon = QIcon()
    icon.addPixmap(QPixmap("./logo.ico"))
    window.setWindowIcon(icon)
    window.setWindowTitle('缺陷检测系统')
    window.show()
    app.exec_()


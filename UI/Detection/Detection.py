import sys
import time
import os
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QIcon, QPixmap, QStandardItem
from UI.Detection.D2 import DetectionWin
from UI.Setting.Setting import Setting
from utils.CommonHelper import CommonHelper
from hkDetect import hkDetect
import numpy as np
import cv2
from Opt import Opt
from utils.Camera import Camera

from  MvCameraControl_class import *
winfun_ctype = WINFUNCTYPE
DETECTION_THRESHOLD = 4000
OUT_PATH = "../../output/Image/"

class Detection(DetectionWin):
    deviceList = MV_CC_DEVICE_INFO_LIST()
    cam = MvCamera()
    def __init__(self,configuration):
        super(Detection,self).__init__()
        self.setupUi(self)
        styleFile = '../resource/Detection.qss'
        # 换肤时进行全局修改，只需要修改不同的QSS文件即可
        style = CommonHelper.readQss(styleFile)
        self.setStyleSheet(style)
        self.cams={}
        self.configuration = configuration
        self.g_bExit = False
        self.cameraDetectionLabels = {"1":self.camera1_detection_label,"2":self.camera2_detection_label,"3":self.camera3_detection_label,"4":self.camera4_detection_label}
        self.cameraResultLabels = {"1":self.camera1_result_label,"2":self.camera2_result_label,"3":self.camera3_result_label,"4":self.camera4_result_label}
        self.GreenButton.clicked.connect(self.detect)
        self.RedButton.clicked.connect(self.stop)
        self.BlueButton.clicked.connect(self.reset)

    def detect(self):
        self.opt = Opt()
        self.opt.cfg = self.setting.cfg_path
        self.opt.output = self.setting.save_img_path
        self.opt.weights = self.setting.weights_path
        self.hkDetect = hkDetect(self.opt)
        for camNo in self.configuration.value('CAM_LIST'):
            cam = Camera(camNo,self.work_thread,self.opt)
            self.cams.update({camNo: cam})
            cam.openCam()



    def stop(self):
        for key in self.cams:
            self.cams[key].g_bExit = True
            self.cams[key].closeCam()

    def reset(self):
        self.speedValue = 0.0
        self.ratio = 0.0
        self.detect_num = 0
        self.good_num = 0
        self.hkDetect = None
        self.detection_number_value.setText(str(self.detect_num) + "个")
        self.speed_value.setText(str(self.speed) + "个/分钟")
        self.ratio_value.setText(str(self.ratio) + "%")


    def image_show(self,image,camNo):
        camera = self.cams[camNo]
        image = cv2.resize(image, (self.opt.width, self.opt.height))
        originalshow = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        area = self.hkDetect.ostu(originalshow)
        if(area>DETECTION_THRESHOLD):
            camera.imgs.append(originalshow)
        elif(len(camera.imgs)):
            camera.detect_num+=1
            select_img = camera.imgs[int(len(camera.imgs)/2)]
            result_image,defect_classes = self.hkDetect.detect(select_img, self.opt)

            if(len(defect_classes)==1 | "ok" in defect_classes ):
                camera.good_num+=1
                self.cameraResultLabels[camNo].setStyleSheet("color:green;font-size:14px")
            else:
                 camera.bad_num+=1
                 self.cameraResultLabels[camNo].setStyleSheet("color:red;font-size:14px")
            result_image = QtGui.QImage(result_image.data, result_image.shape[1], result_image.shape[0],
                                    QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.cameraDetectionLabels[camNo].setPixmap(QtGui.QPixmap.fromImage(result_image))
            defect_type = ''.join(defect_classes)
            self.cameraResultLabels[camNo].setText(defect_type)
            row = int(camNo)-1
            self.model.setItem(row,1, QStandardItem(str(camera.detect_num)))
            self.model.setItem(row,2, QStandardItem(str(camera.good_num)))
            self.model.setItem(row,3, QStandardItem(str(camera.bad_num)))
            self.model.setItem(row,4, QStandardItem(str((camera.bad_num/camera.detect_num)*100)+"%"))
            camera.imgs.clear()


    def work_thread(self, cam=0, pData=0, nDataSize=0, camNo=0):
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
            if ret == 0:

                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
                image = np.asarray(pData).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                self.image_show(image,camNo)
            else:
                print("no data[0x%x]" % ret)
            if self.cams[camNo].g_bExit == True:
                break



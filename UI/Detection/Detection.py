import threading

from PyQt5 import QtGui
from PyQt5.QtGui import QStandardItem

from Service.Serial import COM, com
from UI.Detection.DetectionWin import DetectionWin
from UI.Ticket.Ticket import Ticket
from utils.CommonHelper import CommonHelper
import numpy as np
from model.Opt import Opt
from model.Camera import Camera
from  MvCameraControl_class import *
import time
import matplotlib.pyplot as plt
# winfun_ctype = WINFUNCTYPE
DETECTION_THRESHOLD = 4000

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
        self.Ticket_flag = False
        self.Ticket = Ticket()
        self.Ticket.ticket_signal.connect(self.ticket_info_show)
        self.configuration = configuration
        self.g_bExit = False
        self.cameraDetectionLabels = {"1":self.camera1_detection_label,"2":self.camera2_detection_label,"3":self.camera3_detection_label,"4":self.camera4_detection_label}
        self.cameraResultLabels = {"1":self.camera1_result_label,"2":self.camera2_result_label,"3":self.camera3_result_label,"4":self.camera4_result_label}
        self.GreenButton.clicked.connect(self.detect)
        self.RedButton.clicked.connect(self.stop)
        self.BlueButton.clicked.connect(self.reset)
        self.grayButton.clicked.connect(lambda : self.Ticket.show())
        com.open()



    def detect(self):
        if(not self.Ticket_flag):
            self.Ticket.show()
        else:
            # threading.Thread(target=self.speed_update).start()
            self.grayButton.setEnabled(False)
            opt = Opt()
            opt.cfg = self.configuration.value('CFG_PATH')
            opt.output = self.configuration.value('SAVE_IMG_PATH')
            opt.weights = self.configuration.value('WEIGHTS_PATH')
            for camNo in self.configuration.value('CAM_LIST'):
                cam = Camera(camNo,opt,1)
                cam.show_picture_signal.connect(self.image_show)
                cam.show_detect_info.connect(self.show_detect_info)
                self.cams.update({camNo: cam})
                cam.openCam()
            self.grayButton.setEnabled(False)
            self.grayButton.setEnabled(False)

    def stop(self):
        self.grayButton.setEnabled(True)
        self.Ticket_flag = False
        # self.com.close()
        detect_total = 0
        bad_total =0
        good_total=0
        defectStatistic = {}
        for no in self.cams:
            detect_total += self.cams[no].detect_num
            bad_total += self.cams[no].bad_num
            good_total += self.cams[no].good_num
            for key,value in self.cams[no].defectStatistic.items():
                if key in defectStatistic:
                    defectStatistic[key] += value
                else:
                    defectStatistic[key] = value
            self.cams[no].g_bExit = True
            self.cams[no].closeCam()
        # self.generate_report(detect_total,bad_total,good_total,defectStatistic)


    def generate_report(self,detect_total,bad_total,good_total,defectStatistic):
        if(bad_total == 0):
            return
        plt.figure()
        plt.rcParams["font.family"] = "kaiti"
        plt.suptitle("检测报告", fontsize=20)
        plt.subplot(121)
        labels = ["检测数", "良品数", "瑕疵数"]
        data = [detect_total, good_total, bad_total]
        plt.title("检测表")
        plt.bar(range(len(data)), data, tick_label=labels)

        ratio = []
        defect_class = []
        dist = []
        for key, value in defectStatistic.items():
            defect_class.append(key)
            ratio.append(int(value / bad_total * 100))
            dist.append(0)
        plt.subplot(122)

        plt.title("各类缺陷占比", size=20)
        plt.pie(ratio, labels=defect_class, autopct="%.1f%%", explode=dist, shadow=True)
        plt.savefig("result-"+self.product_id_label.text()+'.jpg')
        plt.show()


    def reset(self):
        self.speedValue = 0.0
        self.ratio = 0.0
        self.detect_num = 0
        self.good_num = 0
        self.detection_number_value.setText(str(self.detect_num) + "个")
        self.speed_value.setText(str(self.speed) + "个/分钟")
        self.ratio_value.setText(str(self.ratio) + "%")

    def image_show(self,result_image,defect_type,camNo):
            if("good" in defect_type ):
                self.cameraResultLabels[camNo].setStyleSheet("color:green;font-size:20px")
            else:
                self.cameraResultLabels[camNo].setStyleSheet("color:red;font-size:20px")
            result_image = QtGui.QImage(result_image.data.tobytes(), result_image.shape[1], result_image.shape[0],
                                    QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.cameraDetectionLabels[camNo].setPixmap(QtGui.QPixmap.fromImage(result_image))
            self.cameraDetectionLabels[camNo].setScaledContents(True)
            self.cameraResultLabels[camNo].setText(defect_type)


    def show_detect_info(self,nums,camNo):
        camera = self.cams[camNo]
        row = int(camNo) - 1
        self.model.setItem(row, 1, QStandardItem(str(nums[0])))
        self.model.setItem(row, 2, QStandardItem(str(nums[1])))
        self.model.setItem(row, 3, QStandardItem(str(nums[2])))
        self.model.setItem(row, 4, QStandardItem(str(int(nums[2] / nums[0]) * 100) + "%"))
        

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


    def ticket_info_show(self,productName,productId):
        self.Ticket_flag = True
        self.product_name_label.setText(productName)
        self.product_id_label.setText(productId)
        local_times = time.localtime(time.time())
        self.detect_time_label.setText(time.strftime("%Y-%m-%d %H:%M",local_times))

    # def speed_update(self):
    #     while True:
    #         speed= com.get_data(50)
    #         if(speed!=""):
    #             speed = float(speed)/10
    #             self.speed_label.setText(str(speed)+" cm/s")




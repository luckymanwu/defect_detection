import os

import numpy as np
from PIL import Image, ImageStat
import multiprocessing
from  MvCameraControl_class import *
from PyQt5.QtCore import pyqtSignal, QObject
from Service.hkDetect import hkDetect
import threading
import cv2
import time
from Service.pre_treat import roi_split
from project.parse_config import parse_rbc_name

winfun_ctype = WINFUNCTYPE
stFrameInfo = POINTER(MV_FRAME_OUT_INFO_EX)
pData = POINTER(c_ubyte)
FrameInfoCallBack = winfun_ctype(None, pData, stFrameInfo, c_void_p)
# DETECTION_THRESHOLD = 4000
deviceList = MV_CC_DEVICE_INFO_LIST()
tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
distance = 75.0
base_bright = 193
base_contrast = 0.62
class Camera(QObject):
    show_picture_signal = pyqtSignal(object, object, object,str)
    show_detect_info = pyqtSignal(object,object)
    sample_signal = pyqtSignal(object,object,object)
    def __init__(self,camNo,opt,slot):
        super(Camera, self).__init__()
        self.camNo = camNo
        # self.show_picture_signal = pyqtSignal(object, object, object)
        self.detect_num = 0
        self.bad_num = 0
        self.good_num = 0
        self.defectStatistic={}
        lambda :self.defectStatistic_init(opt.names)
        self.speedValue=0.0
        self.ratio = 0.0
        self.opt = opt
        self.imgs = []
        if slot == 1:
         self.hkDetect = hkDetect(opt)
        self.g_bExit = False
        self.slot = slot
        self.cache_img = []

    def defectStatistic_init(self, path=None):
        if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
            path = 'data' + os.sep + path

        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            self.defectStatistic[line]=0


    def openCam(self):
        self.start = time.time()
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        self.cam = MvCamera()
        # ch:选择设备并创建句柄 | en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[int(self.camNo)-1], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        # ch:打开设备 | en:Open device
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # ch:设置触发模式为ON | en:Set trigger mode as off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
        if ret != 0:
            print("set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        # ret = self.cam.MV_CC_SetFloatValue("TriggerDelay", 2350000.0)
        ret = self.cam.MV_CC_SetFloatValue("TriggerDelay", 1500000.0)

        if ret != 0:
            print("set trigger delay fail! ret[0x%x]" % ret)
            sys.exit()

        ret = self.cam.MV_CC_SetBoolValue("TriggerCacheEnable",True)
        if ret != 0:
            print("set trigger cache enable fail! ret[0x%x]" % ret)
            sys.exit()
        ret = self.cam.MV_CC_SetEnumValue("Gain",2)

        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", 1100.0)
        if ret != 0:
            print("set Exposure Time fail! ret[0x%x]" % ret)
            sys.exit()
        ret = self.cam.MV_CC_SetEnumValue("Gain", 2)
        # ch:获取数据包大小 | en:Get payload size
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print("get payload size fail! ret[0x%x]" % ret)
            sys.exit()
        nPayloadSize = stParam.nCurValue

            # ch:开始取流 | en:Start grab image
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("start grabbing fail! ret[0x%x]" % ret)
            sys.exit()

        data_buf = (c_ubyte * nPayloadSize)()

        try:
            if(self.slot==1):
                 multiprocessing.Process(target=self.read_work_thread, args=(self.cam, data_buf, nPayloadSize, self.camNo)).start()
                 multiprocessing.Process(target=self.detect_work_thread, args=(self.cam, data_buf , nPayloadSize,self.camNo)).start()
            else:
                 threading.Thread(target=self.train_work_thread, args=(self.cam, data_buf, nPayloadSize, self.camNo)).start()
        except:
            print("error: unable to start thread")


    def closeCam(self):
        # ch:停止取流 | en:Stop grab image
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            print("stop grabbing fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:关闭设备 | Close device
        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            print("close deivce fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:销毁句柄 | Destroy handle
        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            print("destroy handle fail! ret[0x%x]" % ret)
            sys.exit()

    def read_work_thread(self, cam=0, pData=0, nDataSize=0, camNo=0):
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
            if ret == 0:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
                image = np.asarray(pData).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.cache_img.append(image)




    def detect_work_thread(self, cam=0, pData=0, nDataSize=0, camNo=0):
        while True:
            if(len(self.cache_img)>0):
                start = time.time()
                image = self.cache_img[0]
                self.cache_img.pop(0)
                if (np.mean(image) < 255):
                    self.detect_num += 1
                    originalshow = roi_split(image)
                    cv2.imwrite(str(self.detect_num) + '.jpg', originalshow)
                    # image = cv2.resize(image, (self.opt.width, self.opt.height))
                    # ret, image = cv2.threshold(image, 45, 255, cv2.THRESH_TOZERO_INV)
                    # originalshow = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    bright = self.hkDetect.brightness(originalshow)
                    contrast = self.hkDetect.contrast(originalshow)
                    tip = ""
                    if(bright > base_bright * 1.3):
                        tip = "亮度过高"
                    elif (bright < base_bright * 0.7):
                        tip = "亮度过低"

                    if (contrast > base_contrast * 1.3):
                        tip += " 对比度过高"
                    elif (contrast < base_contrast * 0.7):
                        tip += "对比度过低"

                    detected_image, defect_type = self.hkDetect.detect(originalshow, self.opt)

                    # cv2.imwrite("./"+str(self.detect_num)+".jpg",detected_image)
                    # if (len(defect_type)==1 and defect_type[0] == "good"):
                    if (len(defect_type) == 0):
                        self.good_num+=1
                    else:
                        self.bad_num+=1

                    if(defect_type is not ''):
                        # self.defectStatistic[defect_type] =  self.defectStatistic[defect_type]+1

                        self.show_picture_signal.emit(detected_image,defect_type,camNo,tip)

                        # nums = [self.detect_num,self.good_num,self.bad_num]
                        # self.show_detect_info.emit(nums,camNo)
                    end = time.time()
                    print('run time: %s Seconds' % (end-start))


            if self.g_bExit == True:
                break


    def train_work_thread(self, cam=0, pData=0, nDataSize=0, camNo=0):
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
            if ret == 0:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
                image = np.asarray(pData).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if (np.mean(image) < 245):
                    originalshow = roi_split(image)
                    bright = self.brightness(originalshow)


                    self.sample_signal.emit(originalshow, camNo,bright)

            else:
                print("no data[0x%x]" % ret)
            if self.g_bExit == True:
                break


    def brightness(self,image):
        image = Image.fromarray(image)
        stat = ImageStat.Stat(image)
        return stat.mean[0]


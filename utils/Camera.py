import numpy as np
from  MvCameraControl_class import *
from PyQt5.QtCore import QSettings
import hkDetect
from UI.Setting.Setting import Setting
from PyQt5 import  QtGui
from Opt import Opt
from hkDetect import hkDetect
import threading
import cv2
import time
import msvcrt
winfun_ctype = WINFUNCTYPE
stFrameInfo = POINTER(MV_FRAME_OUT_INFO_EX)
pData = POINTER(c_ubyte)
FrameInfoCallBack = winfun_ctype(None, pData, stFrameInfo, c_void_p)
DETECTION_THRESHOLD = 4000
deviceList = MV_CC_DEVICE_INFO_LIST()
tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

class Camera:

    def __init__(self,camNo,work_thread,opt):
        # self.CALL_BACK_FUN = FrameInfoCallBack(self.image_callback)
        self.camNo = camNo
        self.detect_num = 0
        self.bad_num = 0
        self.good_num = 0
        self.speedValue=0.0
        self.ratio = 0.0
        self.opt = opt
        self.imgs=[]
        self.hkDetect = None
        self.g_bExit = False
        self.work_thread = work_thread


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

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print("set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

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
            hThreadHandle = threading.Thread(target=self.work_thread, args=(self.cam, data_buf , nPayloadSize,self.camNo))
            hThreadHandle.start()
        except:
            print("error: unable to start thread")

        # while(self.g_bExit):
        #     hThreadHandle.join()

        # # ch:注册抓图回调 | en:Register image callback
        # ret = self.cam.MV_CC_RegisterImageCallBackEx(self.CALL_BACK_FUN,None)
        # if ret != 0:
        #     print("register image callback fail! ret[0x%x]" % ret)
        #     sys.exit()
        #
        # # ch:开始取流 | en:Start grab image
        # ret = self.cam.MV_CC_StartGrabbing()
        # if ret != 0:
        #     print("start grabbing fail! ret[0x%x]" % ret)
        #     sys.exit()
        # print("press a key to stop grabbing.")
        # msvcrt.getch()



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

    # def image_callback(self,pData, pFrameInfo, pUser):
    #     img_buff = None
    #     stFrameInfo = cast(pFrameInfo, POINTER(MV_FRAME_OUT_INFO_EX)).contents
    #     if stFrameInfo:
    #         print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
    #         stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
    #     if img_buff is None:
    #         img_buff = (c_ubyte * stFrameInfo.nWidth * stFrameInfo.nHeight)()
    #         cdll.msvcrt.memcpy(byref(img_buff), pData, stFrameInfo.nWidth * stFrameInfo.nHeight)
    #     data = np.frombuffer(img_buff, count=int(stFrameInfo.nWidth * stFrameInfo.nHeight), dtype=np.uint8)
    #     self.image_control(data=data, stFrameInfo=stFrameInfo)
    #
    # def image_control(self,data, stFrameInfo):
    #     image = data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
    #     self.image_show(image=image)

    def image_show(self, image):
        # self.hkDetect = hkDetect(self.opt)
        image = cv2.resize(image, (self.opt.width, self.opt.height))
        self.originalshowImage = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                                               QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.camera_detection_label.setPixmap(QtGui.QPixmap.fromImage(self.originalshowImage))  # 往显示视频的Label里 显示QImage
        self.camera_detection_label.repaint()
        # area = self.hkDetect.ostu(originalshow)
        # if (area > DETECTION_THRESHOLD):
        #     self.imgs.append(originalshow)
        # elif (len(self.imgs)):
        #     self.detect_num += 1
        #     select_img = self.imgs[int(len(self.imgs) / 2)]
        #     result_image, defect_type = self.hkDetect.detect(select_img, self.opt)
        #
        #     if ("ok" in defect_type):
        #         self.good_num += 1
        #         ans = self.results[0]
        #     else:
        #         num = self.detect_num - self.good_num
        #
        #     result_image = QtGui.QImage(result_image.data, result_image.shape[1], result_image.shape[0],
        #                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        #     self.camera_detection_label.setPixmap(QtGui.QPixmap.fromImage(result_image))
        #     self.camera_result_label.setText(defect_type)
        #     self.imgs.clear()

        # 为线程定义一个函数






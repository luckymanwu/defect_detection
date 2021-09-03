from UI.Setting.SettingWin1 import SettingWin
from utils.CommonHelper import CommonHelper
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QFileDialog
from  MvCameraControl_class import *
import os

deviceList = MV_CC_DEVICE_INFO_LIST()
class Setting(SettingWin):
    def __init__(self,configuration):
        super(Setting, self).__init__()
        self.setupUi(self)
        styleFile = '../resource/Setting.qss'
        # 换肤时进行全局修改，只需要修改不同的QSS文件即可
        style = CommonHelper.readQss(styleFile)
        self.setStyleSheet(style)
        self.cwd = os.getcwd()
        self.configuration= configuration
        self.camList = []
        self.weights_path=''
        self.cfg_path=''
        self.save_img_path = ''
        self.cams = {"1":self.carmer_one,"2":self.carmer_two,"3":self.carmer_three,"4":self.carmer_four,"5":self.carmer_all}
        self.nDeviceNum = deviceList.nDeviceNum
        self.carmer_one.clicked.connect(lambda: self.checkBoxStatus(str(1)))
        self.carmer_two.clicked.connect(lambda: self.checkBoxStatus(str(2)))
        self.carmer_three.clicked.connect(lambda: self.checkBoxStatus(str(3)))
        self.carmer_four.clicked.connect(lambda: self.checkBoxStatus("4"))
        self.carmer_all.clicked.connect(lambda: self.checkBoxStatus("5"))
        self.confirm.clicked.connect(self.saveSetting)
        self.weights_upload.clicked.connect(lambda: self.open(self.weights_lineEdit))
        self.cfg_upload.clicked.connect(lambda: self.open(self.cfg_lineEdit))
        self.img_path_open.clicked.connect(lambda: self.open(self.save_path_lineEdit))
        self.xml_path_open.clicked.connect(lambda: self.open(self.xml_lineEdit))
        self.initSetting()

    def initSetting(self):
        self.WidthMax_value_label.setText("2448")
        self.HeightMax_value_label.setText("2048")
        if not os.path.exists('./config.ini'):
             self.carmer_one.setChecked(True)
             self.camList.append("1")
             self.dark_radioButton.setChecked(True)
             self.weights_path = "../weights/best.pt"
             self.weights_lineEdit.setText(os.path.abspath(self.weights_path))
             self.save_img_path = "../output/Image"
             self.save_path_lineEdit.setText(os.path.abspath(self.save_img_path))
             self.cfg_path = "../cfg/yolov3-tiny.cfg"
             self.cfg_lineEdit.setText(os.path.abspath(self.cfg_path))
             self.xml_path = "../output/xml"
             self.xml_lineEdit.setText(os.path.abspath(self.xml_path))
             self.fps_LineEdit.setText("23.1400")
             self.Width_spinBox.setValue(400)
             self.Height_spinBox.setValue(360)
        else:
            self.weights_path = self.configuration.value("WEIGHTS_PATH")
            self.weights_lineEdit.setText(os.path.abspath(self.weights_path))
            self.cfg_path = self.configuration.value("CFG_PATH")
            self.cfg_lineEdit.setText(os.path.abspath(self.cfg_path))
            self.save_img_path = self.configuration.value("SAVE_IMG_PATH")
            self.save_path_lineEdit.setText(os.path.abspath(self.save_img_path))
            self.xml_path = self.configuration.value("SAVE_XML_PATH")
            self.xml_lineEdit.setText(os.path.abspath(self.xml_path))
            self.camList = self.configuration.value('CAM_LIST')
            for cam in self.camList:
                if cam == "1":
                    self.carmer_one.setChecked(True)
                elif cam == "2":
                    self.carmer_two.setChecked(True)
                elif cam == "3":
                    self.carmer_three.setChecked(True)
                elif cam == "4":
                    self.carmer_four.setChecked(True)
                elif cam == "5":
                    self.carmer_all.setChecked(True)

    def checkBoxStatus(self,cam):
        if (self.cams[cam].isChecked()):
            self.camList.append(cam)
        else:
            self.camList.remove(cam)

    def saveSetting(self):
        self.configuration.remove("CAM_LIST")
        self.configuration.setValue("CAM_LIST",self.camList)
        self.configuration.setValue("WEIGHTS_PATH",self.weights_path)
        self.configuration.setValue("CFG_PATH",self.cfg_path)
        self.configuration.setValue("SAVE_IMG_PATH",self.save_img_path)
        self.configuration.setValue("SAVE_XML_PATH",self.xml_path)


    def open(self,lineEdit):
        if lineEdit is self.save_path_lineEdit or lineEdit is  self.xml_lineEdit:
            file_path = QFileDialog.getExistingDirectory(self, "选取文件夹", self.cwd)
            lineEdit.setText(file_path)
            self.save_img_path = file_path
        else:
            fname = QFileDialog.getOpenFileName(self, 'Open file', self.cwd)
            lineEdit.setText(fname[0])
            if lineEdit is self.weights_lineEdit:
                self.weights_path = fname[0]
            elif lineEdit is self.cfg_lineEdit:
                self.cfg_path = fname[0]
















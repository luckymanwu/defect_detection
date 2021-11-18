import os
from UI.Train.DataSetWin import DataSetWin
from utils.makext import makext
from utils.voc_label import vocLabel
from utils.CommonHelper import CommonHelper
class DataSet(DataSetWin):
    def __init__(self, parent=None,dataSetPath=None):
        super(DataSet, self).__init__(parent)
        styleFile = 'E:\defect_detection-main/resource/Ticket.qss'
        # 换肤时进行全局修改，只需要修改不同的QSS文件即可
        style = CommonHelper.readQss(styleFile)
        self.setStyleSheet(style)
        self.setupUi(self)
        self.dataSetPath = dataSetPath
        self.confirm.clicked.connect(self.generateDataSet)

    def generateDataSet(self):
        self.classes = self.classes.toPlainText().split(';')
        imgpath = self.dataSetPath+'/images'
        xmlfilepath = self.dataSetPath+'/Annotations'
        txtsavepath  = self.dataSetPath+'/ImageSets'
        if not os.path.exists(txtsavepath):
            os.makedirs(txtsavepath)
        makext(imgpath,xmlfilepath,txtsavepath)
        vocLabel(self.classes,self.dataSetPath)
        self.make_rbc()
        self.tip.setText("数据集生成成功")
        self.tip.setStyleSheet("color:green;font-size:15px")
        self.close()


    def make_rbc(self):
        filename = self.dataSetPath+'/rbc.names'
        with open(filename, 'w') as file_object:
            for cls in self.classes:
                file_object.write(cls+"\n")
        filename = self.dataSetPath+'/rbc.data'
        with open(filename, 'w') as rbc_data:
            rbc_data.write("classes="+str(len(self.classes))+"\n")
            rbc_data.write("label_train="+self.dataSetPath+"/labeltrain.txt\n")
            rbc_data.write("unlabel_train=" + self.dataSetPath + "/unlabelTrain.txt\n")
            rbc_data.write("valid=" + self.dataSetPath + "/test.txt\n")
            rbc_data.write("names=" + self.dataSetPath + "/rbc.names\n")









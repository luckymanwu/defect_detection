from PictureDetection import PictureDetectionWin
from Detection import DetectionWin
from Annotaion import Annotation
from Train import TrainWin
from UI.BatchPicDetection import BatchPicDetectionWin


def changeToDetectionMode(self):
    self.hide()
    self.detectionWin = DetectionWin()
    self.detectionWin.show()
def changeToBatchDetectionMode(self):
    self.hide()
    self.BatchPicDetectionWin = BatchPicDetectionWin()
    self.BatchPicDetectionWin.show()
def changeToPicDetectionMode(self):
    self.hide()
    self.pictureDetectionWin = PictureDetectionWin()
    self.pictureDetectionWin.show()
def switchTrain(self):
    self.hide()  # 隐藏此窗口
    self.trainWin = TrainWin()
    self.trainWin.show()

def switchDetection(self):
    self.hide()  # 隐藏此窗口
    self.detectionWin = DetectionWin()
    self.detectionWin.show()
def switchAnnotation(self):
    self.hide()  # 隐藏此窗口
    self.annotation = Annotation()
    self.annotation.show()
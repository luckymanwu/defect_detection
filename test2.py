#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np

THRESH_BINARY_VALUE = 180
BLUR_KERNEL_SIZE = 5
AREA_THRESHOLD = 200
MODE = 0


def roi_split(imgdir=None):
    if MODE == 0:
        # 读入原始图像
        srcImage = cv2.imread(imgdir)
        if srcImage is None:
            print("Failed to read source image.")
            exit()

        srcImage = cv2.resize(srcImage, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

        # 分离rgb通道
        bChannel, gChannel, rChannel = cv2.split(srcImage)
        # 进行双边滤波
        bilateralImage = cv2.bilateralFilter(bChannel, BLUR_KERNEL_SIZE, 100, 15)
        # 阈值化,大于阈值的使用0表示，小于阈值的使用最大值表示
        _, binaryImage = cv2.threshold(bilateralImage, 170, 255, cv2.THRESH_TOZERO)


        # 闭运算,减少内部轮廓，避免多次检测
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, kernel)

        # cv2.waitKey(-1)
        # # 寻找轮廓
        contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


        '''
         for i in range(len(contours)):
            cv2.drawContours(srcImage, contours, i, 150, 2, 4, hierarchy, 0,)
            cv2.imshow("result1", srcImage)
        '''

        # 遍历轮廓，检测符合条件的色块形状、颜色、位置、角度
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if srcImage.shape[0] * srcImage.shape[0] * 1 / 1> area > AREA_THRESHOLD:  # 面积符合阈值
                rotateRect = cv2.minAreaRect(contours[i])  # 计算外接最小矩形
                box = cv2.boxPoints(rotateRect)
                box = np.int0(box)
                x0 = box[0, 0]
                y0 = box[0, 1]
                x1 = box[2, 0]
                y1 = box[2, 1]
                x00=int((x0+x1)/2-220)
                y00=int((y0+y1)/2-220/1.19)
                x11=int((x0+x1)/2+220)
                y11=int((y0+y1)/2+220/1.19)

                cropped = srcImage[y00:y11,x00:x11]

                if cropped.size == 0:
                    return srcImage
                else:

                    print(imgdir)
                    cv2.imwrite(imgdir, cropped)



import os

path = "C:\\Users\\Administrator\\Desktop\\cixin\\JPEGImages"
# path = "C:\\Users\\Administrator\\Desktop\\0005.jpg"
# roi_split(path)
filelist = os.listdir(path)

for file in filelist:
    Olddir = os.path.join(path, file)
    if os.path.isdir(Olddir):
        continue
    roi_split(Olddir)









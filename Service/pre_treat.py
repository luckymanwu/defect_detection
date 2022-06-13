#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np

THRESH_BINARY_VALUE = 180
BLUR_KERNEL_SIZE = 5
AREA_THRESHOLD = 200
MODE = 0


def roi_split(img=None):
    if MODE == 0:

        # srcImage = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        # # 分离rgb通道
        # bChannel, gChannel, rChannel = cv2.split(srcImage)
        # # 进行双边滤波
        # bilateralImage = cv2.bilateralFilter(bChannel, BLUR_KERNEL_SIZE, 100, 15)
        # # 阈值化
        # _, binaryImage = cv2.threshold(bilateralImage, 150, 255, cv2.THRESH_BINARY_INV)
        #
        # # 闭运算,减少内部轮廓，避免多次检测
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        # binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, kernel)
        #
        # # # 寻找轮廓
        # contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        #
        # '''
        #  for i in range(len(contours)):
        #     cv2.drawContours(srcImage, contours, i, 150, 2, 4, hierarchy, 0)
        #     cv2.imshow("result1", srcImage)
        # '''
        #
        # # 遍历轮廓，检测符合条件的色块形状、颜色、位置、角度
        # for i in range(len(contours)):
        #     area = cv2.contourArea(contours[i])
        #     if srcImage.shape[0] * srcImage.shape[0] * 1 / 1> area > AREA_THRESHOLD:  # 面积符合阈值
        #         rotateRect = cv2.minAreaRect(contours[i])  # 计算外接最小矩形
        #         box = cv2.boxPoints(rotateRect)
        #         box = np.int0(box)
        #         x0 = box[0, 0]
        #         y0 = box[0, 1]
        #         x1 = box[2, 0]
        #         y1 = box[2, 1]
        #         x00=int((x0+x1)/2-220)
        #         y00=int((y0+y1)/2-220/1.19)
        #         x11=int((x0+x1)/2+220)
        #         y11=int((y0+y1)/2+220/1.19)
        #
        #         cropped = srcImage[y00:y11,x00:x11]

        # srcImage = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        # srcImage1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, binary = cv2.threshold(srcImage1, 180, 255, cv2.THRESH_BINARY_INV)
        # contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # for i, contour in enumerate(contours):
        #     area = cv2.contourArea(contour)
        #     #print(area)
        #     if area >= 40000:
        #         x, y, w, h = cv2.boundingRect(contour)
        #         cropped = img[y - 10:y + h + 10, x - 10:x + w + 10]

        srcImage = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        srcImage1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.cv2.bilateralFilter(srcImage1, BLUR_KERNEL_SIZE, 100, 15)

        # circles1 = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 60, param1=100, param2=15, minRadius=60, maxRadius=140)
        circles1 = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 600, param1=100, param2=15, minRadius=400, maxRadius=1000)

        if circles1 is None:
            pass
        else:
            circles = circles1[0, :, :]
            circles = np.uint16(np.around(circles))
            for i in circles[:]:
                # cv2.circle(img, (circle[0], circle[1]), circle[2] - 10, 255, -1)
                cropped = img[i[1] - i[2] - 10:i[1] + i[2] + 10, i[0] - i[2] - 10:i[0] + i[2] + 10]

                if cropped.size == 0:
                    return srcImage
                else:
                    return cropped

        return srcImage

if __name__ == '__main__':
    img_path = "C:\\Users\\Administrator\\Desktop\\c.bmp"
    img = cv2.imread(img_path)
    image = roi_split(img)
    cv2.imwrite("d.jpg",image)







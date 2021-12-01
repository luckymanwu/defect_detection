import cv2

image = cv2.imread("C:\\Users\\Administrator\\Desktop\\a.bmp",0)
image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
cv2.imshow("原图",image)

# cv2.equalizeHist(image)
# cv2.imshow("均衡图",image)
# ret, thresh4 = cv2.threshold(image, 100, 255, cv2.THRESH_TOZERO_INV)

ret, thresh4 = cv2.threshold(image, 85, 255, cv2.THRESH_TOZERO_INV)
# cv2.equalizeHist(thresh4, thresh4;

cv2.imshow("result",thresh4)
cv2.waitKey(-1);
# import time
# times = time.time()
# local_times = time.localtime(times)
# local_time_asctimes= time.strftime("%Y-%m-%d %H:%M",local_times)
# print(local_time_asctimes)
#
# A = {'a': 1, 'b': 2, 'c': 3}
# B = {'b': 4, 'c': 6, 'd': 8}
# for key,value in B.items():
#    if key in A:
#      A[key] += value
#    else:
#     A[key] = value
# print(A)

import matplotlib.pyplot as plt
import numpy as np


def generate_report( detect_total, bad_total, good_total, defectStatistic):
    plt.figure()
    plt.rcParams["font.family"] = "kaiti"
    plt.suptitle("检测报告",fontsize=20)
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

    plt.savefig('result.jpg')
    plt.show()

# detect_total =100
# bad_total =10
# good_total = 90
# defectStatistic={"spot":3,"damaged":4,"repeat":6}
# generate_report(detect_total,bad_total,good_total,defectStatistic)







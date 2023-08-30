# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 14:31:22 2022

@author: salasrew
"""
import cv2
import numpy as np 
import matplotlib.pyplot as plt

def grayIntensity(img):
    count = [0] * 256
    imgs = img[:,:,0]
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            brightness = imgs[i][j]
            count[brightness] = count[brightness]+1
    return count

def zeroPadding(img):
    zeroImg = np.zeros((img.shape[0]+2,img.shape[1]+2))
    for i in range(1,img.shape[0]+1):
        for j in range(1, img.shape[1]+1):
            zeroImg[i][j] = img[i-1][j-1]
    return zeroImg

def medianFilter(img):
    n, m,_ = img.shape
    temp = img[:,:,0]
    padding = zeroPadding(temp)
    new_img = img.copy()

    for i in range(0, n - 2):
        for j in range(0, m - 2):
            a = padding[i:i + 3, j:j + 3]
            temp = np.median(a)
            new_img[i][j] = temp
    return new_img

def meanFilter(img):
    n, m,_ = img.shape
    temp = img[:,:,0]
    padding = zeroPadding(temp)
    new_img = img.copy()

    for i in range(0, n - 2):
        for j in range(0, m - 2):
            a = padding[i:i + 3, j:j + 3]
            temp = np.average(a)
            new_img[i][j] = temp
    return new_img

img = cv2.imread('noise_image.png')
img_0 = img[:,:,0]
plt.figure(figsize=(10,10))

# mean
# img_meanBlur = cv2.blur(img,(3,3))
img_meanBlur = meanFilter(img)

# median
# img_medianBlur = cv2.medianBlur(img,3)
img_medianBlur = medianFilter(img)

# zeroPadding
# img_zeroPadding = zeroPadding(img_0)

# 原始影像
plt.subplot(3,2,1)
plt.title("noise_image")
plt.imshow(img)

# 原始影像的灰階強度直方圖
plt.subplot(3,2,2)
plt.title("noise_image_his.png")
plt.hist(img_0.reshape(-1), 256,[0,256])

# 經過meanFilter後
plt.subplot(3,2,3)
# plt.title("imgMeanBlur")
plt.title("output1")
plt.imshow(img_meanBlur)

# 經過meanFilter後的灰階強度直方圖
plt.subplot(3,2,4)
plt.title("imgMeanBlur_his")
plt.title("output1_his")
img_meanBlur_0 = img_meanBlur[:,:,0]
plt.hist(img_meanBlur_0.reshape(-1), 256,[0,256])

plt.subplot(3,2,5)
# plt.title("imgMedianBlur")
plt.title("output2")
plt.imshow(img_medianBlur)

plt.subplot(3,2,6)
# plt.title("imgMedianBlur")
plt.title("output2_his")
img_medianBlur_0 = img_medianBlur[:,:,0]
plt.hist(img_medianBlur_0.reshape(-1), 256,[0,256])

plt.savefig('plot.png')

cv2.imwrite('output1.png',img_meanBlur)
cv2.imwrite('output2.png',img_medianBlur)


plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 14:31:22 2022

@author: salasrew
"""
import cv2
import numpy as np 
import matplotlib.pyplot as plt




img = cv2.imread('noise_image.png')
img_0 = img[:,:,0]
plt.figure(figsize=(10,10))

# mean
img_meanBlur = cv2.blur(img,(3,3))

# median
img_medianBlur = cv2.medianBlur(img,3)

plt.subplot(3,2,1)
plt.title("noise_image")
plt.imshow(img)

plt.subplot(3,2,2)
plt.title("noise_image_his.png")
plt.hist(img_0.reshape(-1), 256,[0,256])

plt.subplot(3,2,3)
plt.title("imgMeanBlur")
plt.imshow(img_meanBlur)

plt.subplot(3,2,4)
plt.title("noise_image_his.png")
img_meanBlur_0 = img_meanBlur[:,:,0]
plt.hist(img_meanBlur_0.reshape(-1), 256,[0,256])


plt.subplot(3,2,5)
plt.title("imgMedianBlur")
plt.imshow(img_medianBlur)

plt.subplot(3,2,6)
plt.title("noise_image_his.png")
img_medianBlur_0 = img_medianBlur[:,:,0]
plt.hist(img_medianBlur_0.reshape(-1), 256,[0,256])



plt.show()
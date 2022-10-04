# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('car.png')

# 1.Read a RGB image and write a function to convert the image to grayscale image.

# 拆通道
b,g,r = img[:,:,0],img[:,:,1],img[:,:,2]

# RGB 轉灰階公式
gray = (r*0.299 + g*0.587 + b *0.114)

cv2.imwrite("output.jpg", gray)

img_output = cv2.imread('output.jpg')

cv2.imshow('My Image', img_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Write a convolution operation with edge detection kernel
# and activation function (ReLU: rectified linear unit)

kernel = np.array([[-1,-1,-1],
                   [-1,8,-1],
                   [-1,-1,-1]])

img_gray = cv2.imread('output.jpg')

def convolution(kernel, data):
    n,m,_ = data.shape
    con_img = []
    for i in range(n-3):
        line = []
        for j in range(m-3):
            a = data[i:i+3,j:j+3]
            line.append(np.sum(np.multiply(kernel, a)))
        con_img.append(line)
    return np.array(con_img)

con_img = convolution(kernel, img_gray)

#卷積結果可視化
#cv2.imwrite("con_output.jpg", con_img)
#con_img = cv2.imread('con_output.jpg')

#cv2.imshow('My Image', con_img)

print(con_img)

plt.imshow(con_img, cmap="gray")
#plt.imsave('target.jpg',con_img,cmap='gray')
plt.axis('off')


# activation function 
def ReLU(x):
    if x<0:
        return 0
    return x 
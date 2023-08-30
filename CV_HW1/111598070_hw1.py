# -*- coding: utf-8 -*-
import cv2
import numpy as np

# Read a RGB image and write a function to convert the image to grayscale image
def rgb2gray(img):
    img_b,img_g,img_r = img[:,:,0],img[:,:,1],img[:,:,2]
    grayed = img_b*0.114 + img_g*0.587 + img_r*0.299
    cv2.imwrite("imgGray.png", grayed)
    
#2.Write a convolution operation with edge detection kernel
#and activation function (ReLU: rectified linear unit)
def convolution(kernel, data):
    n,m,_ = data.shape

    convolution_img = []
    for i in range(0,n-3):
        line = []
        for j in range(0,m-3):
            a = data[i:i+3,j:j+3] 
            temp = np.sum(np.multiply(kernel, a))
            # reLu
            temp = reLU(temp)
            line.append(temp)
        convolution_img.append(line)
    return np.array(convolution_img)


# activation function 
def reLU(x):
    if x<0:
        return 0
    return x 


def np_relu(x):
    return(np.maximum(0,x))

# Write a pooling operation with using Max pooling, 2x2 filter, and  stride 2
def pool_max(data):
    n,m,_ = data.shape
    
    pool = []
    for i in range(0,n,2):
        line = []
        for j in range(0,m,2):
            a = data[i:i+2,j:j+2] 
            line.append(np.max(a))
        pool.append(line)

    return np.array(pool,dtype='uint8')

# Write a binarization operation (threshold = 128). (>=128) set 255 (<128) set 0
def binarilize(num):
    if num>=128:
        num = 255
    else:
        num = 0   
    return num

def binarization(data):
    n,m,_ = data.shape
    
    binImg = []
    for i in range(n):
        line = []
        for j in range(m):
            line.append(binarilize(data[i,j][0]))
        binImg.append(line)
    return np.array(binImg)

#img = cv2.imread('car.png')
img = cv2.imread('liberty.png')

kernel = np.array([[-1,-1,-1],
                   [-1,8,-1],
                   [-1,-1,-1]])

imgGray = rgb2gray(img)

imgGray = cv2.imread('imgGray.png')

imgConvolution = convolution(kernel, imgGray)

cv2.imwrite('imgConvolution.png', imgConvolution)

img2 = cv2.imread('imgConvolution.png')

pool_img = pool_max(img2)
cv2.imwrite('imgMaxPooled.png', pool_img)

imgMaxPooled = cv2.imread('imgMaxPooled.png')

binImg = binarization(imgMaxPooled)
cv2.imwrite('imgBin.png', binImg)


cv2.imshow("test",binImg)
#-----------------------------





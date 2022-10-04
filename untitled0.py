from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = Image.open('car.png')
plt.axis("off")
plt.imshow(img)

gray = img.convert('L')
plt.figure()
plt.imshow(gray, cmap='gray')
plt.axis('off')

r, g, b = img.split()
np.array(img)
np.array(r)
np.array(g)
np.array(b)

k = np.array([
    [0,1,2],
    [2,2,0],
    [0,1,2]
])
k1 = np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])

k2 = np.array([
    [1,1,1],
    [0,0,0],
    [-1,-1,-1]
])

k2 = np.array([[-1,-1,-1],
              [-1,8,-1],
              [-1,-1,-1]])


data = np.array(r)
n,m = data.shape

def convolution(k, data):
    n,m = data.shape
    img_new = []
    for i in range(n-3):
        line = []
        for j in range(m-3):
            a = data[i:i+3,j:j+3]
            line.append(np.sum(np.multiply(k, a)))
        img_new.append(line)
    return np.array(img_new)


img_new = convolution(k2, data)#卷積過程

#卷積結果可視化
plt.imshow(img_new, cmap='gray')
plt.axis('off')
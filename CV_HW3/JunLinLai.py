import cv2
import numpy as np
import math

def zeroPadding(img):
    zeroImg = np.zeros((img.shape[0]+3,img.shape[1]+3))
    for i in range(1,img.shape[0]+1):
        for j in range(1, img.shape[1]+1):
            zeroImg[i][j] = img[i-1][j-1]
    return zeroImg

def convolution(kernel, data):
    n,m = data.shape

    convolution_img = []
    for i in range(0,n-3):
        line = []
        for j in range(0,m-3):
            a = data[i:i+3,j:j+3]
            temp = np.sum(np.multiply(kernel, a))
            line.append(temp)
        convolution_img.append(line)
    return np.array(convolution_img)


def gaussianBlur(img):

    x, y = np.mgrid[-1:2, -1:2]
    gaussian_kernel = np.exp(-(x ** 2 + y ** 2))

    # Normalization
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    # print(gaussian_kernel)
    colized = convolution(gaussian_kernel,img)

    return colized

def gradientCal(gx,gy):
    G = np.hypot(gx,gy)
    G = G / G.max() * 255
    theta = np.arctan2(gy,gx)
    return (G,theta)

def nonmaxSuppression(img,D):
    m,n = img.shape
    copyed = np.zeros((m, n), dtype=np.int32)
    angle = D * 180 / np.pi
    angle[ angle<0 ] += 180

    for i in range(1,m-1):
        for j in range(1,n-1):
            try:
                q = 255
                r = 255

                if(0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i,j+1]
                    r = img[i,j-1]
                elif(22.5 <= angle[i,j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    copyed[i, j] = img[i, j]
                else:
                    copyed[i, j] = 0

            except IndexError as e:
                pass

    return copyed

def threshold(img, lowthresholdRatio,highthresholdRatio):
    highRatio = img.max() * highthresholdRatio
    lowRatio = highRatio * lowthresholdRatio

    m,n = img.shape
    res = np.zeros((m,n),dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highRatio)

    weak_i, weak_j = np.where((img <= highRatio) & (img >= lowRatio))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res

def EdgeLinking(img, weak=25, strong=255):
    m, n = img.shape
    for i in range(1, m-1):
        for j in range(1, n-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or
                        (img[i+1, j] == strong) or
                        (img[i+1, j+1] == strong) or
                        (img[i, j-1] == strong) or
                        (img[i, j+1] == strong) or
                        (img[i-1, j-1] == strong) or
                        (img[i-1, j] == strong) or
                        (img[i-1, j+1] == strong)):

                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def CannyEdgeDetect(img):
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    img_gx = convolution(Gx,img)
    img_gy = convolution(Gy,img)

    mag = np.hypot(img_gx,img_gy)
    mag *=255.0 / np.max(mag)

    gradMat , theta = gradientCal(img_gx,img_gy)

    non = nonmaxSuppression(mag,theta)

# 1
#     thresholdImg = threshold(non,lowthresholdRatio=0.09,highthresholdRatio=0.5)
# 2
#     thresholdImg = threshold(non,lowthresholdRatio=0.03,highthresholdRatio=0.8)
# 3
#     thresholdImg = threshold(non,lowthresholdRatio=0.01,highthresholdRatio=0.1)
# 4
    thresholdImg = threshold(non,lowthresholdRatio=0.01,highthresholdRatio=0.8)

    imgEdgeLinking = EdgeLinking(thresholdImg)

    cv2.imwrite('result_img2.jpg', imgEdgeLinking)

    return imgEdgeLinking

def hough_line(img, angle_step=1, value_threshold=120):

    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    m, n = img.shape
    diagLen = int(round(math.sqrt(m * m + n * n)))
    rhos = np.linspace(-diagLen, diagLen, diagLen * 2)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((2 * diagLen, num_thetas), dtype=np.uint8)

    edges = img > value_threshold if 1 else img < value_threshold

    y_idxs, x_idxs = np.nonzero(edges)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = diagLen + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    m = accumulator.shape[0]
    n = accumulator.shape[1]
    accumulator = accumulator.reshape((m * n))

    def peak_vote(acc,thetas,rhos):
        idx = np.argmax(acc)
        rho = rhos[int(idx / n)]
        theta = thetas[idx % n]

        return idx,theta,rho

    for i in range(5):
        idx , theta , rho = peak_vote(accumulator,thetas,rhos)

        y1 = int((rho) / math.sin(theta))
        x2 = img.shape[1]
        y2 = int((rho - (x2 * math.cos(theta))) / math.sin(theta))

        point1 , point2 = (0,y1) , (x2,y2)

        cv2.line(img2,point1,point2, (255, 0, 0), 1, cv2.LINE_AA)

        np.delete(accumulator, [0])
        accumulator = np.delete(accumulator, np.where(accumulator == accumulator[idx]))

    cv2.imwrite('result_img3.jpg', img2)

src = "./test_images/1.jpg"

img = cv2.imread(src)
img2 = img.copy()

img_gray = cv2.imread(src , cv2.IMREAD_GRAYSCALE)
img_padding_grad = zeroPadding(img_gray)
img_col = gaussianBlur(img_padding_grad)

cv2.imwrite('result_img1.jpg', img_col)

img_cos = cv2.imread('./result_img1.jpg' , cv2.IMREAD_GRAYSCALE)

img_Canny = CannyEdgeDetect(img_cos)

hough_line(img_Canny)

cv2.imshow("Original",img)
cv2.imshow("Processed", img2)
cv2.waitKey()
cv2.destroyAllWindows()
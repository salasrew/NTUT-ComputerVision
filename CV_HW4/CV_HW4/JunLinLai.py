import cv2
from matplotlib import pyplot as plt
from itertools import cycle
import numpy as np
from scipy import ndimage as ndi


class _fcycle(object):
    def __init__(self, iterable):
        """Call functions from the iterable each time it is called."""
        self.nextStep = cycle(iterable)

    def __call__(self, *args, **kwargs):
        f = next(self.nextStep)
        return f(*args, **kwargs)


P = [np.eye(3),
     np.array([[0, 1, 0]] * 3, dtype=object),
     np.flipud(np.eye(3)),
     np.rot90([[0, 1, 0]] * 3)]


# 初始輪廓配置
def initialContour(img, a, b, c, d):
    init_ct = np.zeros(img.shape, dtype=np.int8)
    init_ct[a:b, c:d] = 1
    return init_ct


#  侵蝕 erode
def erode(u):
    erosions = []
    for P_i in P:
        erosions.append(ndi.binary_erosion(u, P_i))

    return np.array(erosions, dtype=np.int8).max(0)


#  膨脹 dilate
def dilate(u):
    dilations = []
    for P_i in P:
        dilations.append(ndi.binary_dilation(u, P_i))

    return np.array(dilations, dtype=np.int8).min(0)


_curvop = _fcycle([lambda u: erode(dilate(u)),
                   lambda u: dilate(erode(u))])


def _initLevelSet(initLevelSet, image_shape):
    if isinstance(initLevelSet, str):
        res = circleLevelSet(image_shape)
    else:
        res = initLevelSet
    return res


def circleLevelSet(image_shape, center=None, radius=None):
    if center is None:
        center = tuple(i // 2 for i in image_shape)

    if radius is None:
        radius = min(image_shape) * 3.0 / 8.0

    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum((grid) ** 2, 0))
    res = np.int8(phi > 0)
    return res


def inverseGaussianGradient(image, alpha=100.0, sigma=5.0):
    gradnorm = ndi.gaussian_gradient_magnitude(image, sigma, mode='nearest')
    return 1.0 / np.sqrt(1.0 + alpha * gradnorm)


def activeContour(image, iterations, initLevelSet='circle', smoothing=1, threshold=0.1, balloon=0,
                  iter_callback=lambda x: None):
    initLevelSet = _initLevelSet(initLevelSet, image.shape)

    structure = np.ones((3,) * len(image.shape), dtype=np.int8)
    dimage = np.gradient(image)
    # threshold_mask = image > threshold
    if balloon != 0:
        threshold_mask_balloon = image > threshold / np.abs(balloon)

    u = np.int8(initLevelSet > 0)
    iter_callback(u)

    for _ in range(iterations):
        # print("itr: = " + str(_) + "!")
        # Balloon
        if balloon > 0:
            aux = ndi.binary_dilation(u, structure)
        elif balloon < 0:
            aux = ndi.binary_erosion(u, structure)
        if balloon != 0:
            u[threshold_mask_balloon] = aux[threshold_mask_balloon]

        # Image attachment
        aux = np.zeros_like(image)
        du = np.gradient(u)
        for el1, el2 in zip(dimage, du):
            aux += el1 * el2
        u[aux > 0] = 1
        u[aux < 0] = 0

        # Smoothing
        for _ in range(smoothing):
            u = _curvop(u)
        iter_callback(u)
    return u


def visualCallback(background, fig=None):
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()

    # 清除當前圖形
    fig.clf()
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.imshow(background, cmap=plt.cm.gray)

    plt.pause(0.001)

    def callback(levelset):
        if ax0.collections:
            del ax0.collections[0]
        ax0.contour(levelset, [0.5], colors='y')
        fig.canvas.draw()
        plt.pause(0.001)

    return callback


path = r".\test_images\pic1.jpg"
# path = r".\test_images\pic2.jpg"
# path = r".\test_images\pic3.jpg"

# Load the image.
imgcolor = cv2.imread(path)

# 梯度 = 膨胀 - 腐蚀
kernel = np.ones((8, 8), np.uint8)
cv2.dilate(imgcolor, kernel, iterations=5)  # 膨胀
cv2.erode(imgcolor, kernel, iterations=5)  # 腐蚀
gradient = cv2.morphologyEx(imgcolor, cv2.MORPH_GRADIENT, kernel)

gray = cv2.cvtColor(gradient, cv2.COLOR_RGB2GRAY)  # 轉灰

# pic1 10 255
# pic2 49 255
# pic3 50 255
ret, img = cv2.threshold(gray, 49, 255, cv2.THRESH_BINARY)  # 二值化
# g(I)
grayImg = inverseGaussianGradient(img, alpha=500, sigma=4.0)  # 反高斯梯度

callback = visualCallback(imgcolor)

# MorphGAC.
# pic1 itr 340
# pic2 itr 370
# pic3 itr 355
activeContour(grayImg, 400, initialContour(img, 10, -10, 10, -10), smoothing=5, balloon=-1, iter_callback=callback)

plt.show()
import logging
import cv2
import numpy as np
from imageio import imread
from matplotlib import pyplot as plt
import morphsnakes as ms    # 调用morphsnakes.py

def visual_callback_2d(background, fig=None):
 
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()

    # 清除當前圖形
    fig.clf()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    plt.pause(0.001)
 
    def callback(levelset):
        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='y')
        fig.canvas.draw()
        plt.pause(0.001)

    return callback
 
def example(PATH):
    logging.info('Running: example_coins (MorphGAC)...')
 
    # Load the image.
    imgcolor = imread(PATH)

    # 梯度 = 膨胀 - 腐蚀
    kernel = np.ones((8, 8), np.uint8)
    cv2.dilate(imgcolor, kernel, iterations=5)  # 膨胀
    cv2.erode(imgcolor, kernel, iterations=5)  # 腐蚀
    gradient = cv2.morphologyEx(imgcolor, cv2.MORPH_GRADIENT, kernel)
 
 
    gray = cv2.cvtColor(gradient, cv2.COLOR_RGB2GRAY)  # 转化为灰度图像

    # pic1 10 255
    # pic2 10 255
    # pic3 50 255
    ret, img = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)  # 二化值 根据明暗调节第二个值
    # g(I)
    gimg = ms.inverse_gaussian_gradient(img, alpha=500, sigma=4.0)   # 反高斯梯度
 
    # 初始轮廓设置
    init_ls = np.zeros(img.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1
 
    # Callback for visual plotting
    callback = visual_callback_2d(imgcolor)

    # MorphGAC.
    # pic1 itr 340
    # pic2 itr 370
    # pic3 itr 355
    ms.morphological_geodesic_active_contour(gimg, 400, init_ls,
                                             smoothing=5, threshold=0.1,
                                             balloon=-1, iter_callback=callback)

 
Path1 = r".\test_images\pic1.jpg"
Path2 = r".\test_images\pic2.jpg"
Path3 = r".\test_images\pic3.jpg"
# example(Path1)
# example(Path2)
example(Path3)

plt.show()
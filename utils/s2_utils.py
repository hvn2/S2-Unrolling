from utils.common_utils import *

def sentinel2RGB(Yin, thresholdRGB):
    '''
    Function create RGB image from Sentinel image
    Y - a sentinel image with B2,B3 and B4
    thresholdRGB =1, then the tail and head of RGB values are removed.
    This increases contrast
    '''
    [ydim, xdim, zdim] = np.shape(Yin)
    Y = np.transpose(np.reshape(Yin, (ydim * xdim, zdim)))
    Y = Y / np.max(Y)
    # T matrix from "Natural color representation of Sentinell-2 data " paper
    T = np.array([[0.180, 0.358, 0.412],
                  [0.072, 0.715, 0.213],
                  [0.950, 0.119, 0.019]])

    XYZ = np.matmul(T, Y)
    # % Convert to RGB
    M = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    sRGB = np.matmul(M, XYZ)
    # % Correct gamma
    gamma_map = (sRGB > 0.0031308)
    sRGB[gamma_map] = 1.055 * np.power(sRGB[gamma_map], (1. / 2.4)) - 0.055
    sRGB[~gamma_map] = 12.92 * sRGB[~gamma_map]
    sRGB[sRGB > 1] = 1;
    sRGB[sRGB < 0] = 0;
    if (thresholdRGB > 0):
        thres = 0.01;
        for idx in range(3):
            y = sRGB[idx, :]
            [a, b] = np.histogram(y, bins=100)
            a = np.cumsum(a) / np.sum(a)
            th = b[0]
            i = np.argwhere(a < thres)
            if len(i) > 0:
                th = b[i[-1]]
            y = np.maximum(0, y - th)
            [a, b] = np.histogram(y, bins=100)
            a = np.cumsum(a) / np.sum(a)
            i = np.argwhere(a > 1 - thres)
            th = b[i[0]]
            y[y > th] = th
            y = y / th
            sRGB[idx, :] = y

    RGB = np.zeros_like(Yin)
    RGB[:, :, 0] = np.reshape(sRGB[0, :], (ydim, xdim))
    RGB[:, :, 1] = np.reshape(sRGB[1, :], (ydim, xdim))
    RGB[:, :, 2] = np.reshape(sRGB[2, :], (ydim, xdim))
    return RGB

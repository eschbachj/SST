from skimage.color import rgb2hed, hed2rgb
import numpy as np
import cv2

blur_threshold = 80

def check_blur(im):
    var = cv2.Laplacian(im, cv2.CV_64F).var()
    blur = var>blur_threshold
    #returns true if not blurry, false if blurry
    return blur


def deconv(im):
    ihc_hed = rgb2hed(im)
    null = np.zeros_like(ihc_hed[:, :, 0])
        
    sst = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    cells = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    return sst,cells
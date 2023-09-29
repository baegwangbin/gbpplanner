import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

''' Input: img of square dimensions.
    Output: Blurred image for use as a distance field 
    Obstacles should be BLACK, background is WHITE '''
def create_distance_field(img):
    if (img.shape[0]!=1000 or img.shape[1]!=1000):
        print("Resizing to 1000 x 1000")
        img = cv2.resize(img, (1000, 1000))

    # Convolve with a Gaussian to effect a blur.
    blur = gaussian_filter(img, sigma=5)
    blur = cv2.normalize(blur, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    distance_field = blur
    return distance_field
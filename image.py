#################################################################################################################
# A Modularized implementation for
# Image enhancement, extracting descriptors, clustering, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_DESCRIPTORS - CLUSTERING - EVALUATION_VISUALIZATION
# Implementation of the IMAGE_ENHANCEMENTS class
# Author: Aya Saad
# Date created: 24 September 2019
#
#################################################################################################################

import cv2

class Image:
    def __init__(self, image_name):
        self.image_name = image_name


    def image_read(self, normalize=True, blur=True, resize=False):
        # load the image, convert it to grayscale, and detect edges
        self.img = cv2.imread(self.image_name)
        if normalize:
            self.img = self.image_normalize()
        if blur:
            self.img = self.image_blur()
        if resize:
            self.img = self.image_resize()
        return self.img

    def image_normalize(self, nmin= 0, nmax=255):
        print('Normalization...')
        return cv2.normalize(self.img, None, alpha=nmin, beta=nmax, norm_type=cv2.NORM_MINMAX)

    def image_blur(self, g_size=3):
        print('Median Blurring...')
        return cv2.medianBlur(self.img, g_size)

    def image_resize(self, width=64, height=64):
        print('Resize...')
        return cv2.resize(self.img, (width, height), interpolation=cv2.INTER_CUBIC)
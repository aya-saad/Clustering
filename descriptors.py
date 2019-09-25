#################################################################################################################
# A Modularized implementation for
# Image enhancement, extracting descriptors, clustering, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_DESCRIPTORS - CLUSTERING - EVALUATION_VISUALIZATION
# Implementation of the IMAGE_DESCRIPTOR class
# Author: Aya Saad
# Date created: 24 September 2019
#
#################################################################################################################

import cv2
from skimage import feature

class Descriptor:

    def __init__(self, name, image):
        '''
        Class Descriptor constructor
        :param name:    Name of the Algorithm/model
        :param image:   Name of the image
        '''
        self.name = name
        self.image = image

    def algorithm(self, alg_name):
        '''
        Choose the feature extractor algorithm
        :param alg_name: Name of the algorithm
        '''
        if alg_name == 'HOG':
            return self._alg_hog()
        if alg_name == 'ORB':
            return self._alg_orb()
        elif alg_name == 'SIFT':
            return self._alg_sift()
        elif alg_name == 'SURF':
            return self._alg_surf()
        elif alg_name == 'FAST':
            return self._alg_fast()
        elif alg_name == 'BRIEF':
            return self._alg_brief()

    def _alg_hog(self):
        print('Extracting HOG decriptors for image ...')
        # extract Histogram of Oriented Gradients from the logo
        print(self.image.shape)
        fd, hog = feature.hog(self.image, orientations=9, pixels_per_cell=(3, 3),
                                          cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
                                          visualize=True)
        return self.image, fd, hog

    def _alg_orb(self):
        print('Extracting ORB decriptors for image ...')

        # orb = cv.ORB_create(WTA_K=cv.NORM_HAMMING2)  # default nfeatures=500
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp, desc = orb.detectAndCompute(self.image, None)

        img = cv2.drawKeypoints(self.image, kp, None, (0, 0, 255), 4)

        if desc is None:
            print('None Object descriptor!!')
        else:
            print(len(desc), '  ', len(kp))
            # for d in desc:
            #  print(d)
        return img, kp, desc

    def _alg_sift(self):
        print('Extracting SIFT decriptors for image ...')
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp, desc = sift.detectAndCompute(self.image, None)

        img = cv2.drawKeypoints(self.image, kp, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        if desc is None:
            print('None Object descriptor!!')
        else:
            print(len(desc), '  ', len(kp))
            # for d in desc:
            #  print(d)
        return img, kp, desc

    def _alg_surf(self):
        print('Extracting SURF decriptors for image ...')
        surf = cv2.xfeatures2d.SURF_create()
        # find the keypoints and descriptors with SURF
        kp, desc = surf.detectAndCompute(self.image, None)

        img = cv2.drawKeypoints(self.image, kp, None, (0, 0, 255), 4)

        if desc is None:
            print('None Object descriptor!!')
        else:
            print(len(desc), '  ', len(kp))
            # for d in desc:
            #  print(d)
        return img, kp, desc

    def _alg_fast(self):
        print('Extracting FAST decriptors for image ...')
        fast = cv2.FastFeatureDetector_create()

        #orb = cv.ORB_create(nfeatures=100000, scoreType=cv.ORB_FAST_SCORE)
        #orb = cv.ORB_create(edgeThreshold=5, patchSize=31, nlevels=8, fastThreshold=10, scaleFactor=1.2, WTA_K=2,
        #                     scoreType=cv.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=150000000) # cv.ORB_HARRIS_SCORE
        #orb = cv.ORB_create(edgeThreshold=2, patchSize=16, nlevels=8, fastThreshold=fast.getThreshold(), scaleFactor=1.2, WTA_K=2,
        #                    scoreType=cv.ORB_FAST_SCORE, firstLevel=0, nfeatures=15000) # cv.ORB_HARRIS_SCORE
        orb = cv2.ORB_create()
        # find and draw the keypoints
        kp = fast.detect(self.image, None)
        # Print all default params
        print("Threshold: {}".format(fast.getThreshold()))
        print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
        print("neighborhood: {}".format(fast.getType()))
        print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
        kp2 , desc = orb.detectAndCompute(self.image, cv2.ORB_FAST_SCORE)
        img = cv2.drawKeypoints(self.image, kp, None, (0, 0, 255), flags=cv2.DrawMatchesFlags_DEFAULT)   # 4

        if kp is None:
          print('None Object descriptor!!')
        else:
          print(len(kp))
          # for d in desc:
          #  print(d)
        return img, kp, desc

    def _alg_brief(self, img):
        print('Extracting BRIEF decriptors for image ...')

        # Initiate FAST detector
        #star = cv.xfeatures2d.StarDetector_create()
        fast = cv2.FastFeatureDetector_create()
        # Initiate BRIEF extractor
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        # find the keypoints with STAR
        #kp = star.detect(img, None)
        kp = fast.detect(img, None)
        # compute the descriptors with BRIEF
        kp, desc = brief.compute(img, kp)

        # find the keypoints with STAR
        img = cv2.drawKeypoints(img, kp, None, (0, 0, 255), 4)

        if desc is None:
          print('None Object descriptor!!')
        else:
          print(len(desc),'  ', len(kp))
          #for d in desc:
          #  print(d)
        return img, kp, desc

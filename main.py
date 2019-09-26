#################################################################################################################
# A Modularized implementation for
# Image enhancement, extracting descriptors, clustering, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_DESCRIPTORS - CLUSTERING - EVALUATION_VISUALIZATION
# Author: Aya Saad
# Date created: 24 September 2019
# Project: AILARON
# funded by RCN FRINATEK IKTPLUSS program (project number 262701) and supported by NTNU AMOS
#
#################################################################################################################
import numpy as np
from utils import *
from config import get_config
from dataset import Dataset
from descriptors import Descriptor
from skimage import exposure

import cv2
from image import Image
from clusterAlg import KNearestClustering, KMeansClustering

def main(config):
    np.random.seed(config.random_seed)
    prepare_dirs(config)
    header_file = config.data_dir + '/header.tfl.txt'
    filename = 'image_set.dat'
    dataset = Dataset(config.data_dir, header_file, filename)

    # Dataset retrieval
    cl = dataset.get_classes()
    train_x, train_y, test_x, test_y = dataset.split_data()

    print('output_dir: ', config.output_dir)
    print('data_dir: ', config.data_dir)
    print('classes:', cl)
    print('train_x.shape: ', train_x.shape)
    print('train_y.shape: ', train_y.shape)
    print('test_x.shape: ', test_x.shape)
    print('test_y.shape: ', test_y.shape)

    # Get image descriptor
    data = []
    for x in train_x:
        print(x)
        image = Image(x).image_read(resize=True)
        _, fd, _ = Descriptor('HOG', image).algorithm('HOG')
        data.append(fd)

    # "train" the nearest neighbors classifier
    print("[INFO] training classifier...")
    classifier = KNearestClustering()
    #classifier = KMeansClustering()
    classifier.build_model()
    classifier.train(train_x=data, train_y=train_y)

    print("[INFO] evaluating...")
    X_test = []
    for (i,x) in enumerate(test_x):
        print(x)
        image = Image(x).image_read(resize=True)
        image, fd, hogImage = Descriptor('HOG', image).algorithm('HOG')
        X_test.append(fd.reshape(1,-1))
        #pred = model.predict(fd.reshape(1,-1))[0]
        #print('i, Predicted: ',i, pred, x, cl[pred-1], cl[test_y.iloc[i]-1])

        # visualize the HOG image
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
        cv2.imshow("HOG Image #{}".format(i + 1), hogImage)

    predict = classifier.predict(test_x=X_test)
    print(test_x)
    y_test = []
    y_pred = []
    for (i,pred) in enumerate(predict):
        print('i, Predicted: ', i, pred[0], cl[pred[0] - 1], cl[test_y.iloc[i] - 1], test_x.iloc[i])
        y_test.append(test_y.iloc[i])
        y_pred.append(pred[0])
    print(y_test)
    print(y_pred)

    classifier.performance(y_test,y_pred)


    return

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
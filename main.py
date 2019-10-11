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
from clusterAlg import *

def main(config):
    np.random.seed(config.random_seed)
    prepare_dirs(config)
    header_file = config.data_dir + '/header.tfl.txt'
    filename = 'image_set.dat'
    dataset = Dataset(config.data_dir, header_file, filename)
    input_data = dataset.get_data()

    # Dataset retrieval
    cl = dataset.get_classes()

    # Get image descriptor
    data = []
    temp = []
    for x in input_data.iloc[:, 0]:
        print(x)
        image = Image(x).image_read(resize=True)
        img = np.float64(image) / np.max(image)
        img = img.astype('float32')
        temp.append(img)
        _, _, fd = Descriptor('HOG', image).algorithm('HOG')
        data.append(fd)

    X = np.stack(temp)
    X /= 255.0
    x_size = len(input_data)
    X = X.reshape(x_size, -1).astype('float32')

    X_hog = np.stack(data)
    X_hog = X_hog.reshape(x_size, -1).astype('float32')

    print(X.shape)


    # Hierarchical Clustering
    labels = HierarchicalClustering().draw_dendogram(X)
    #labels = MeanShiftAlgo().meanshift_fit(X)
    PrincipleComponentAnalysis().pca_fit(labels,X)
    TSNEAlgo().tsne_fit(X, input_data, labels)


    labels_hog = HierarchicalClustering().draw_dendogram(X_hog)
    #labels = MeanShiftAlgo().meanshift_fit(X_hog)
    PrincipleComponentAnalysis().pca_fit(labels_hog, X_hog)
    TSNEAlgo().tsne_fit(X_hog, input_data, labels_hog)





    return

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
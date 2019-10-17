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
from descriptors import Descriptor, BFMatcher
from skimage import exposure

import cv2
from image import Image
from clusterAlg import *

import scipy.spatial.distance as ssd
def calculate_distances(input_data, desc ='SIFT'):
    desc_len = 64       # SURF descriptor length
    if desc == 'SIFT':
        desc_len = 128
    X_desc = []
    temp = []
    for x in input_data.iloc[:, 0]:
        print(x)
        image = Image(x).image_read(resize=False)
        img = np.float64(image) / np.max(image)
        img = img.astype('float32')
        temp.append(img)
        _, _, fd = Descriptor(desc, image).algorithm(desc)
        X_desc = np.append(X_desc, None)  # Allocate space for your new array
        X_desc[-1] = fd

    dist = []
    print('len(X_desc)',len(X_desc))
    for i in range(0, len(X_desc)):
        for j in range(i + 1, len(X_desc)):
            sim = -1
            if ((X_desc[i] is not None) and (X_desc[j] is not None)):
                sim = BFMatcher().match_images(desc_a=X_desc[i], desc_b=X_desc[j])
            dist.append(sim)

    for i in range(0, len(X_desc)):
        if X_desc[i] is None:
            X_desc[i] = np.zeros((2, 3), dtype=float)
    print('max(X) = ', max(dist))
    max_value = max(dist)
    dist_desc = [x if x >= 0 else max_value for x in dist]

    ####################################################
    X_I = np.zeros([len(X_desc), len(max(X_desc, key=lambda x: len(x))), desc_len])

    for i, j in enumerate(X_desc):
        for k, l in enumerate(j):
            X_I[i][k][0:len(l)] = l
        print(i, len(j), X_I[i][0:len(j)], j, end='  ')

    x_size = len(input_data)
    X_I = X_I.reshape(x_size, -1).astype('float32')
    print('X_I.shape  ', X_I.shape)

    ####################################################

    return X_I, dist_desc

def calculate_hog_distances(input_data, desc ='HOG'):
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
    return X, X_hog


def main(config):
    np.random.seed(config.random_seed)
    prepare_dirs(config)
    header_file = config.data_dir + '/header.tfl.txt'
    filename = 'image_set.dat'
    dataset = Dataset(config.data_dir, header_file, filename)
    input_data = dataset.get_data()

    # Get image descriptor
    X_I, X_SIFT = calculate_distances(input_data, desc='SIFT')
    X_U, X_SURF = calculate_distances(input_data, desc='SURF')
    X, X_hog = calculate_hog_distances(input_data, desc='HOG')
    print('------------------------------------------')

    # Hierarchical Clustering
    labels = HierarchicalClustering()\
        .draw_dendogram(X,
                        title='Hierarchical Clustering Dendrogram')
    #labels = MeanShiftAlgo().meanshift_fit(X)
    #PrincipleComponentAnalysis().pca_fit(labels,X)
    TSNEAlgo().tsne_fit(X, input_data, labels)

    labels_hog = \
        HierarchicalClustering()\
            .draw_dendogram(X_hog,
                            title='Hierarchical Clustering Dendrogram of the HOG Descriptors')
    #labels = MeanShiftAlgo().meanshift_fit(X_hog)
    #PrincipleComponentAnalysis().pca_fit(labels_hog, X_hog)
    TSNEAlgo().tsne_fit(X_hog, input_data, labels_hog, title='HOG Descriptors TSNE Representation')
    TSNEAlgo().tsne_fit(X, input_data, labels_hog, title='HOG Labeled TSNE Representation')

    labels_sift = \
        HierarchicalClustering() \
                .draw_dendogram(X_SIFT,
                            title='Hierarchical Clustering Dendrogram of the SIFT Descriptors')
    TSNEAlgo().tsne_fit(X_I, input_data, labels_sift, title='SIFT Descriptors TSNE Representation')
    TSNEAlgo().tsne_fit(X, input_data, labels_sift, title='SIFT Labeled TSNE Representation')


    labels_surf = \
        HierarchicalClustering() \
            .draw_dendogram(X_SURF,
                            title='Hierarchical Clustering Dendrogram of the SURF Descriptors')
    TSNEAlgo().tsne_fit(X_U, input_data, labels_surf, title='SURF Descriptors TSNE Representation')
    TSNEAlgo().tsne_fit(X, input_data, labels_surf, title='SURF Labeled TSNE Representation')



    return

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
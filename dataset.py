#################################################################################################################
# A Modularized implementation for
# Image enhancement, extracting descriptors, clustering, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_DESCRIPTORS - CLUSTERING - EVALUATION_VISUALIZATION
# Implementation of the LOAD_DATA class
# Author: Aya Saad
# Date created: 24 September 2019
#
#################################################################################################################

import os
import csv
import pandas as pd

class Dataset:
    def __init__(self, data_dir, header_file, filename):
        '''
        Dataset constructor
        :param data_dir:    name of the data directory
        :param header_file: name of the header file
        :param filename:    name of the dataset file
        '''
        self.header_file = header_file
        self.data_dir = data_dir
        self.filename = filename
        self.input_data = self.get_data()

    def get_classes(self):
        '''
        Get the list of classes from the header file
        return: clst  the class list
        '''
        cl_file = self.header_file
        with open(cl_file) as f:
            reader = csv.reader(f)
            cl = [r for r in reader]
        clst = cl[0]
        return clst

    def get_data(self):
        '''
        Read the data file and get the list of images along with their labels
        :return the data set
        '''
        input_data = pd.read_csv(os.path.join(self.data_dir, self.filename), header=None, delimiter=' ')
        print(input_data.head())
        return input_data

    def split_data(self):
        '''
        Split the dataset into training and testing
        :return train_x, train_y, test_x, test_y
        '''
        x = self.input_data.iloc[:, 0]
        y = self.input_data.iloc[:, 1]
        split_size = int(y.shape[0] * 0.95)
        print('split_size ', split_size)
        train_x, test_x = x[:split_size], x[split_size:]
        train_y, test_y = y[:split_size], y[split_size:]

        return train_x, train_y, test_x, test_y

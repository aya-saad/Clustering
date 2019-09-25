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
        self.header_file = header_file
        self.data_dir = data_dir
        self.filename = filename
        self.input_data = self.get_data()

    def get_classes(self):
        cl_file = self.header_file
        with open(cl_file) as f:
            reader = csv.reader(f)
            cl = [r for r in reader]
        clst = cl[0]
        return clst

    def get_data(self):
        input_data = pd.read_csv(os.path.join(self.data_dir, self.filename), header=None, delimiter=' ')
        print(input_data.head())
        return input_data

    def split_data(self):
        x = self.input_data.iloc[:, 0]
        y = self.input_data.iloc[:, 1]
        split_size = int(y.shape[0] * 0.95)
        print('split_size ', split_size)
        train_x, test_x = x[:split_size], x[split_size:]
        train_y, test_y = y[:split_size], y[split_size:]

        return train_x, train_y, test_x, test_y

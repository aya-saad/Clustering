import os
import csv
import pandas as pd


def get_classes(header_file):
    cl_file = header_file
    with open(cl_file) as f:
        reader = csv.reader(f)
        cl = [r for r in reader]
    clst = cl[0]
    return clst

def get_data(data_dir, filename):
    input_data = pd.read_csv(os.path.join(data_dir, filename), header=None, delimiter=' ')
    print(input_data.head())
    return input_data

def split_data(input_data):
    x = input_data.iloc[:, 0]
    y = input_data.iloc[:, 1]
    split_size = int(y.shape[0] * 0.95)
    print('split_size ', split_size)
    train_x, test_x = x[:split_size], x[split_size:]
    train_y, test_y = y[:split_size], y[split_size:]

    return train_x, train_y, test_x, test_y

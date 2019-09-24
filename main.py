import numpy as np
from utils import *
from config import get_config
from dataset import get_classes, get_data, split_data
from descriptors import get_hog
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
import cv2

def main(config):
    np.random.seed(config.random_seed)
    prepare_dirs(config)

    # Dataset retrieval
    cl = get_classes(config.data_dir + '/header.tfl.txt')

    filename = 'image_set.dat'
    input_data = get_data(config.data_dir, filename)
    train_x, train_y, test_x, test_y = split_data(input_data)

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
        _, fd, _ = get_hog(x)
        data.append(fd)

    # "train" the nearest neighbors classifier
    print("[INFO] training classifier...")
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(data, train_y)
    print("[INFO] evaluating...")
    for (i,x) in enumerate(test_x):
        print(x)
        image, fd, hogImage = get_hog(x)
        pred = model.predict(fd.reshape(1,-1))[0]
        print('i, Predicted: ',i, pred, x, cl[pred-1], cl[test_y.iloc[i]-1])

        # visualize the HOG image
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
        cv2.imshow("HOG Image #{}".format(i + 1), hogImage)

    return

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
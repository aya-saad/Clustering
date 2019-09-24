import cv2
from skimage import feature

def get_hog(imagePath, normalize=True, blur=True):
    # load the image, convert it to grayscale, and detect edges
    print(imagePath)
    img = cv2.imread(imagePath)
    ## IMAGE NORMALIZATION
    nmin = 0
    nmax = 255
    img_norm = cv2.normalize(img, None, alpha=nmin, beta=nmax, norm_type=cv2.NORM_MINMAX)
    img_blur = cv2.medianBlur(img_norm, 3)
    img_blur = cv2.resize(img_blur, (64, 64), interpolation=cv2.INTER_CUBIC)

    # extract Histogram of Oriented Gradients from the logo
    fd, hog = feature.hog(img_blur, orientations=12, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
                                      visualize=True)

    return img,fd, hog
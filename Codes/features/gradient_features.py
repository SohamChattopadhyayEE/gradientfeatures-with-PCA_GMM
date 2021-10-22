from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2 as cv


def hog_features(image_path, h = 64, w = 64, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    img = imread(image_path)
    
    resized_img = resize(img, (h*4, w*4))
    
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
    return fd, hog_image
    
def sift_features(image_path): 
    #load image
    image = cv.imread(image_path)

    #convert to grayscale image
    gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #initialize SIFT object
    sift = cv.xfeatures2d.SIFT_create()

    #detect keypoints
    keypoints, _= sift.detectAndCompute(image, None)

    #draw keypoints
    sift_image = cv.drawKeypoints(gray_scale, keypoints, None)

    return sift_image

def choose_features(image_path, feature_type = 'hog'):
    if feature_type == 'hog' :
        _, hog_image = hog_features(image_path)
        return hog_image
    elif feature_type == 'sift':
        sift_image =  sift_features(image_path)
        return sift_image
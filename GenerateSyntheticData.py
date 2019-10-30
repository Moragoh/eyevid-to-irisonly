#import DetectEyeFromPhoto
#import DetectFunctionFromHeightMap
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy import misc

def run_transform(f,image,heightmap):
    newimage,newheight = f(image,heightmap)
    return(newimage,newheight)



def saturate(image, array):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = s*2
    final_hsv = cv2.merge((h, s, v))
    saturated_image = cv2.cvtColor(final_csv, cv2.COLOR_HSV2BGR)

    return(saturated_image, array)

"""
def noise_generator (noise_type, image):
    row,col,ch= image.shape
    if noise_type == "gauss":
        mean = 0.0
        var = 0.01
        sigma = var**0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return (noisy, gauss, image)
"""
img = cv2.imread('testImage.jpg');
#img = img.astype(np.uint8)

"""
def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 10
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      #gauss = gauss.astype(np.uint8)
      noisy = image + gauss

      info = np.iinfo(data.dtype) # Get the information of the incoming image type
      data = data.astype(np.float64) / info.max # normalize the data to 0 - 1
      data = 255 * data # Now scale by 255
      noisy = data.astype(np.uint8)
      return noisy

#cv2.imshow("img", img)
#cv2.imshow("gaussian", gaussian)
noised = noisy('gauss', img)
cv2.waitKey(0)
cv2.imshow("noisy", noised)
cv2.imshow("origina", img)
cv2.waitKey(0)
"""

def rotate_image(image, array):
    image_array = []
    editied_arrays = []

    # Rotate the image
    image_array.append(image)
    rotated_90 = cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
    image_array.append(rotated_90)
    rotated_180 = cv::rotate(image, image, cv::ROTATE_180);
    iamge_array.append(rotated_180)
    rotated_270 = cv::rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
    image_array.append(rotated_180)

    # Rotate the arrays
    edited_arrays.append(rotate(matr_, 90))
    edited_arrays.append(rotate(matr_, 180))
    edited_arrays.append(rotate(matr_, -90))

def noise_image(image, array):
    img = cv2.imread(image);
    image_open = Image.open(image)
    w, h = image_open.size
    w, h = w - 1, h - 1
    min = -10
    max = 10
    noise = randint(min, max)
    i = 0

    while i < 10000:
        x_value = randint(0, w)
        y_value = randint(0, h)
        #print(img[x_value, y_value])

        pixel = img[x_value, y_value]

        pixel[0] = pixel[0] + noise
        pixel[1] = pixel[1] + noise
        pixel[2] = pixel[2] + noise

        img[x_value, y_value] = pixel
        i = i + 1

    




# Source code: https://artemrudenko.wordpress.com/2014/08/28/python-rotate-2d-arraymatrix-90-degrees-one-liner/
def rotate(matrix, degree):
    if abs(degree) not in [0, 90, 180, 270, 360]:
        # raise error or just return nothing or original
    if degree == 0:
        return matrix
    elif degree > 0:
        return rotate(zip(*matrix[::-1]), degree-90)
    else:
        return rotate(zip(*matrix)[::-1], degree+90)

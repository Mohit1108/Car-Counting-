# This code is given by M Tech-G
# See youtube Video for How to run this program!
# A small Project Done By M Tech-G
# Subscribe My Channel for more Videos !
# What We Are Build?
# We are Building our Own Counter Program in just 10 Lines of Code using Python!
# You have to install these libraries !
# using { pip install <lib name> }
# opencv-python
# cvlib
# matplotlib
# tenserflow
# keras
# if you are not install these libraries then it will not work properly!
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
detector = cv2.imread('car1.jpg')
bbox,label,conf = cv.detect_common_objects(detector)
output_image = draw_bbox(detector,bbox,label,conf)
plt.imshow(output_image)
plt.show()
print('Number of cars in this image is ' + str(label.count('car')))


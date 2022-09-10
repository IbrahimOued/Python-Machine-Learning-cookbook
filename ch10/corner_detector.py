# 1 Let's make the basic imports
import sys
import cv2
import numpy as np

# 2 Load the input image. We'll use the box.png image
# Load the input image 'box.png'
input_file='ch10/box.png'
img=cv2.imread(input_file)
cv2.imshow('Input image', img)

# 3 Convert the image into grayscale and cast it to floating-point
# values. We need the floating-points values for the corner detector to work
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray=np.float32(img_gray)

# 4 Run the HARRIS CORNER DETECTOR function on the grayscale image
# Harris corner detector
img_harris=cv2.cornerHarris(img_gray, 7, 5, .04)

# 5 To mark the corners, we need to dilate the image, as follows
# Resultant image is dilated to mark the corners
img_harris = cv2.dilate(img_harris, None)

# 6 Let's threshold the image to display the importants points
# Threshold the image
img[img_harris > .01 * img_harris.max()] = [0, 0, 0]

# 7 Display the output image
cv2.imshow('Harris corners', img)
cv2.waitKey()
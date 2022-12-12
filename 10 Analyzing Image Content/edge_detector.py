# 1 Let's make the basics imports
import sys
import cv2

# 2 Load the input image. We'll work with the chair.jpg image
# load the input file
# and convert it to grayscale(In fact we'll load it)
# with the grayscale parameter of openCV)
# input_file = sys.argv[1] 
input_file = 'ch10/siracusa.jpg'
img=cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

# 3 Extract the height and width of the image
h, w = img.shape

# 4 The SOBEL FILTER is a type of edge detector that uses a 3 x 3
# kernel to detect horizontal and vertical edges separately
sobel_horizontal=cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

# 5 Run the vertical Sobel detector
sobel_vertical=cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

# 6 The LAPLACIAN EDGE DETECTOR detects edges in both directions. We use
# it as follows
laplacian=cv2.Laplacian(img, cv2.CV_64F)

# 7 Even though Laplacian addresses the shortcomings of Sobel,
# the output is still very noisy.
# The Canny edge detector outperforms all of them because of
# the way it treats the problem. It is a multistage process,
# and it uses hysteresis to come up with clean edges:
canny = cv2.Canny(img, 50, 240)

# 8 Display all the output images
cv2.imshow('Original', img) 
cv2.imshow('Sobel horizontal', sobel_horizontal)
cv2.imshow('Sobel vertical', sobel_vertical) 
cv2.imshow('Laplacian', laplacian) 
cv2.imshow('Canny', canny) 
cv2.waitKey() 

# 9 We will run the code in the terminal window using the command
# python edge_detector.py siracusa.jpg

# At the top of the screenshot is the original image, the horizontal
# Sobel edge detector output, and the vertical Sobel edge detector
# output. Note how the detected lines tend to be vertical. This is
# due to the fact that it's a horizontal edge detector, and it tends
# to detect changes in this direction. At the bottom of the
# screenshot is the Laplacian edge detector 
# output and the Canny edge detector, which detects all the edges nicely.
# 1 Make the basic imports
import  sys
import cv2

# 2 Load the input image. We'll use the sunrise.jpg image
# Load input image
input_file='ch10/gubbio.jpg'
img=cv2.imread(input_file)

# 3 Convert the image into grayscale and display it
# Convert it to grayscale
img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Input grayscale', img_gray)

# 4 Equalize the histogram of the grayscale image and display it
# Equalize the histogram
img_gray_histeq=cv2.equalizeHist(img_gray)
cv2.imshow('Histogram equalized', img_gray_histeq)

# 5 OpenCV loads images in the BGR format by default, so let's convert
# it from BGR into YUV first
# Histogram equalization of color images
img_yuv=cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# 6 Equalize the Y channel as follows
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# 7 convert back into BGR
img_histeq=cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# 8 Display the input and output images
cv2.imshow('Input color image', img) 
cv2.imshow('Histogram equalized - color', img_histeq) 
 
cv2.waitKey()

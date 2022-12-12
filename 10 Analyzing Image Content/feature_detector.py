# 1 Make the basic imports
import cv2
import  numpy as np

# 2 Load the input image. We'll use table.jpg
# load input image --table.jpg
input_file='ch10/flowers.jpg'
img = cv2.imread(input_file)

# 3 Convert this image into grayscale
img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4 Initialize the SIFT detector object and extract the keypoints
sift = cv2.xfeatures2d.SIFT_create()
keypoints=sift.detect(img_gray, None)

# 5 The keypoints are the salient points, but they are not features. This
# basically gives us the location of the salient points. SIFT also functions as a
# very effective feature extractor

# 6 Draw the keypoints on top of the input image, as follows
img_sift=np.copy(img)
cv2.drawKeypoints(img, keypoints, img_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 7 Display the input and output images
cv2.imshow('Input image', img) 
cv2.imshow('SIFT features', img_sift) 
cv2.waitKey() 
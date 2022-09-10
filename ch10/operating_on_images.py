# 1 Make the basic imports
import sys
import cv2

def display(winname, image):
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 2 Specify the input image as the first argument to the file,
# and read it using the image read function.
# We will use the forest.jpg file that is provided to you, as follows:
input_file = 'ch10/forest.jpg'
img = cv2.imread(input_file)


# 3 Display the image
# Load and display and image
# Load and display an image -- 'forest.jpg'
cv2.imshow('Original', img)

# 4 Cropping and image
h, w = img.shape[:2]
start_row, end_row = int(.21*h), int(.73*h)
start_col, end_col = int(.37*w), int(.92*w)

# 5 Crop the image using NumPy style slicing and display it
img_cropped = img[start_row:end_row, start_col:end_col]
# cv2.imshow('Cropped', img_cropped)
display('Cropped', img_cropped)

# 6 Resize the image using to 1.3 times its original size and display it
# Resizing an image
scaling_factor = 1.3
img_scaled = cv2.resize(img, None, fx=scaling_factor,
                        fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
# cv2.imshow('Uniform resizing', img_scaled)
display('Uniform resizing', img_scaled)

# 7 The previous method will uniformly scale the image on both dimensions. Let's
# assume that we want to skew the image based on specific output dimensions. We
# will use the following code
img_scaled = cv2.resize(img, (250, 400), interpolation=cv2.INTER_AREA)
# cv2.imshow("Skewed resizing", img_scaled)
display("Skewed resizing", img_scaled)

# 8 Save the image to an output file
# save an image
output_file = input_file[:-4] + '_cropped.jpg'
cv2.imwrite(output_file, img_cropped)
cv2.waitKey()
# The waitKey() function displays the images until you hit a key on the keyboard.


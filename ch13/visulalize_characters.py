# 1 Lets make the basic imports
import cv2
import numpy as np

# 2 Define the input file name
# Load input data
input_file = 'ch13/letter.data'

# 3 Define visualization parameters
# Define visualization parameters
# Define visualization parameters
scaling_factor = 10
start_index = 6
end_index = 6
end_index = -1
h, w = 16, 8

# 4 Keep looking through the file until the user
# presses Escape key. Split the line into tab-separated characters

# 4 Loop until you encounter the Esc key
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = np.array([255*float(x) for x in line.split('\t')[start_index:end_index]])

        # 5 Reshape the array into the required shape, resize it, and display it:
        img = np.reshape(data, (h,w)) 
        img_scaled = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor) 
        cv2.imshow('Image', img_scaled)

        # 6 Ifthe user press escape, break the loop
        c = cv2.waitKey()
        if c == 27:
            break

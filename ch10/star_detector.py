# 1 Make the basic imports
import cv2

# 2 Define a class to handle all the functions that are
# related to Start feature detection
class StarFeatureDetector(object):
    def __init__(self) -> None:
        self.detector = cv2.xfeatures2d.StarDetector_create() 

# 3 Define a function to run the detector on the input image
    def detect(self, img):
        return self.detector.detect(img)

# 4 Load the input image in the main function. We will use table.jpg

if __name__ == '__main__':
    # load input image 'flowers.jpg'
    input_file='ch10/flowers.jpg'
    input_img=cv2.imread(input_file)

    # 5 Convert to grayscale
    img_gray=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # 6 Detect features using the Star feature detector
    # Detect features using Star features detector
    keypoints=StarFeatureDetector().detect(input_img)

    # 7 Draw keypoints on top of the input image
    cv2.drawKeypoints(input_img, keypoints, input_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 8 Display the output image
    cv2.imshow("Start features", input_img)
    cv2.waitKey()
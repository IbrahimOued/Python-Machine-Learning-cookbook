# 1 Let's make the basic imports
import argparse
import numpy as np
import cv2
import imageio
from sklearn import cluster
import matplotlib.pyplot as plt

# 2 Let's create a function to parse the input arguments
# We will be able to pass the image and the number of bits per pixel as input arguments


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Compress the input image using clustering')
    parser.add_argument('--input-file', dest='input_file',
                        required=True, help='Input image')
    parser.add_argument('--num-bits', dest='num_bits', required=False,
                        type=int, help="Number of bits used to represent each pixel")
    return parser

# 3 Let's create a function to read the input image:


def compress_image(img, num_clusters):
    # Convert input image into (num_samples, num_features)
    # array to run kmeans clustering algorithm
    X = img.reshape((-1, 1))

    # Run kmeans on input data
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=4, random_state=5)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_

    # Assign each value to the nearest centroid and
    # reshape it to the original image shape
    input_image_compressed = np.choose(labels, centroids).reshape(img.shape)

    return input_image_compressed

# 4 Once we compress the image, we need to see how it affects the quality.
# Let's define a function to plot the output image


def plot_image(img, title):
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 5 We are now ready to use all these functions.
# Let's define the main function that takes the
# input arguments, processes them, and extracts the output image:
if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    input_file = args.input_file
    num_bits = args.num_bits

    if not 1 <= num_bits <= 8:
        raise TypeError('Number of bits should be between 1 and 8')

    num_clusters = np.power(2, num_bits)

    # Print compression rate
    compression_rate = round(100 * (8.0 - args.num_bits) / 8.0, 2)
    print("The size of the image will be reduced by a factor of", 8.0/args.num_bits)
    print("Compression rate = " + str(compression_rate) + "%")

    # 6 Let's load the input image:
    # Load input image
    input_image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

    # original image
    plot_image(input_image, 'Original image')

    # 7 Now, let's compress this image using the input argument:
    # compressed image
    input_image_compressed = compress_image(input_image, num_clusters)
    plot_image(input_image_compressed, 'Compressed image; compression rate = '
               + str(compression_rate) + '%')
    # 8 We are now ready to run the code; run the following command on your Terminal:
    # $ python vector_quantization.py --input-file flower_image.jpg --num-bits 4

    # 9 Let's compress the image further by reducing the number of bits to 2.
    # Run the following command on your Terminal:
    # $ python vector_quantization.py --input-file flower_image.jpg --num-bits 2

    # 10 If you reduce the number of bits to 1, you can see that it will become
    # a binary image with black and white as the only two colors. Run the following command:
    # $ python vector_quantization.py --input-file flower_image.jpg --num-bits 1
    # We have seen how, by compressing the image further, the quality of the image has undergone considerable downsizing.

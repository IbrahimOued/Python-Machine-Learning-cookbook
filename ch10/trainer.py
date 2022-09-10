# 1 Late's make the basic imports
import argparse
from cProfile import label
import _pickle as pickle

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing

# 2 Define an argument parser


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the classifier')
    parser.add_argument("--feature-map-file", dest="feature_map_file", required=True,
                        help="Input pickle file containing the feature map")
    parser.add_argument("--model-file", dest="model_file", required=False,
                        help="Output file where the trained model will be stored")
    return parser

# 3 Define a class to handle ERF training. We will use a label encoder to encode
# our training labels


class ERFTrainer(object):
    def __init__(self, X, label_words) -> None:
        self.le = preprocessing.LabelEncoder()
        self.clf = ExtraTreesClassifier(
            n_estimators=100, max_depth=16, random_state=0)

        # 4 Encode the labels and train the classifier
        y = self.encode_labels(label_words)
        self.clf.fit(np.asarray(X), y)

    # 5 Define a function to encode the labels
    def encode_labels(self, label_words):
        self.le.fit(label_words)
        return np.array(self.le.transform(label_words), dtype=np.float32)

    # 6 Define a function to classify an unknown datapoint

    def classify(self, X):
        label_nums = self.clf.predict(np.asarray(X))
        label_words = self.le.inverse_transform([int(x) for x in label_nums])
        return label_words


# 7 Define the main function and parse the input arguments
if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    feature_map_file = args.feature_map_file
    model_file = args.model_file

    # 8 Load the feature map that we created in the previous recipe:
    # Load the feature map
    with open(feature_map_file, 'rb') as f:
        feature_map = pickle.load(f)

    # 9 Extract the feature vectors:
    # Extract feature vectors and the labels
    label_words = [x['object_class'] for x in feature_map]
    dim_size = feature_map[0]['feature_vector'].shape[1]
    X = [np.reshape(x['feature_vector'], (dim_size,)) for x in feature_map]

    # 10 Train the ERF, which is based on the training data:
    # Train the Extremely Random Forests classifier
    erf = ERFTrainer(X, label_words)

    # 11 Save the trained ERF model, as follows:
    if args.model_file:
        with open(args.model_file, 'wb') as f:
            pickle.dump(erf, f)

    # 12 Now, you should run the code in the Terminal:
    # run python build_features.py --data-folder ./training_images/ --codebook-file ./codebook.pkl --feature-map-file ./feature_map.pkl
    # This will generate a file called erf.pkl. We will use this file in the next recipe.

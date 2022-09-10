# 1 Let's make the basic imports
import numpy as np
import json

# 2 We will now define a function to compute the euclidian
# score between two users. The first step is to check whether the users
# are present in the database
# Returns the euclidian score between user1 and user2


def euclidian_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('User ' + user1 + ' not present in the dataset')

    if user2 not in dataset:
        raise TypeError('User ' + user2 + ' not present in the dataset')


    # 3 In order to compute the score, we need to extract the movies that
    # both users have rated
    # Movies rated by both user1 and user2
    rated_by_both = {}
    for item in dataset[user1]:
        if item in dataset[user2]:
            rated_by_both[item] = 1

    # 4 If there are no common movies, then there is no
    # similarity between the two users(or at least, we cannot compute it given the
    # ratings in the database)
    # If there are no common movies, the score is 0
    if len(rated_by_both) == 0:
        return 0

    # 5 For each of the common ratings, we just compute the square root of the
    # sum of squared differences and normalize it, so that the score is between 0 and 1
    squared_differences = []
    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_differences.append(
                np.square(dataset[user1][item] - dataset[user2][item]))
            return 1/(1 + np.sqrt(np.sum(squared_differences)))

            # If the ratings are similar, then the sul of squared differences will be very low
            # Hence, the score will become high, which is what we want from the metric
# 6 We will use the movie_ratings.json file, let's load it
if __name__ == '__main__':
    with open('ch06/movie_ratings.json', 'r') as f:
        data = json.loads(f.read())

    # 7 Let's consider 2 random users and compute the euclidian distance score
    user1 = 'John Carson'
    user2 = 'Michelle Peterson'

    print('Euclidian score:')
    print(euclidian_score(data, user1, user2))

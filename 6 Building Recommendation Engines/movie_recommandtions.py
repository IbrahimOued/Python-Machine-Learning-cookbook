# 1 Let's make the basic imports
import json
import numpy as np
from pearson_score import pearson_score

# 2 We will define a function to generate movie recommandations for
# a given user. The first step is to check whether the user exusts in the dataset
# Generate recommendations for a given user
def generate_recommendations(dataset, user):
    if user not in dataset:
        raise TypeError('User ' + user + ' not present in the dataset')
    # 3 Let's now compute the Pearson score of this user with all the other users in the dataset:
    total_scores = {}
    similarity_sums = {}

    for u in [x for x in dataset if x != user]:
        similarity_score = pearson_score(dataset, user, u)

        if similarity_score <= 0:
            continue

    # 4 We need to find the movies that haven't been rated by this user:
    for item in [x for x in dataset[u] if x not in dataset[user] or dataset[user][x] == 0]:
        total_scores.update({item: dataset[u][item] * similarity_score})
        similarity_sums.update({item: similarity_score})

    # 5 If the user has watched every single movie in the database, then we cannot
    # recommend anything to this user. Let's take care of this condition
    if len(total_scores) == 0:
        return ['No recommendations possible']

    # 6 We now have a list of these scores. Let's create a normalized list of movie ranks
    # Create the normalized list
    movie_ranks = np.array([[total/similarity_sums[item], item]
                            for item, total in total_scores.items()])

    # 7 We need to sort the list in descending order based on the score:
    #  Sort in decreasing order based on the first column
    movie_ranks = movie_ranks[np.argsort(movie_ranks[:, 0])[::-1]]

    # 8 We are finally ready to extract the movie recommendations:
    # Extract the recommended movies
    recommendations = [movie for _, movie in movie_ranks]

    return recommendations


# 9 Now, let's define the main function and load the dataset:
if __name__ == '__main__':
    data_file = 'ch06/movie_ratings.json'

    with open(data_file, 'r') as f:
        data = json.loads(f.read())
# 10 Let's now generate recommendations for Michael Henry, as follows:
user = 'Michael Henry'
print("Recommendations for " + user + ":")
movies = generate_recommendations(data, user)
for i, movie in enumerate(movies):
    print(str(i+1) + '. ' + movie)

# 11 The John Carson user has watched all the movies. Therefore,
# if we try to generate recommendations for him,
# it should display 0 recommendations. Let's see whether this happens, as follows:
user = 'John Carson'
print("Recommendations for " + user + ":")
movies = generate_recommendations(data, user)
for i, movie in enumerate(movies):
    print(str(i+1) + '. ' + movie)

# 12 If you run this code, you will see the following output on your Terminal

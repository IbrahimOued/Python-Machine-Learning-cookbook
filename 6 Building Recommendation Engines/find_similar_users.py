# 1 Let's make some basic imports
import json 
import numpy as np 
 
from pearson_score import pearson_score

# 2 Let's define a function to find users who
# are similar to the input user. It takes three
# input arguments: the database, the input user,
# and the number of similar users that we are
# looking for. Our first step is to check whether
# the user is present in the database. If the user
# exists, we need to compute the Pearson correlation
# score between this user and all the other users in the database
# Finds a specified number of users who are similar to the input user 
def find_similar_users(dataset, user, num_users): 
    if user not in dataset: 
        raise TypeError('User ' + user + ' not present in the dataset')
    # Compute Pearson scores for all the users 
    scores = np.array([[x, pearson_score(dataset, user, x)] for x in dataset if user != x]) 

    # 3 The next step is to sort these scores in descending order
    # Sort the scores based on second column 
    scores_sorted = np.argsort(scores[:, 1]) 
    # Sort the scores in decreasing order (highest score first)  
    scored_sorted_dec = scores_sorted[::-1]

    # 4 Let's extract the k top scores and then return them
    # Extract top 'k' indices
    top_k = scored_sorted_dec[0: num_users]
    return scores[top_k]


# 5 Let's now define the main function and load the input database
if __name__ == '__main__':
    data_file = 'ch06/movie_ratings.json'
    with open(data_file, 'r') as f:
        data = json.loads(f.read())

    # 6 We want to find three similar users to, John Carson, for exemple. We do
    # this by using the following steps
    user = 'John Carson'
    print("Users similar to " + user + ":\n")
    similar_users = find_similar_users(data, user, 3) 
    print("User\t\t\tSimilarity score\n")
    for item in similar_users:
        print(item[0], '\t\t', round(float(item[1]), 2))
    
    # 7 Let's run the code


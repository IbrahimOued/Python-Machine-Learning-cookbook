# 1 Let's make the basic imports
import numpy as np
import json

# 2 We will define a function to compute the Pearson correlation score between two users
# in the database. Our 1st step is to confirm that these users are present in the database
# Returns the Pearson correlation score between user1 and user2  
def pearson_score(dataset, user1, user2): 
    if user1 not in dataset: 
        raise TypeError('User ' + user1 + ' not present in the dataset') 
 
    if user2 not in dataset: 
        raise TypeError('User ' + user2 + ' not present in the dataset') 

    # 3 The next step is to get the movies that both of these users rated:
    # Movies rated by both user1 and user2 
    rated_by_both = {} 
 
    for item in dataset[user1]: 
        if item in dataset[user2]: 
            rated_by_both[item] = 1 
 
    num_ratings = len(rated_by_both)

    # 4 f there are no common movies, then there is no discernible
    # similarity between these users; hence, we return 0:
    # If there are no common movies, the score is 0  
    if num_ratings == 0: 
        return 0

    # 5 We need to compute the sum of squared values of common movie ratings:
    # Compute the sum of ratings of all the common preferences  
    user1_sum = np.sum([dataset[user1][item] for item in rated_by_both]) 
    user2_sum = np.sum([dataset[user2][item] for item in rated_by_both])

    # 6 Now, let's compute the sum of squared ratings of all the common movie ratings:
    # Compute the sum of squared ratings of all the common preferences  
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in rated_by_both]) 
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in rated_by_both])

    # 7 Let's now compute the sum of the products
    # Compute the sum of products of the common ratings  
    product_sum = np.sum([dataset[user1][item] * dataset[user2][item] for item in rated_by_both])

    # 8 We are now ready to compute the various elements that we require to calculate the Pearson correlation score:
    # Compute the Pearson correlation 
    Sxy = product_sum - (user1_sum * user2_sum / num_ratings) 
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    # 9 We need to take care of the case where the denominator becomes 0
    if Sxx * Syy == 0:
        return 0
    
    # 10 If everything is good, we return the Pearson correlation score, as follows
    return Sxy / np.sqrt(Sxx * Syy)

# 11 Let's now define the main function and compute the Pearson correlation score between two users
if __name__=='__main__': 
    data_file = 'ch06/movie_ratings.json' 
 
    with open(data_file, 'r') as f: 
        data = json.loads(f.read()) 
 
    user1 = 'John Carson' 
    user2 = 'Michelle Peterson' 
 
    print("Pearson score:")
    print(pearson_score(data, user1, user2)) 
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# fetch data from a dataset
# movielens data set is a big csv dataset that contains 70 K movie rating from 1 K users
# each user has rated at least twenty movies on a scale from 1 to 5
data = fetch_movielens(min_rating=4.0)

# print training and testing data - we'll store it in a dictionary
print(repr(data['train']))

print(repr(data['test']))

# create a model
model = LightFM(loss='warp')  # warp - weighted approximate-rank pairwise

# train model
model.fit(data['train'], epochs=30, num_threads=2)


# user_ids - ids of users that we want to make recommendations for
def sample_recommendation(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items = data['train'].shape

    # generate recommendation for each user we input
    for user_id in user_ids:
        # movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        print("user %s" % user_id)
        print("known positives: ")

        for x in known_positives[:3]:
            print("%s" % x)

        print("recommended: ")

        for x in top_items[:3]:
            print("         %s" % x)


sample_recommendation(model, data, [3, 25, 450])

# 1 Let's make the basic imports
import numpy as np
import pandas as pd
import tensorflow as tf

# 2 We will load the data in the MovieLens 1M dataset
data = pd.read_csv('ch06/ratings.csv', sep=';', names=['user', 'item', 'rating', 'timestamp'], header=None)
data = data.iloc[:, 0:3]
num_items = data.item.nunique()
num_users = data.user.nunique()

print('Item: ', num_items)
print('USers: ', num_users)
# The following are returned
# Item: 3706
# Users: 6040

# 3 Now, let's perform data scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data['rating'] = data['rating'].values.astype(float)
data_scaled = pd.DataFrame(scaler.fit_transform(data['rating'].values.reshape(-1, 1)))
data['rating'] = data_scaled

# 4 We will builf the user item matrix
user_item_matrix = data.pivot(index='user', columns='item', values='rating')
user_item_matrix.fillna(0, inplace=True)

users = user_item_matrix.index.tolist()
items = user_item_matrix.columns.tolist()

user_item_matrix = user_item_matrix.to_numpy()

# 5 Now, we can set some network parameters
num_input = num_items
num_hidden1 = 10
num_hidden2 = 5

# 6 Now, we will initialize the TensorFlow placeholder. Then, weights and biases are randomly initialized:
# X = tf.placeholder(tf.float64, [None, num_input])

# X = tf.placeholder(tf.float32, name=key)

X = tf.keras.Input([None, num_input], dtype=tf.dtypes.float64)

weights = {
    'EncoderH1': tf.Variable(tf.random.normal([num_input, num_hidden1], dtype=tf.float64)),
    'EncoderH2': tf.Variable(tf.random.normal([num_hidden1, num_hidden2], dtype=tf.float64)),
    'DecoderH1': tf.Variable(tf.random.normal([num_hidden2, num_hidden1], dtype=tf.float64)),
    'DecoderH2': tf.Variable(tf.random.normal([num_hidden1, num_input], dtype=tf.float64)),
}

biases = {
    'EncoderB1': tf.Variable(tf.random.normal([num_hidden1], dtype=tf.float64)),
    'EncoderB2': tf.Variable(tf.random.normal([num_hidden2], dtype=tf.float64)),
    'DecoderB1': tf.Variable(tf.random.normal([num_hidden1], dtype=tf.float64)),
    'DecoderB2': tf.Variable(tf.random.normal([num_input], dtype=tf.float64)),
}

# 7 Now, we can build the encoder and decoder model, as follows:
def encoder(x):
    Layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['EncoderH1']), biases['EncoderB1']))
    Layer2 = tf.nn.sigmoid(tf.add(tf.matmul(Layer1, weights['EncoderH2']), biases['EncoderB2']))
    return Layer2

def decoder(x):
    Layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['DecoderH1']), biases['DecoderB1']))
    Layer2 = tf.nn.sigmoid(tf.add(tf.matmul(Layer1, weights['DecoderH2']), biases['DecoderB2']))
    return Layer2

# 8 We will construct the model and predict the value, as follows:

EncoderOp = encoder(X)
DecoderOp = decoder(EncoderOp)

YPred = DecoderOp
YTrue = X

# 9 We will now define loss and optimizer, and minimize the squared error and the evaluation metrics:
loss = tf.losses.mean_squared_error(YTrue, YPred)
optimizer = tf.keras.optimizers.RMSprop(0.03).minimize(loss, var_list=[X], tape=tf.GradientTape())
evalX = tf.placeholder(tf.int32, )
evalY = tf.placeholder(tf.int32, )
pre, preOp = tf.metrics.precision(labels=evalX, predictions=evalY)

# 10 Let's now initialize the variables, as follows:
init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
pred_data = pd.DataFrame()

# 11 Finally, we can start to train our model:
with tf.Session() as session:
    Epochs = 120
    BatchSize = 200

    session.run(init)
    session.run(local_init)

    NumBatches = int(user_item_matrix.shape[0] / BatchSize)
    UserItemMatrix = np.array_split(user_item_matrix, NumBatches)
    
    for i in range(Epochs):

        AvgCost = 0

        for batch in UserItemMatrix:
            _, l = session.run([optimizer, loss], feed_dict={X: batch})
            AvgCost += l

        AvgCost /= NumBatches

        print("Epoch: {} Loss: {}".format(i + 1, AvgCost))

    UserItemMatrix = np.concatenate(UserItemMatrix, axis=0)

    Preds = session.run(DecoderOp, feed_dict={X: UserItemMatrix})

    PredData = pred_data.append(pd.DataFrame(Preds))

    PredData = PredData.stack().reset_index(name='rating')
    PredData.columns = ['user', 'item', 'rating']
    PredData['user'] = PredData['user'].map(lambda value: users[value])
    PredData['item'] = PredData['item'].map(lambda value: items[value])
    
    keys = ['user', 'item']
    Index1 = PredData.set_index(keys).index
    Index2 = data.set_index(keys).index

    TopTenRanked = PredData[~Index1.isin(Index2)]
    TopTenRanked = TopTenRanked.sort_values(['user', 'rating'], ascending=[True, False])
    TopTenRanked = TopTenRanked.groupby('user').head(10)
    
    print(TopTenRanked.head(n=10))
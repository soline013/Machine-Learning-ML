%tensorflow_version 1.x
import tensorflow as tf



import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# One cell RNN input_dim (4) -> output_dim (2)
hidden_size = 2
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size) #.contrib.rnn.BasicRNNCell()

x_data = np.array([[h]], dtype=np.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32) #.nn.dynamic_rnn()

sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
'''
array([[[ 0.08458287, -0.12298959]]], dtype=float32)
'''

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5
hidden_size = 2
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
x_data = np.array([[h, e, l, l, o]], dtype=np.float32)
print(x_data.shape)
pp.pprint(x_data)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
'''
(1, 5, 4)
array([[[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]]], dtype=float32)
array([[[-0.4770963 ,  0.36427602],
        [-0.7350998 , -0.44640467],
        [ 0.72994703,  0.86558473],
        [ 0.3648536 , -0.14056201],
        [ 0.3807112 , -0.44179758]]], dtype=float32)
'''

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch 3
x_data = np.array([[h, e, l, l, o],
                   [e, o, l, l, l],
                   [l, l, e, e, l]], dtype=np.float32)
pp.pprint(x_data)
    
hidden_size = 2
cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True) #.BasicLSTMCell()
outputs, _states = tf.nn.dynamic_rnn(
    cell, x_data, dtype=tf.float32)
sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
'''
array([[[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]],

       [[0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [0., 0., 1., 0.],
        [0., 0., 1., 0.],
        [0., 0., 1., 0.]],

       [[0., 0., 1., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.]]], dtype=float32)
array([[[-0.09509785,  0.07050316],
        [-0.17198484,  0.12891157],
        [-0.12589686,  0.07874432],
        [-0.10102756,  0.02463672],
        [ 0.07038042, -0.08202118]],

       [[-0.12700574,  0.08467649],
        [ 0.05858154, -0.0424508 ],
        [ 0.00083902, -0.04020142],
        [-0.02455638, -0.03646189],
        [-0.04260794, -0.03561934]],

       [[-0.02127589, -0.01376881],
        [-0.03791282, -0.0226119 ],
        [-0.15317386,  0.07066068],
        [-0.21553406,  0.12426013],
        [-0.15725984,  0.07271042]]], dtype=float32)
'''



import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

y_data = [[1, 0, 2, 3, 3, 4]]    # ihello

num_classes = 5
input_dim = 5
hidden_size = 5
batch_size = 1
sequence_length = 6
learning_rate = 0.1

X = tf.placeholder(
    tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32) #.zero_state()
outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fc, fc_w) + fc_b
outputs = tf.contrib.layers.fully_connected( #.contrib.layers.fully_connected()
    inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length]) #.ones()
sequence_loss = tf.contrib.seq2seq.sequence_loss( #.contrib.seq2seq.sequence_loss()
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#.AdamOptimizer()

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))
'''
45 loss: 0.0013789787 prediction:  [[1 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  ihello
46 loss: 0.0013257032 prediction:  [[1 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  ihello
47 loss: 0.0012777768 prediction:  [[1 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  ihello
48 loss: 0.0012345855 prediction:  [[1 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  ihello
49 loss: 0.0011956157 prediction:  [[1 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  ihello
'''



import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

sample = " if you want you"
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex

# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
rnn_hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.2

sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello

X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

# flatten the data (ignore batches for now). No effect if the batch size is 1
X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
X_for_softmax = tf.reshape(X_one_hot, [-1, rnn_hidden_size])

# softmax layer (rnn_hidden_size -> num_classes)
softmax_w = tf.get_variable("softmax_w", [rnn_hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes]) #.get_variable()
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b

# expend the data (revive the batches)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])

# Compute sequence cost/loss
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)  # mean all sequence loss
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "Prediction:", ''.join(result_str))
'''
2995 loss: 0.27728233 Prediction: yf you yant you
2996 loss: 0.2772823 Prediction: yf you yant you
2997 loss: 0.2772823 Prediction: yf you yant you
2998 loss: 0.2772823 Prediction: yf you yant you
2999 loss: 0.27728224 Prediction: yf you yant you
'''



from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10  # Any arbitrary number
learning_rate = 0.1

dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

# One-hot encoding
X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)  # check out the shape

# Make a lstm cell with hidden_size (each unit output vector size)
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
#.MultiRNNCell()

# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

# All weights are 1 (equal weights)
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, l, results = sess.run(
        [train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), l)

# Let's print the last char of each result to check it works
results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')
'''
499 165 py of the  0.22880386
499 166 h of the s 0.22880386
499 167  of the se 0.22880386
499 168 tf the sea 0.22880386
499 169 n the sea. 0.22880386
t you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.
'''



import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.set_random_seed(777)  # reproducibility

#if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    #matplotlib.use('Agg') 사용하지 않음.

import matplotlib.pyplot as plt

def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0) #.min()
    denominator = np.max(data, 0) - np.min(data, 0) #.max()
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500

# Open, High, Low, Volume, Close
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)

# train/test split
train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]  # Index from [train_size - seq_length] to utilize past sequence

# Scale each
train_set = MinMaxScaler(train_set) #MinMaxScaler()
test_set = MinMaxScaler(test_set)

# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]  # Next close price
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))
'''
[[0.79125947 0.78809309 0.83049214 0.16295496 0.7958306 ]
 [0.77153656 0.76256986 0.79947427 0.10961814 0.79493482]
 [0.79725431 0.81461829 0.83903438 0.12098623 0.8285645 ]
 [0.81529885 0.82251706 0.84757626 0.1060959  0.83698714]
 [0.83034597 0.81556125 0.85575466 0.07523435 0.84403567]
 [0.84347469 0.8426171  0.88749985 0.10121322 0.86858608]
 [0.86925245 0.87626864 0.92209184 0.1140722  0.90185774]] -> [0.90908564]
[[0.77153656 0.76256986 0.79947427 0.10961814 0.79493482]
 [0.79725431 0.81461829 0.83903438 0.12098623 0.8285645 ]
 [0.81529885 0.82251706 0.84757626 0.1060959  0.83698714]
 [0.83034597 0.81556125 0.85575466 0.07523435 0.84403567]
 [0.84347469 0.8426171  0.88749985 0.10121322 0.86858608]
 [0.86925245 0.87626864 0.92209184 0.1140722  0.90185774]
 [0.88723699 0.88829938 0.92518158 0.08714288 0.90908564]] -> [0.90030461]
[[0.79725431 0.81461829 0.83903438 0.12098623 0.8285645 ]
 [0.81529885 0.82251706 0.84757626 0.1060959  0.83698714]
 [0.83034597 0.81556125 0.85575466 0.07523435 0.84403567]
 [0.84347469 0.8426171  0.88749985 0.10121322 0.86858608]
 [0.86925245 0.87626864 0.92209184 0.1140722  0.90185774]
 [0.88723699 0.88829938 0.92518158 0.08714288 0.90908564]
 [0.88939504 0.88829938 0.94014512 0.13380794 0.90030461]] -> [0.93124657]
[[0.81529885 0.82251706 0.84757626 0.1060959  0.83698714]
 [0.83034597 0.81556125 0.85575466 0.07523435 0.84403567]
 [0.84347469 0.8426171  0.88749985 0.10121322 0.86858608]
 [0.86925245 0.87626864 0.92209184 0.1140722  0.90185774]
 [0.88723699 0.88829938 0.92518158 0.08714288 0.90908564]
 [0.88939504 0.88829938 0.94014512 0.13380794 0.90030461]
 [0.89281215 0.89655181 0.94323484 0.12965206 0.93124657]] -> [0.95460261]
[[0.83034597 0.81556125 0.85575466 0.07523435 0.84403567]
 [0.84347469 0.8426171  0.88749985 0.10121322 0.86858608]
 [0.86925245 0.87626864 0.92209184 0.1140722  0.90185774]
 [0.88723699 0.88829938 0.92518158 0.08714288 0.90908564]
 [0.88939504 0.88829938 0.94014512 0.13380794 0.90030461]
 [0.89281215 0.89655181 0.94323484 0.12965206 0.93124657]
 [0.91133638 0.91818448 0.95944078 0.1885611  0.95460261]] -> [0.97604677]
[step: 495] loss: 0.7044439911842346
[step: 496] loss: 0.703827440738678
[step: 497] loss: 0.7032116055488586
[step: 498] loss: 0.7025963068008423
[step: 499] loss: 0.7019817233085632
RMSE: 0.051355425268411636
'''

# Plot predictions
plt.plot(testY)
plt.plot(test_predict)
plt.xlabel("Time Period") #.xlabel()
plt.ylabel("Stock Price") #.ylabel()
plt.show()
'''
![image](https://user-images.githubusercontent.com/66259854/103098065-d3372f00-464c-11eb-97de-c264316d6c2f.png)
'''

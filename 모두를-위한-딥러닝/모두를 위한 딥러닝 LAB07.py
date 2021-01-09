%tensorflow_version 1.x
import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



x_data = [[1,2,1], [1,3,2], [1,3,4], [1,5,5,], [1,7,5], [1,2,5], [1,6,6,], [1,7,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]
x_test = [[2,1,1,], [3,1,2,], [3,3,4]]
y_test = [[0,0,1], [0,0,1], [0,0,1]]

x = tf.placeholder("float", shape=[None, 3])
y = tf.placeholder("float", shape=[None, 3])
w = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3])) 
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for step in range(201):
    cost_val, W_val, _ = sess.run([cost, w, optimizer], feed_dict={x: x_data, y: y_data})
    print("Step:", step, "Cost:", cost_val, "\n", W_val)
  print("Prediction:", sess.run(prediction, feed_dict={x: x_test}))
  print("Accuracy:", sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
'''
Step: 196 Cost: 0.58980745 
 [[-0.22758487 -0.9732693   1.678012  ]
 [ 0.90166056  0.709741    0.6607038 ]
 [ 0.22625132  0.07292911 -0.81909895]]
Step: 197 Cost: 0.5891347 
 [[-0.23167685 -0.973026    1.6818607 ]
 [ 0.90185565  0.71003866  0.660211  ]
 [ 0.22765663  0.07271246 -0.8202876 ]]
Step: 198 Cost: 0.5884651 
 [[-0.23576029 -0.97277963  1.6856978 ]
 [ 0.90204793  0.7103343   0.6597231 ]
 [ 0.22906105  0.07249619 -0.82147574]]
Step: 199 Cost: 0.58779883 
 [[-0.23983523 -0.9725303   1.6895235 ]
 [ 0.90223753  0.7106279   0.6592399 ]
 [ 0.23046465  0.07228023 -0.82266337]]
Step: 200 Cost: 0.5871357 
 [[-0.24390176 -0.972278    1.6933377 ]
 [ 0.90242434  0.7109197   0.6587613 ]
 [ 0.23186731  0.0720647  -0.8238505 ]]
Prediction: [2 2 2]
Accuracy: 1.0
'''



#From GitHub
import numpy as np
tf.set_random_seed(777)  # for reproducibility


def min_max_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


xy = np.array(
    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)

# very important. It does not work without it.
xy = min_max_scaler(xy)
print(xy)
'''
[[0.99999999 0.99999999 0.         1.         1.        ]
 [0.70548491 0.70439552 1.         0.71881782 0.83755791]
 [0.54412549 0.50274824 0.57608696 0.606468   0.6606331 ]
 [0.33890353 0.31368023 0.10869565 0.45989134 0.43800918]
 [0.51436    0.42582389 0.30434783 0.58504805 0.42624401]
 [0.49556179 0.42582389 0.31521739 0.48131134 0.49276137]
 [0.11436064 0.         0.20652174 0.22007776 0.18597238]
 [0.         0.07747099 0.5326087  0.         0.        ]]
'''

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        _, cost_val, hy_val = sess.run(
            [train, cost, hypothesis], feed_dict={X: x_data, Y: y_data}
        )
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
'''
100 Cost:  0.49615628 
Prediction:
 [[1.3413807 ]
 [2.1911035 ]
 [1.49674   ]
 [0.74714625]
 [1.098949  ]
 [0.96328515]
 [0.4680009 ]
 [0.68740016]]
'''

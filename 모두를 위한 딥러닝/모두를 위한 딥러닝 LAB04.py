%tensorflow_version 1.x
import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



x1_data = [73, 93, 89, 96, 73]
x2_data = [80, 88, 91, 98, 66]
x3_data = [75, 93, 90, 100, 70]
y_data = [152, 185, 180, 196, 142]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1*w1 + x2*w2 + x3*w3 + b

cost = tf.reduce_mean(tf.square(hypothesis -y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
  cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                 feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data,})
  if step % 10 == 0:
    print("Step: ", step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
'''
Step:  1960 Cost:  0.3213133 
Prediction:
 [150.94762 185.02405 180.40924 196.00891 142.57526]
Step:  1970 Cost:  0.32036033 
Prediction:
 [150.94925 185.0229  180.40973 196.00925 142.57379]
Step:  1980 Cost:  0.31941897 
Prediction:
 [150.95088 185.0218  180.41025 196.0096  142.57234]
Step:  1990 Cost:  0.31847495 
Prediction:
 [150.95251 185.02069 180.41074 196.00995 142.57089]
Step:  2000 Cost:  0.31753626 
Prediction:
 [150.95413 185.01956 180.41122 196.01028 142.56943]
'''



x_data = [[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]]
y_data = [[152], [185], [180], [196], [142]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])
w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w)+b #.matmul()

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
  cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x: x_data, y: y_data})
  if step % 10 == 0:
    print("Step: ", step, "Cost: ", cost_val, "\nPrediction\n", hy_val)
'''
Step:  2000 Cost:  0.45228338 
Prediction
 [[150.69113]
 [185.18141]
 [180.27364]
 [196.13223]
 [142.65039]]
'''

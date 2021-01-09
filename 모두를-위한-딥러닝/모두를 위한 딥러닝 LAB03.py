%tensorflow_version 1.x
import tensorflow as tf

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()



x = [1, 2, 3]
y = [1, 2, 3]
W = tf.placeholder(tf.float32)
hypothesis = x * W

cost = tf.reduce_mean(tf.square(hypothesis - y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []
for i in range(-30, 50):
  feed_W = i * 0.1
  curr_cost, curr_W = sess.run([cost, W], feed_dict={W : feed_W})
  W_val.append(curr_W) #.append()
  cost_val.append(curr_cost)

plt.plot(W_val, cost_val) #.plot()
plt.show() #.show()
'''
![image](https://user-images.githubusercontent.com/66259854/93185850-e3e82700-f778-11ea-878b-93b843a27eab.png)
'''



x_data = [1, 2, 3]
y_data = [1, 2, 3]
W = tf.Variable(tf.random_normal([1]), name='weight')
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
hypothesis = x * W

cost = tf.reduce_sum(tf.square(hypothesis -y)) #.reduce_sum()

# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * x - y) * x)
descent = W - learning_rate * gradient
update = W.assign(descent) #.assign()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
  sess.run(update, feed_dict={x: x_data, y: y_data})
  print("step:", step, "cost:", sess.run(cost, feed_dict={x: x_data, y: y_data}), "W:", sess.run(W))
'''
step: 16 cost: 6.8753536e-10 W: [0.99999297]
step: 17 cost: 1.9607072e-10 W: [0.99999624]
step: 18 cost: 5.8960836e-11 W: [0.999998]
step: 19 cost: 1.6896706e-11 W: [0.9999989]
step: 20 cost: 4.2241766e-12 W: [0.99999946]
'''



x = [1, 2, 3]
y = [1, 2, 3]
W = tf.Variable(5.0)
hypothesis = x * W

cost = tf.reduce_mean(tf.square(hypothesis -y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
  print("step:", step, "W:", sess.run(W))
  sess.run(train)
'''
step: 0 W: 5.0
step: 1 W: 1.2666664
step: 2 W: 1.0177778
step: 3 W: 1.0011852
step: 4 W: 1.000079
step: 5 W: 1.0000052
step: 6 W: 1.0000004
step: 7 W: 1.0
step: 8 W: 1.0
step: 9 W: 1.0
'''



x = [1, 2, 3]
y = [1, 2, 3]
W = tf.Variable(5.0)
hypothesis = x * W
gradient = tf.reduce_mean((W * x - y) * x) * 2

cost = tf.reduce_mean(tf.square(hypothesis -y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# Get gradients
# Optional: modify gradient if necessary
# gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
gvs = optimizer.compute_gradients(cost, [W]) #.compute_gradients()
apply_gradients = optimizer.apply_gradients(gvs) #.apply_gradients()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
  print(step, sess.run([gradient, W, gvs]))
  sess.run(apply_gradients)
'''
95 [0.0033854644, 1.0003628, [(0.0033854644, 1.0003628)]]
96 [0.0030694802, 1.0003289, [(0.0030694804, 1.0003289)]]
97 [0.0027837753, 1.0002983, [(0.0027837753, 1.0002983)]]
98 [0.0025234222, 1.0002704, [(0.0025234222, 1.0002704)]]
99 [0.0022875469, 1.0002451, [(0.0022875469, 1.0002451)]]
'''

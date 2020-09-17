%tensorflow_version 1.x
import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5,], [1,2,5,6,], [1,6,6,6], [1,7,7,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]
#One Hot Encoding [0,0,1]=2, [0,1,0]=1, [1,0,0]=0

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])
nb_classes = 3
w = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b) #.softmax()

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(optimizer, feed_dict={x: x_data, y: y_data})
    if step % 200 == 0:
      print("Step:", step, "Cost:", sess.run(cost, feed_dict={x: x_data, y: y_data}))
'''
Step: 1200 Cost: 0.20908271
Step: 1400 Cost: 0.19131012
Step: 1600 Cost: 0.17623389
Step: 1800 Cost: 0.16328973
Step: 2000 Cost: 0.15206131
'''



# Testing & One-hot encoding
a = sess.run(hypothesis, feed_dict={x: [[1,11,7,9]]})
print(a, sess.run(tf.arg_max(a, 1))) #.arg_max()

print('----------')

b = sess.run(hypothesis, feed_dict={x: [[1,3,4,3]]}) 
print(b, sess.run(tf.arg_max(b, 1)))

print('----------')

c = sess.run(hypothesis, feed_dict={x: [[1,1,0,1]]}) 
print(c, sess.run(tf.arg_max(c, 1)))

print('----------')

all = sess.run(hypothesis, feed_dict={x: [[1,11,7,9], [1,3,4,3], [1,1,0,1]]}) 
print(all, sess.run(tf.arg_max(all, 1)))
'''
[[5.5885576e-03 9.9440151e-01 9.9859089e-06]] [1]
----------
[[0.84592074 0.13424273 0.01983646]] [0]
----------
[[1.059492e-08 3.194520e-04 9.996805e-01]] [2]
----------
[[5.58856269e-03 9.94401515e-01 9.98589985e-06]
 [8.45920742e-01 1.34242758e-01 1.98364612e-02]
 [1.05949205e-08 3.19452316e-04 9.99680519e-01]] [1 0 2]
'''



#원래 강의와 다르게 직접 데이터를 입력함.

x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5,], [1,2,5,6,], [1,6,6,6], [1,7,7,7]]
y_data = [[2], [2], [2], [1], [1], [1], [0], [0]]

nb_classes = 3

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.int32, shape=[None, 1])
y_one_hot = tf.one_hot(y, nb_classes) #.one_hot()
y_one_hot = tf.reshape(y_one_hot, [-1, nb_classes]) #.reshape()
#rank N이 rank N+1이 되기 때문에 .one_hot() 이후 .reshape()을 사용한다.
w = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
logits = tf.matmul(x, w) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
cost = tf.reduce_mean(cost_i) #.softmax_cross_entropy_with_logits()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#add values.
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={x: x_data, y: y_data})
    if step % 200 == 0:
      #print("Step:", step, "Cost:", cost_val, "Acc:", acc_val)
      print("Step: {:5} Cost: {:.3f} Acc: {:.2%}".format(step, cost_val, acc_val))
'''
Step:     0 Cost: 6.420 Acc: 37.50%
Step:   200 Cost: 0.543 Acc: 62.50%
Step:   400 Cost: 0.451 Acc: 75.00%
Step:   600 Cost: 0.375 Acc: 87.50%
Step:   800 Cost: 0.302 Acc: 87.50%
Step:  1000 Cost: 0.242 Acc: 100.00%
Step:  1200 Cost: 0.219 Acc: 100.00%
Step:  1400 Cost: 0.200 Acc: 100.00%
Step:  1600 Cost: 0.183 Acc: 100.00%
Step:  1800 Cost: 0.170 Acc: 100.00%
Step:  2000 Cost: 0.158 Acc: 100.00%
'''

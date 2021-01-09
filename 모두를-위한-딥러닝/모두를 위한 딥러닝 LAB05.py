%tensorflow_version 1.x
import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]] #0,1 Encoding

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])
w = tf.Variable(tf.random_normal([2, 1]), name='weight') #.random_normal([n: 들어오는 개수, m:나가는 개수])
b = tf.Variable(tf.random_normal([1]), name='bias') #.random_normal([m: 나가는 개수])
hypothesis = tf.sigmoid((tf.matmul(x, w) + b)) #.sigmoid()

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis)) #.log()
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) #.cast()
accurary = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32)) #.equal()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for step in range(10001):
    cost_val, _ = sess.run([cost, train], feed_dict={x: x_data, y: y_data})
    if step % 200 == 0:
      print(step, cost_val)

  h, c, a = sess.run([hypothesis, predicted, accurary],
                   feed_dict={x: x_data, y: y_data})
  print("\nHypothesis: ", h, "\nCorrect(Y): ", c, "\nAccuracy: ", a)
  
#그냥 실행을 돌릴 경우 nan이 출력되어 0.1을 hypothesis에 곱해주었다.
#이후 실행은 되지만, 제대로 된 결과가 나오지 않음.

#cost에 -를 붙여주면 제대로 된 결과가 나온다. 계산 결과 음수가 나온 것을 다시 양수로 바꿔주는 과정. 오류를 해결하였음.
'''
Hypothesis:  [[0.0377278 ]
 [0.16735658]
 [0.33672744]
 [0.767181  ]
 [0.9302757 ]
 [0.97709996]] 
Correct(Y):  [[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]] 
Accuracy:  1.0
'''

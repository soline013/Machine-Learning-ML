#Broadcasting.
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
(matrix1+matrix2).eval()
'''
array([[5., 5.],
       [5., 5.]], dtype=float32)
'''

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
(matrix1+matrix2).eval()
'''
array([[5., 5.]], dtype=float32)
'''

tf.reduce_mean([1, 2], axis=0).eval()
'''
1
'''



x = [[1., 2.],
     [3., 4.]]

#Reduce_mean.
tf.reduce_mean(x).eval()
'''
2.5
'''

tf.reduce_mean(x, axis=0).eval()
'''
array([2., 3.], dtype=float32)
'''

tf.reduce_mean(x, axis=1).eval()
'''
array([1.5, 3.5], dtype=float32)
'''

tf.reduce_mean(x, axis=-1).eval()
'''
array([1.5, 3.5], dtype=float32)
'''



#Reduce_sum.
tf.reduce_sum(x).eval()
'''
10.0
'''

tf.reduce_sum(x, axis=0).eval()
'''
array([4., 6.], dtype=float32)
'''

tf.reduce_sum(x, axis=-1).eval()
'''
array([3., 7.], dtype=float32)
'''

tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval()
'''
5.0
'''

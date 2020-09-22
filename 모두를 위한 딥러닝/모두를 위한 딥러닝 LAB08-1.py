%tensorflow_version 1.x
import tensorflow as tf

import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

import pprint
tf.set_random_seed(777)

pp = pprint.PrettyPrinter(indent=4) #.PrettyPrinter()
sess = tf.InteractiveSession()



#Simple ID Array and Slicing.
t = np.array([0, 1, 2, 3, 4, 5, 6]) #.array()
pp.pprint(t)
print(t.ndim) #rank
print(t.shape) #shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])
'''
array([0, 1, 2, 3, 4, 5, 6])
1
(7,)
0 1 6
[2 3 4] [4 5]
[0 1] [3 4 5 6]
'''



t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t) #.pprint()
print(t.ndim) # rank
print(t.shape) # shape
'''
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.],
       [ 7.,  8.,  9.],
       [10., 11., 12.]])
2
(4, 3)
'''

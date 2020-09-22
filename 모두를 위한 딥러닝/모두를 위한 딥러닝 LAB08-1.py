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



#Shape, Rank, Axis.
t = tf.constant([1,2,3,4])
tf.shape(t).eval() #.shape(), .eval()
'''
array([4], dtype=int32)
'''

t = tf.constant([[1,2],
                 [3,4]])
tf.shape(t).eval()
'''
array([2, 2], dtype=int32)
'''

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
tf.shape(t).eval()
'''
array([1, 2, 3, 4], dtype=int32)
'''

[ #Axis=0
    [ #Axis=1
        [ #Axis=2
            [1,2,3,4], 
            [5,6,7,8],
            [9,10,11,12] #Axis=3 or -1
        ],
        [
            [13,14,15,16],
            [17,18,19,20], 
            [21,22,23,24]
        ]
    ]
]
'''
[[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]
'''



#Matmul & Multiply.
matrix1 = tf.constant([[1.,2], [3.,4.]])
matrix2 = tf.constant([[1.],[2.]])
print("Metrix 1 shape", matrix1.shape)
print("Metrix 2 shape", matrix2.shape)
tf.matmul(matrix1, matrix2).eval()
'''
Metrix 1 shape (2, 2)
Metrix 2 shape (2, 1)
array([[ 5.],
       [11.]], dtype=float32)
'''

(matrix1 * matrix2).eval()
'''
array([[1., 2.],
       [6., 8.]], dtype=float32)
'''

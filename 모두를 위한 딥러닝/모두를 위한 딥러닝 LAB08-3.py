#Argmax.
x = [[0, 1, 2],
     [2, 1, 0]]
tf.argmax(x, axis=0).eval()
'''
array([1, 0, 0])
'''

tf.argmax(x, axis=1).eval()
'''
array([2, 0])
'''

tf.argmax(x, axis=-1).eval()
'''
array([2, 0])
'''



#Reshape.
t = np.array([[[0, 1, 2], 
               [3, 4, 5]],
              
              [[6, 7, 8], 
               [9, 10, 11]]])
t.shape
'''
(2, 2, 3)
'''

tf.reshape(t, shape=[-1, 3]).eval
'''
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
'''

tf.reshape(t, shape=[-1, 1, 3]).eval()
'''
array([[[ 0,  1,  2]],

       [[ 3,  4,  5]],

       [[ 6,  7,  8]],

       [[ 9, 10, 11]]])
'''

tf.squeeze([[0], [1], [2]]).eval() #.squeeze()
'''
array([0, 1, 2], dtype=int32) #1차원 줄여짐.
'''

tf.expand_dims([0, 1, 2], 1).eval() #.expand_dims()
'''
array([[0],
       [1],
       [2]], dtype=int32) #1차원 늘어남. 숫자 지정.
'''



#One_hot.
tf.one_hot([[0], [1], [2], [0]], depth=3).eval()
'''
array([[[1., 0., 0.]], #[0], 원소 3개.

       [[0., 1., 0.]], #[1], 원소 3개.

       [[0., 0., 1.]], #[2], 원소 3개.

       [[1., 0., 0.]]], dtype=float32) #[0], 원소 3개.
'''

t = tf.one_hot([[0], [1], [2], [0]], depth=3)
tf.reshape(t, shape=[-1, 3]).eval()
'''
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.]], dtype=float32)
'''

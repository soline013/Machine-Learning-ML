#Casting.
tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()
'''
array([1, 2, 3, 4], dtype=int32)
'''

tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval()
'''
array([1, 0, 1, 0], dtype=int32)
'''



#Stack.
x = [1, 4]
y = [2, 5]
z = [3, 6]

tf.stack([x, y, z]).eval() #.stack()
'''
array([[1, 4],
       [2, 5],
       [3, 6]], dtype=int32)
'''

tf.stack([x, y, z], axis=1).eval()
'''
array([[1, 2, 3],
       [4, 5, 6]], dtype=int32) #axis=1일 때.
'''



#Ones_like & Zeros_like.
x = [[0, 1, 2],
     [2, 1, 0]]

tf.ones_like(x).eval() #.ones_like()
'''
array([[1, 1, 1],
       [1, 1, 1]], dtype=int32)
'''

tf.zeros_like(x).eval() #.zeros_like()
'''
array([[0, 0, 0],
       [0, 0, 0]], dtype=int32)
'''



#Zip.
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)
for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
'''
1 4
2 5
3 6
1 4 7
2 5 8
3 6 9
'''

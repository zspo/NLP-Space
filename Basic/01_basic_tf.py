# -*- coding: utf-8 -*-
'''
@Author: songpo.zhang
@Description: 
'''
# -*- coding: utf-8 -*-
'''
@Author: songpo.zhang
@Description: 
'''

import tensorflow as tf

# 
a = tf.constant(1)
b = tf.constant(2)

with tf.Session() as sess:
    print(sess.run(a+b))
    print(sess.run(a-b))
    print(sess.run(a*b))

# 
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print(sess.run(add, feed_dict={a:1, b:2}))
    print(sess.run(mul, feed_dict={a:1, b:2}))

#
matrix_a = tf.constant([[3., 3.]])
matrix_b = tf.constant([[2.], [2.]])
product = tf.matmul(matrix_a, matrix_b)

with tf.Session() as sess:
    print(sess.run(product))
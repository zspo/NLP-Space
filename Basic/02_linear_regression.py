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

import random
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

LEARNING_RATE = 0.01
EPOCHS = 50
DISPLAY_STEP = 2

# prepare train datasets
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# tf graph input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# set weights
W = tf.Variable(np.random.randn(), name='W')
b = tf.Variable(np.random.randn(), name='b')

# 
pred = tf.add(tf.multiply(X, W), b)

# mse
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)

# train_op
train_op = optimizer.minimize(cost)

# initialize
init = tf.global_variables_initializer()

fig, ax = plt.subplots()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(EPOCHS):
        for (x, y) in zip(train_X, train_Y):
            sess.run(train_op, feed_dict={X: x, Y: y})

        if (epoch+1) % DISPLAY_STEP == 0:
            _, loss = sess.run([train_op, cost], feed_dict={X: train_X, Y: train_Y})
            print('Epoch: {}\tloss={:.2f}'.format(epoch, loss))

        # Graphic display
        ax.cla()   # 清除键
        ax.plot(train_X, train_Y, 'ro', label='Original data')
        ax.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
        ax.legend()
        # plt.show()
        plt.pause(0.02)

        # time.sleep(1)

    print('train finish')
    train_loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print('Train_Loss={:.2f}'.format(loss)) 

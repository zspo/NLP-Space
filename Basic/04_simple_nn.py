# -*- coding: utf-8 -*-
'''
@Author: songpo.zhang
@Description: 
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/', one_hot=True)

LEARNING_RATE = 0.01
EPOCHS = 500
BATCH_SIZE = 512
DISPLAY_STEP = 10

n_hidden_1 = 256
n_hidden_2 = 256
num_inputs = 784
num_class = 10

X = tf.placeholder(tf.float32, [None, num_inputs])
Y = tf.placeholder(tf.float32, [None, num_class])

weights = {'w1': tf.Variable(tf.random_normal([num_inputs, n_hidden_1])),
           'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
           'out': tf.Variable(tf.random_normal([n_hidden_2, num_class]))
           }

bias = {'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_class]))
        }

def neural_net(x):
    layer1 = tf.add(tf.matmul(x, weights['w1']), bias['b1'])
    layer2 = tf.add(tf.matmul(layer1, weights['w2']), bias['b2'])
    out_put = tf.add(tf.matmul(layer2, weights['out']), bias['out'])
    return out_put

logits = neural_net(X)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(EPOCHS):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_op, feed_dict={X: batch_xs, Y: batch_ys})

        if epoch % DISPLAY_STEP == 0:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_xs, Y: batch_ys})
            print("Epoch: {},\tTrain loss: {:.4f},\tTrain accuracy: {:.4f}".format(epoch, loss, acc))

    print('train finish !')
    print('Test Accuracy: {:.4f}'.format(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})))

# -*- coding: utf-8 -*-

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/', one_hot=True)

LEARNING_RATE = 0.01
EPOCHS = 500
BATCH_SIZE = 512
DISPLAY_STEP = 10

num_inputs = 28
time_steps = 28
num_hidden = 128
num_classes = 10

X = tf.placeholder(tf.float32, [None, time_steps, num_inputs])
Y = tf.placeholder(tf.float32, [None, num_classes])

W = tf.get_variable(name='W', initializer=tf.random_normal([num_hidden, num_classes]))
b = tf.get_variable(name='b', initializer=tf.random_normal([num_classes]))

def RNN(x):
    x = tf.unstack(x, time_steps, 1)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], W) + b

logits = RNN(X) 
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_op)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(EPOCHS):
        for step in range(100):
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            batch_x = batch_x.reshape((BATCH_SIZE, time_steps, num_inputs))
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            
            if step % DISPLAY_STEP == 0:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={
                                     X: batch_x, Y: batch_y})
                print('Epoch: {}\tstep: {}\ttrain loss: {:.4f}\ttrain accuracy: {:.4f}'.format(
                    epoch, step, loss, acc))

        test_loss, test_acc = sess.run([loss_op, accuracy], feed_dict={
                                       X: mnist.test.images.reshape((-1, time_steps, num_inputs)), Y: mnist.test.labels})
        print('Epoch: {}\ttest loss: {:.4f}\ttest accuracy: {:.4f}\n'.format(
            epoch, test_loss, test_acc))

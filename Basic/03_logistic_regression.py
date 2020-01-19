# -*- coding: utf-8 -*-
'''
@Author: songpo.zhang
@Description: 
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/', one_hot=True)

LEARNING_RATA = 0.01
EPOCHS = 50
BATCH_SIZE = 512
DISPALY_STEP = 2

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATA)

train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(EPOCHS):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / BATCH_SIZE)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss = sess.run([train_op, cost], feed_dict={X: batch_xs, Y: batch_ys})

            avg_cost += loss / total_batch
        
        if (epoch+1) % DISPALY_STEP == 0:
            print('Epoch: {}\tloss={:.2f}'.format(epoch, avg_cost))
    
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

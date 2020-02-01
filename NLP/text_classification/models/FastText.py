# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

class FastText(object):
    def __init__(self, num_classes, sequence_length,
                 w2v_model_embedding, vocab_size, embedding_size,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 l2_reg_lambda=0.001):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.w2v_model_embedding = tf.cast(w2v_model_embedding, tf.float32)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initializer = initializer
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_loss = tf.constant(0.0)

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name='label')

        self._instantiate_weights()
        self.logits = self._inference()
        self.loss = self._loss()
        self.accuracy = self._accuracy()

    def _instantiate_weights(self):
        with tf.name_scope('weights'):
            if self.w2v_model_embedding is None:
                self.Embedding = tf.get_variable(name='embedding',
                                                 shape=[self.vocab_size, self.embedding_size],
                                                 initializer=self.initializer)  # [vocab_size, embedding_size]
            else:
                self.Embedding = tf.get_variable(name='embedding',
                                                 initializer=self.w2v_model_embedding,
                                                 dtype=tf.float32)
            
            self.W = tf.get_variable(name='W',
                                     shape=[self.embedding_size, self.num_classes],
                                     initializer=self.initializer)
            self.b = tf.get_variable(name='b', shape=[self.num_classes])

    def _inference(self):
        sentence_embedding = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        self.sentence_embedding = tf.reduce_mean(sentence_embedding, axis=1) # [None, self.embedding_size]

        logits = tf.matmul(self.sentence_embedding, self.W) + self.b
        return logits

    def _loss(self):
        with tf.name_scope('loss'):
            self.l2_loss += tf.nn.l2_loss(self.b)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
        return loss

    def _accuracy(self):
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
        return accuracy

    def compile(self, optimizer, loss, metrics=None):
        if loss == 'binary_crossentropy':
            loss = self.loss
        grads_and_vars = optimizer.compute_gradients(loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars)

    def _next_batch(self, train_x, train_y=None, epochs=1, batch_size=None, shuffle=True):
        data_size = len(train_x)
        num_batches_per_epoch = int(data_size / batch_size) + 1
        
        for epoch in range(epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = train_x[shuffle_indices]
                if train_y is not None:
                    shuffled_data_y = train_y[shuffle_indices]
            else:
                shuffled_data = train_x
                if train_y is not None:
                    shuffled_data_y = train_y

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                
                if train_y is None:
                    yield shuffled_data[start_index:end_index]
                else:
                    yield shuffled_data[start_index:end_index], shuffled_data_y[start_index:end_index]

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, valid_x=None, valid_y=None, checkpoint_dir=None):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # with tf.Session(config=config) as self.sess:  
        if checkpoint_dir:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")

            if os.path.exists(checkpoint_dir):
                print("Restoring Variables from Checkpoint.")
                saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))
            else:
                print('Initializing Variables')
                self.sess.run(tf.global_variables_initializer())
                os.makedirs(checkpoint_dir)
        else:
            self.sess.run(tf.global_variables_initializer())

        train_step = 0
        for epoch in range(epochs):
            step = 0
            for batch_x, batch_y in self._next_batch(x, y, batch_size=batch_size):
                feed_dict = {self.input_x: batch_x,
                             self.input_y: batch_y,
                             }
                self.sess.run([self.loss, self.accuracy, self.train_op], feed_dict)
                train_step += 1
                step += 1

                if step % verbose == 0:
                    feed_dict = {self.input_x: batch_x,
                                 self.input_y: batch_y,
                                 }
                    train_loss, train_acc = self.sess.run([self.loss, self.accuracy], feed_dict)

                    if valid_x is not None:
                        feed_dict = {self.input_x: valid_x,
                                     self.input_y: valid_y,
                                     }
                        val_loss, val_acc = self.sess.run([self.loss, self.accuracy], feed_dict)
                        print('Epoch {}\tBatch {}\tTrain Loss:{:.4f}\tTrain Acc:{:.4f}\tValid Loss:{:.4f}\tValid Acc:{:.4f}'.format(
                            epoch, step, train_loss, train_acc, val_loss, val_acc))
                    else:
                        print('Epoch {}\tBatch {}\tTrain Loss:{:.4f}\tTrain Acc:{:.4f}'.format(epoch, step, train_loss, train_acc))

                if checkpoint_dir:
                    if train_step % 50 == 0:
                        print("Going to save model..")
                        saver.save(self.sess, checkpoint_prefix, global_step=train_step)

    def predict(self, x, batch_size=None, verbose=0, checkpoint_dir=None):
        predict_scores = []
        if not checkpoint_dir:
            sess = self.sess
        else:
            print('Restore model from checkpoint.')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            saver = tf.train.Saver()
            checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        if batch_size is None:
            predict_scores = sess.run(self.logits, feed_dict={self.input_x: x})
        else:
            for batch_x in self._next_batch(x, batch_size=batch_size):
                batch_result = sess.run(self.logits, feed_dict={self.input_x: batch_x})
                predict_scores += batch_result.tolist()
        
        return np.array(predict_scores)


# -*- coding: utf-8 -*-

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

        self.instantiate_weights()
        self.logits = self.inference()
        self.loss = self.loss()
        self.accuracy = self.accuracy()

    def instantiate_weights(self):
        with tf.name_scope('weights'):
            if self.w2v_model_embedding is None:
                self.Embedding = tf.get_variable(name='embedding',
                                                 shape=[self.vocab_size,self.embedding_size],
                                                 initializer=self.initializer)  # [vocab_size, embedding_size]
            else:
                self.Embedding = tf.get_variable(name='embedding',
                                                 initializer=self.w2v_model_embedding,
                                                 dtype=tf.float32)
            
            self.W = tf.get_variable(name='W',
                                     shape=[self.embedding_size, self.num_classes],
                                     initializer=self.initializer)
            self.b = tf.get_variable(name='b', shape=[self.num_classes])
    def inference(self):
        sentence_embedding = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        self.sentence_embedding = tf.reduce_mean(sentence_embedding, axis=1) # [None, self.embedding_size]

        logits = tf.matmul(self.sentence_embedding, self.W) + self.b
        return logits

    def loss(self):
        with tf.name_scope('loss'):
            self.l2_loss += tf.nn.l2_loss(self.b)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
        return loss

    def accuracy(self):
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
        return accuracy


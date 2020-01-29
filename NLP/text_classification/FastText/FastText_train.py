# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('..')
import time
import pickle
import numpy as np 
import pandas as pd 
import tensorflow as tf

from FastText import FastText
from Utils import data_helper

FLAGS = tf.app.flags.FLAGS
# Data params
tf.app.flags.DEFINE_string('data_path', '../text_data/input_data/', 'input data path')
# Model params
tf.app.flags.DEFINE_integer("num_classes", 2, "num_classes")
# Training params
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning_rate (default: 0.01)")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of training epochs (default: 10)")
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("checkpoint_every", 100, "Save model every steps (default: 100)")
tf.app.flags.DEFINE_string("checkpoint_dir", './model_save/', "checkpoint_dir")
tf.app.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")


FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def main(_):
    train_x, train_y, valid_x, valid_y, embedding, word2index, index2word, vocab_size, maxlen = data_helper.load_data(FLAGS.data_path)
    print(train_x.shape)
    print(vocab_size)
    print(embedding.shape)
    print(embedding.dtype)
    print(maxlen)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = FastText( 
                        num_classes=FLAGS.num_classes, 
                        sequence_length=maxlen,
                        w2v_model_embedding=embedding,
                        vocab_size=vocab_size,
                        embedding_size=200)

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars)

        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=FLAGS.num_checkpoints)
        checkpoint_dir = os.path.abspath(os.path.join(FLAGS.checkpoint_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        if os.path.exists(checkpoint_dir):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            os.makedirs(checkpoint_dir)

        train_step = 0
        for epoch in range(FLAGS.epochs):
            step = 0
            for batch_x, batch_y in data_helper.next_batch(train_x, train_y, FLAGS.batch_size):
                feet_dict = {model.input_x: batch_x,
                             model.input_y: batch_y,
                            }
                train_loss, train_acc, _ = sess.run([model.loss, model.accuracy, train_op], feet_dict)
                train_step += 1
                step += 1

                feet_dict = {model.input_x: valid_x,
                             model.input_y: valid_y,
                            }
                val_loss, val_acc = sess.run([model.loss, model.accuracy], feet_dict)
                print('Epoch {}\tBatch {}\tTrain Loss:{:.4f}\tTrain Acc:{:.4f}\tValid Loss:{:.4f}\tValid Acc:{:.4f}'.format(
                    epoch, step, train_loss, train_acc, val_loss, val_acc))

                if train_step % FLAGS.checkpoint_every == 0:
                    print("Going to save model..")
                    saver.save(sess, checkpoint_prefix, global_step=train_step)

if __name__ == "__main__":
    tf.app.run()

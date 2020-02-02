# -*- coding: utf-8 -*-

import os
import sys
import tensorflow as tf
from tensorflow.python.framework import graph_util
 

def freeze_graph(ckpt_model_dir, export_path_base, model_version):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    checkpoint = tf.train.get_checkpoint_state(ckpt_model_dir)#检查目录下ckpt文件状态是否可用
    if not checkpoint:
        print('dir not')
        exit()
    input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    export_path = os.path.join(tf.compat.as_bytes(export_path_base),
                               tf.compat.as_bytes(str(model_version)))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
 
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    # input_graph_def = tf.get_default_graph().as_graph_def()
    # node_names = [n.name for n in input_graph_def.node]
    # for node in node_names:
    #     print(node)
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        input_x = sess.graph.get_tensor_by_name('input_x:0')
        output = sess.graph.get_tensor_by_name('output/add:0')

        tensor_info_x = tf.saved_model.utils.build_tensor_info(input_x) # 输入
        tensor_info_y = tf.saved_model.utils.build_tensor_info(output) # 输出

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs={'x': tensor_info_x},
                                                                                      outputs={'y': tensor_info_y},
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                                                                                      )

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(sess, 
                                             [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={'predictions': prediction_signature},
                                             legacy_init_op=legacy_init_op)

    builder.save()

    print('Done exporting!') 
 
input_checkpoint='text_classification/examples/model_save/checkpoints/'
out_pb_path='text_classification/online/save_model/'
freeze_graph(input_checkpoint, out_pb_path, 1)
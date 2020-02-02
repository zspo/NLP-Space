# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
from tensorflow.python.framework import graph_util
  
def freeze_graph(ckpt_model_dir, output_graph):
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
 
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    # 直接用最后输出的节点，可以在tensorboard中查找到，tensorboard只能在linux中使用
    output_node_names = "output/add"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() # 获得默认的图
    
    input_graph_def = tf.get_default_graph().as_graph_def()

    node_names = [n.name for n  in input_graph_def.node]
    for node in node_names:
        print(node)
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(sess=sess, # 模型持久化，将变量值固定
                                                                     input_graph_def=input_graph_def, # 等于:sess.graph_def
                                                                     output_node_names=output_node_names.split(",")) # 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点
 
input_checkpoint='text_classification/examples/model_save/checkpoints/'
out_pb_path='text_classification/online/pb_model/'
freeze_graph(input_checkpoint, out_pb_path)
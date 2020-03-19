#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
用于训练语音识别系统语音模型的程序

"""
import platform as plat
import os

import tensorflow as tf

from SpeechModel24 import ModelSpeech

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel("ERROR")

datapath = ''
modelpath = 'model_speech'

if(not os.path.exists(modelpath)):  # 判断保存模型的目录是否存在
    os.makedirs(modelpath)  # 如果不存在，就新建一个，避免之后保存模型的时候炸掉

system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
if(system_type == 'Windows'):
    datapath = 'E:\\语音数据集'
    modelpath = modelpath + '\\'
elif(system_type == 'Linux'):
    datapath = 'dataset'
    modelpath = modelpath + '/'
else:
    print('*[Message] Unknown System\n')
    datapath = 'dataset'
    modelpath = modelpath + '/'

if 'COLAB_TPU_ADDR' in os.environ.keys():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    with strategy.scope():
        ms = ModelSpeech(datapath)
else:
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)]
        )
    ms = ModelSpeech(datapath)

# ms.LoadModel(modelpath + 'speech_model251_e_0_step_327500.model')
ms.TrainModel(datapath, epoch = 50, batch_size = 4, save_step = 500)



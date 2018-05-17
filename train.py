from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import keras
from skimage import io
import os

from model.base import networks
from model.losses import *
from model.augmented_data import data_generator
import config

# list of gpu ids (at least 4 GPUs)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids

batch_size = config.batch_size
steps_per_epoch = config.steps_per_epoch
epochs = config.epochs
learning_rate = config.learning_rate
delete_train = config.delete_train

# remove logs files
if delete_train:
    path_logs = './logs/*'
    print('rm -Rf' + path_logs)
    os.system('rm -Rf ' + path_logs)

    path_logs = './checkpoint/*'
    print('rm -Rf' + path_logs)
    os.system('rm -Rf ' + path_logs)

# input image dimensions
img_x, img_y, img_z = 256, 256, 30

# train values range from 0-255
target_train = io.imread('fixed.tif')
source_train = io.imread('moving.tif')

# Validation
target_val = io.imread('fixed.tif')
source_val = io.imread('moving.tif')

target_val = np.reshape(target_val/255.0, (1, img_x, img_y, img_z, 1)).astype('float32')
source_val = np.reshape(source_val/255.0, (1, img_x, img_y, img_z, 1)).astype('float32')
zero_flow_val = np.zeros((1, img_x, img_y, img_z, 3))

###########################################################################################
# input shape
input_shape = (img_x, img_y, img_z, 1)

model = networks(input_shape)

model.compile(loss={'moved_source': cross_corelation_3D(),
                    'flow_source': total_variation_3D(),
                    'moved_target': cross_corelation_3D(),
                    'flow_target': total_variation_3D()},
              loss_weights={'moved_source': 1.0, 'flow_source': 0.1,
                            'moved_target': 1.0, 'flow_target': 0.1},
              optimizer=keras.optimizers.Adam(lr=learning_rate))

# callback functions
cb_tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                            batch_size=batch_size, write_graph=True,
                            write_grads=False, write_images=False)

cb_temp_cpkt = keras.callbacks.ModelCheckpoint('./checkpoint/weights.{epoch:02d}-{val_loss:.5f}.h5', monitor='val_loss', verbose=1,
                                               save_best_only=True, save_weights_only=False,
                                               mode='min', period=20)
# data generator
generator = data_generator

with tf.device('/gpu:4'):
    model.fit_generator(generator(target_train, source_train, batch_size, input_shape, isAugmentation=True),
              steps_per_epoch=steps_per_epoch,
              epochs=epochs,
              verbose=1, use_multiprocessing=False, callbacks=[cb_tensorboard, cb_temp_cpkt],
              validation_data=({'input_target': np.array(target_val),
                               'input_source': np.array(source_val)},
                              {'moved_source': np.array(target_val),
                               'flow_source': np.array(zero_flow_val),
                               'moved_target': np.array(source_val),
                               'flow_target': np.array(zero_flow_val)})
              )
model.save('./checkpoint/my_model.h5')

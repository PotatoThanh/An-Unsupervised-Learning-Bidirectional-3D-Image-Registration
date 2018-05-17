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
delete_test = config.delete_train


# remove logs files
if delete_test:
    path_logs = './logs/*'
    print('rm -Rf' + path_logs)
    os.system('rm -Rf ' + path_logs)

# input image dimensions
img_x, img_y, img_z = 256, 256, 30

# Image3D values range from 0-255
target_test = io.imread('fixed.tif')
source_test = io.imread('moving.tif')

target_test = np.reshape(target_test/255.0, (1, img_x, img_y, img_z, 1)).astype('float32')
source_test = np.reshape(source_test/255.0, (1, img_x, img_y, img_z, 1)).astype('float32')
zero_flow_test = np.zeros((1, img_x, img_y, img_z, 3))

#####################################################################

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

ckpt = 'ckpt name'
model.load_weights(ckpt)

eval_test = model.evaluate({'input_target': np.array(target_test),
               'input_source': np.array(source_test)},
              {'moved_source': np.array(target_test),
               'flow_source': np.array(zero_flow_test),
               'moved_target': np.array(source_test),
               'flow_target': np.array(zero_flow_test)}, verbose=1)

print(eval_test)

# predict
moved_source, flow_source, moved_target, flow_target = model.predict({'input_target': np.array(target_test),
                                                                      'input_source': np.array(source_test)},
                                                                     verbose=1)

# results
moved_source = np.array(255.0*moved_source).astype(np.uint8)
moved_target = np.array(255.0*moved_target).astype(np.uint8)
flow_source = np.array(255.0*abs(flow_source[:,:,:, :, 0])).astype(np.uint8)
flow_target = np.array(255.0*abs(flow_target[:,:,:, :, 0])).astype(np.uint8)

# save images
io.imsave('./deploy/moved_source.tif', np.reshape(moved_source, (img_z, img_x, img_y)))
io.imsave('./deploy/moved_target.tif', np.reshape(moved_target, (img_z, img_x, img_y)))
io.imsave('./deploy/flow_source.tif', np.reshape(flow_source, (img_z, img_x, img_y)))
io.imsave('./deploy/flow_target.tif', np.reshape(flow_target, (img_z, img_x, img_y)))
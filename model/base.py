from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
from model.spatial_transformer_3D import Dense3DSpatialTransformer


def networks(input_shape):
    input_target = Input(shape=input_shape, name='input_target')
    input_source = Input(shape=input_shape, name='input_source')

    inputs = concatenate([input_source, input_target])

    # Encoder
    with tf.device('/gpu:0'):
        with tf.name_scope('Encoder'):
            en = my_Conv3D(inputs, 16)

            en1 = my_Conv3D(en, 32, strides=(2, 2, 2))

            en2 = my_Conv3D(en1, 32, strides=(2, 2, 3))

            en3 = my_Conv3D(en2, 64, strides=(2, 2, 1))

            en4 = my_Conv3D(en3, 64, strides=(2, 2, 1))

            en5 = my_Conv3D(en4, 128, strides=(2, 2, 1))

    # Decoder1 (source->target)
    with tf.device('/gpu:1'):
        with tf.name_scope('Forward'):
            de11 = my_Conv3D(en5, 64)
            de11 = UpSampling3D((2, 2, 1))(de11)
            de11 = concatenate([de11, en4])

            de12 = my_Conv3D(de11, 64)
            de12 = UpSampling3D((2, 2, 1))(de12)
            de12 = concatenate([de12, en3])

            de13 = my_Conv3D(de12, 32)
            de13 = UpSampling3D((2, 2, 1))(de13)
            de13 = concatenate([de13, en2])

            de14 = my_Conv3D(de13, 32)
            de14 = UpSampling3D((2, 2, 3))(de14)
            de14 = concatenate([de14, en1])

            de15 = my_Conv3D(de14, 16)
            de15 = UpSampling3D((2, 2, 2))(de15)
            de15 = concatenate([de15, en])

            de15 = my_Conv3D(de15, 16)

            flow_source = Conv3D(3, (3, 3, 3), padding='same', name='flow_source')(de15)
            moved_source = Dense3DSpatialTransformer(name='moved_source')([input_source, flow_source])

    # Decoder2 (target->source)
    with tf.device('/gpu:2'):
        with tf.name_scope('Backward'):
            de21 = my_Conv3D(en5, 64)
            de21 = UpSampling3D((2, 2, 1))(de21)
            de21 = concatenate([de21, en4])

            de22 = my_Conv3D(de21, 64)
            de22 = UpSampling3D((2, 2, 1))(de22)
            de22 = concatenate([de22, en3])

            de23 = my_Conv3D(de22, 32)
            de23 = UpSampling3D((2, 2, 1))(de23)
            de23 = concatenate([de23, en2])

            de24 = my_Conv3D(de23, 32)
            de24 = UpSampling3D((2, 2, 3))(de24)
            de24 = concatenate([de24, en1])

            de25 = my_Conv3D(de24, 16)
            de25 = UpSampling3D((2, 2, 2))(de25)
            de25 = concatenate([de25, en])

            de25 = my_Conv3D(de25, 16)

            flow_target = Conv3D(3, (3, 3, 3), padding='same', name='flow_target')(de25)
            moved_target = Dense3DSpatialTransformer(name='moved_target')([input_target, flow_target])

    model = Model(inputs=[input_target, input_source],
                  outputs=[moved_source, flow_source, moved_target, flow_target])

    return model

def my_Conv3D(inputs, filters, filter_size = (3, 3, 3), strides=(1, 1, 1), dilation_rate=(1, 1, 1),
              padding='same', batch_normalization='True'):
    outputs = Conv3D(filters=filters, kernel_size=filter_size, strides=strides,
                     dilation_rate=dilation_rate, padding=padding)(inputs)
    if batch_normalization:
        outputs = BatchNormalization()(outputs)

    outputs = LeakyReLU(0.2)(outputs)
    return outputs
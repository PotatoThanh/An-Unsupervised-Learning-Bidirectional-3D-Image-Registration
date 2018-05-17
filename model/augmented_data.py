import numpy as np
from imgaug import augmenters as iaa

def data_generator(target, source, batch_size, img_shape, isAugmentation=False):

    size_x, size_y, size_z, channels = img_shape

    # reshape follows imgaug lib description
    target = np.reshape(target, (size_z, size_x, size_y))
    source = np.reshape(source, (size_z, size_x, size_y))

    zero_flow = np.zeros((batch_size, size_x, size_y, size_z, 3))

    aug = iaa.OneOf([iaa.Dropout(p=(0, 0.2)),
                     iaa.AdditiveGaussianNoise(scale=0.1 * 255),
                     iaa.GaussianBlur(sigma=(0,1.0))])
    while True:

        list_target = np.zeros((batch_size, size_x, size_y, size_z, channels))
        list_source = np.zeros((batch_size, size_x, size_y, size_z, channels))
        
        for i in range(batch_size):
            if isAugmentation:
                aug_target = aug.augment_images(target)
                aug_target = np.reshape(aug_target, (size_x, size_y, size_z, channels)).astype('float32') 

                aug_source = aug.augment_images(source)
                aug_source = np.reshape(aug_source, (size_x, size_y, size_z, channels)).astype('float32')
            else:
                aug_target = np.reshape(target, (size_x, size_y, size_z, channels)).astype('float32')
                aug_source = np.reshape(source, (size_x, size_y, size_z, channels)).astype('float32')
            list_target[i] = aug_target
            list_source[i] = aug_source

        list_target = list_target/255.0
        list_source = list_source/255.0
        

        yield({'input_target': np.array(list_target), 'input_source': np.array(list_source)},
              {'moved_source': np.array(list_target),
               'flow_source': np.array(zero_flow),
               'moved_target': np.array(list_source),
               'flow_target': np.array(zero_flow)})


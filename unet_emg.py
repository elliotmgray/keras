'''
this file will train a keras model using multi-gpu and data generator from directory with sequence.

By Elliot Gray, January 2018
OHSU Computational Biology
adapted from:
1) https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulenet-multi-gpu.py
2) https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
'''

import numpy as np
import tensorflow as tf
import os
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks, optimizers
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout

def dice_coef(y_true, y_pred):

    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):

    return 1. - dice_coef(y_true, y_pred)

# setting image dimensions
img_width, img_height = 150, 150
K.set_image_data_format('channels_last')
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def train(model, args):

    # unpack input args. all dependencies on arg names are in this code chunk.
    seed = args.seed
    debug = args.debug
    train_image_dir = args.train_image_dir
    train_mask_dir = args.train_mask_dir
    val_image_dir = args.validation_image_dir
    val_mask_dir = args.validation_mask_dir
    save_dir = args.save_dir
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    # find out how many files are in each directory specified
    nb_train_samples = len(
        [f for f in os.listdir(train_image_dir) if os.path.isfile(os.path.join(train_image_dir, f))])
    nb_validation_samples = len(
        [f for f in os.listdir(val_image_dir) if os.path.isfile(os.path.join(val_image_dir, f))])
    
    # callbacks
    log = callbacks.CSVLogger(save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=save_dir + '/tensorboard-logs',
                               batch_size=batch_size, histogram_freq=debug)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))

    # compile the model
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=[dice_coef])

    train_data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1)
    train_image_datagen = ImageDataGenerator(**train_data_gen_args)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_args)

    test_data_gen_args = {}
    test_image_datagen = ImageDataGenerator(**test_data_gen_args)
    test_mask_datagen = ImageDataGenerator(**test_data_gen_args)

    train_image_generator = train_image_datagen.flow_from_directory(
        train_image_dir,
        batch_size=batch_size,
        class_mode=None,
        seed=seed)

    train_mask_generator = train_mask_datagen.flow_from_directory(
        train_mask_dir,
        batch_size=batch_size,
        class_mode=None,
        seed=seed)

    train_generator = zip(train_image_generator, train_mask_generator)

    validation_image_generator = test_image_datagen.flow_from_directory(
        val_image_dir,
        batch_size=batch_size,
        class_mode=None,
        seed=seed)

    validation_mask_generator = test_mask_datagen.flow_from_directory(
        val_mask_dir,
        batch_size=batch_size,
        class_mode=None,
        seed=seed)

    validation_generator = zip(validation_image_generator, validation_mask_generator)


    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[log, tb, lr_decay])

    return model


def get_unet(img_rows, img_cols, img_channels):

    inputs = Input((img_rows, img_cols, img_channels))
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=2)(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=2)(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=2)(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=2)(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = concatenate([Conv2DTranspose(512, 2, strides=2, padding='same', kernel_initializer='he_normal')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(256, 2, strides=2, padding='same', kernel_initializer='he_normal')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(128, 2, strides=2, padding='same', kernel_initializer='he_normal')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(64, 2, strides=2, padding='same', kernel_initializer='he_normal')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    unet_model = Model(inputs=[inputs], outputs=[conv10])

    return unet_model

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description="Elliot's deep learning trainer.")

    parser.add_argument('--epochs', default=1, type=int)

    parser.add_argument('--batch_size', default=32, type=int)
    
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")

    parser.add_argument('--debug', default=0, type=int,
                        help="Save weights by TensorBoard")

    parser.add_argument('--save_dir', default='/Users/grael/Desktop/melanoma_epidermis/dev_model')
    parser.add_argument('--train_image_dir', default='/Users/grael/Desktop/melanoma_epidermis/train/image')
    parser.add_argument('--train_label_dir', default='/Users/grael/Desktop/melanoma_epidermis/train/label')
    parser.add_argument('--validation_image_dir', default='/Users/grael/Desktop/melanoma_epidermis/test/image')
    parser.add_argument('--validation_label_dir', default='/Users/grael/Desktop/melanoma_epidermis/test/label')

    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")

    parser.add_argument('--gpus', default=2, type=int)

    parser.add_argument('--seed', default=1, type=int)

    args = parser.parse_args()

    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define model
    with tf.device('/cpu:0'):
        model = get_unet(128, 128, 3)

    model.summary()

    multi_model = multi_gpu_model(model, gpus=args.gpus)

    # offhand note, that the model.compile step happens in the train function.
    train(model=multi_model, args=args)

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

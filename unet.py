import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from unet_data import *
import argparse

def dice_coef(y_true, y_pred):

    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):

    return 1. - dice_coef(y_true, y_pred)


class MyUnet(object):

    def __init__(self, img_rows=512, img_cols=512, img_channels=1,
                 data_path='../train/image',
                 label_path='../train/label',
                 test_path='../test',
                 model_path='../model',
                 npy_path='../npydata',
                 img_type='tif'):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.data_path = data_path
        self.label_path = label_path
        self.test_path = test_path
        self.model_path = model_path
        self.npy_path = npy_path
        self.img_type = img_type

    def load_data(self):

        mydata = dataProcess(self.img_rows, self.img_cols, self.img_channels,
                             data_path=self.data_path,
                             label_path=self.label_path,
                             test_path=self.test_path,
                             npy_path=self.npy_path,
                             img_type=self.img_type)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self):

        inputs = Input((self.img_rows, self.img_cols, self.img_channels))
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

        model = Model(inputs=[inputs], outputs=[conv10])

        # model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=[dice_coef])

        return model

    def train(self):

        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")

        model_checkpoint = ModelCheckpoint(self.model_path + '/unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=10, verbose=1, validation_split=0.2, shuffle=True,
                  callbacks=[model_checkpoint])



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Unet with Keras.')

    parser.add_argument('loo_num', type=int, default=0,
                        help='Image Dir to leave out.')
    args = parser.parse_args()

    loo_imgs = [
        '70901-64576',
        '71001-64408',
        '71288-64806',
        '71357-64285',
        '71446-63955'
    ]

    loo_img = '/loo_' + loo_imgs[args.loo_num]

    myunet = MyUnet(256,
                    256,
                    3,
                    data_path="/home/users/grael/pydeep/melanoma_epidermis/train/image" + loo_img,
                    label_path="/home/users/grael/pydeep/melanoma_epidermis/train/label" + loo_img,
                    test_path="/home/users/grael/pydeep/melanoma_epidermis/test/image" + loo_img,
                    model_path="/home/users/grael/pydeep/melanoma_epidermis/model" + loo_img,
                    npy_path="/home/users/grael/pydeep/melanoma_epidermis/npydata" + loo_img,
                    img_type='tif')
    myunet.train()

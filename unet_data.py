from keras.preprocessing.image import array_to_img, img_to_array, load_img
import os
import numpy as np
import glob


class dataProcess(object):

    def __init__(self, out_rows, out_cols, out_channels, data_path="../deform/train", label_path="../deform/label",
                 test_path="../test", npy_path="../npydata", img_type="tif"):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.out_channels = out_channels
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def create_train_data(self):

        i = 0
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)
        imgs = glob.glob(self.data_path + "/*." + self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, self.out_channels), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/") + 1:]
            img = load_img(self.data_path + "/" + midname, grayscale=False)
            label = load_img(self.label_path + "/" + midname, grayscale=True)
            img = img_to_array(img)
            label = img_to_array(label)
            imgdatas[i] = img
            imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def create_test_data(self):

        i = 0
        print('-' * 30)
        print('Creating test images...')
        print('-' * 30)
        imgs = glob.glob(self.test_path + "/*." + self.img_type)
        imgs.sort()
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, self.out_channels), dtype=np.uint8)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/") + 1:]
            img = load_img(self.test_path + "/" + midname, grayscale=False)
            img = img_to_array(img)
            imgdatas[i] = img
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

    def load_train_data(self):

        print('-' * 30)
        print('load train images...')
        print('-' * 30)
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        return imgs_train, imgs_mask_train

    def load_test_data(self):

        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test


if __name__ == "__main__":

    loo_imgs = [
        '70901-64576',
        '71001-64408',
        '71288-64806',
        '71357-64285',
        '71446-63955'
    ]

    for i in range(len(loo_imgs)):
        loo_img = '/loo_' + loo_imgs[i]

        mydata = dataProcess(256, 256, 3,
                             data_path="/home/users/grael/pydeep/melanoma_epidermis/train/image" + loo_img,
                             label_path="/home/users/grael/pydeep/melanoma_epidermis/train/label" + loo_img,
                             test_path="/home/users/grael/pydeep/melanoma_epidermis/test/image" + loo_img,
                             npy_path="/home/users/grael/pydeep/melanoma_epidermis/npydata" + loo_img,
                             img_type="tif"
                             )
        mydata.create_train_data()
        mydata.create_test_data()

from unet import *
from unet_data import *
from skimage.io import imsave, imread

"""In pre-processing, pixel intensities are scaled to [0, 1]."""

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
                         img_type="tif")

    imgs_test = mydata.load_test_data()

    myunet = MyUnet(256,
                    256,
                    3,
                    data_path="/home/users/grael/pydeep/melanoma_epidermis/train/image" + loo_img,
                    label_path="/home/users/grael/pydeep/melanoma_epidermis/train/label" + loo_img,
                    test_path="/home/users/grael/pydeep/melanoma_epidermis/test/image" + loo_img,
                    model_path="/home/users/grael/pydeep/melanoma_epidermis/model" + loo_img,
                    npy_path="/home/users/grael/pydeep/melanoma_epidermis/npydata" + loo_img,
                    img_type='tif')

    model = myunet.get_unet()

    model.load_weights(myunet.model_path + '/unet.hdf5')

    imgs_mask_test = model.predict(imgs_test, verbose=1)

    for i in range(imgs_mask_test.shape[0]):
        img = imgs_mask_test[i]
        img = array_to_img(img)
        img.save("/home/users/grael/pydeep/melanoma_epidermis/test/result" + loo_img + "/%d.tif" % (i))

    '''
    for i, image_mask in enumerate(imgs_mask_test):
        imsave("/Users/grael/Desktop/melanoma_epidermis/test/result/" + str(i) + ".tif", image_mask)
    '''

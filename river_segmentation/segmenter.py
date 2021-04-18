# import os
# import cv2
# import numpy as np
# import keras
# from segmentation_models import Unet
# from segmentation_models import get_preprocessing
# from segmentation_models.losses import bce_jaccard_loss
# from segmentation_models.metrics import iou_score
# from keras.models import load_model

from tensorflow import keras


class SegmentationMasks:

    def __init__(self, path):
        self.model = load_model(path, compile=False)
        self.model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

    def load_preprocessing(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        height = 256
        width = 256
        dim = (width, height)
        image = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        image = image.astype('float32')
        means = image.mean(axis=(0, 1), dtype='float64')
        image -= means
        mins = np.min(image, axis=(0, 1))
        maxs = np.max(image, axis=(0, 1))
        image = (image - mins) / (maxs - mins)
        img_min = np.min(img, axis=(0, 1))
        img_max = np.max(img, axis=(0, 1))
        img = (img - img_min) / (img_max - img_min)

        return image, img

    def preprocessing(self, img):
        height = 256
        width = 256
        dim = (width, height)
        image = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        image = image.astype('float32')
        means = image.mean(axis=(0, 1), dtype='float64')
        image -= means
        mins = np.min(image, axis=(0, 1))
        maxs = np.max(image, axis=(0, 1))
        image = (image - mins) / (maxs - mins)
        img_min = np.min(img, axis=(0, 1))
        img_max = np.max(img, axis=(0, 1))
        img = (img - img_min) / (img_max - img_min)

        return image, img

    def mask(self, image_path):
        test_image, original_image = self.preprocessing(image_path)
        test = self.model.predict(test_image.reshape(1, 256, 256, 3))
        test = self.test.reshape(256, 256, 2)[:, :, 0].round()
        test = cv2.resize(test, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return test

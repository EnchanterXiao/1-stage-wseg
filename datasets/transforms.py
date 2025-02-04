import random
import torch
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as tf
import torchvision.transforms.functional as F

from functools import partial

class Compose:
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label, score):
        # allow for intermediate representations
        result = (image, label, score)
        for t in self.segtransform:
            result = t(*result)

        # ensure we have just the image
        # and the label in the end
        image, label, score = result
        return image, label, score

class MaskRandResizedCrop:

    def __init__(self, cfg):
        self.rnd_crop = tf.RandomResizedCrop(cfg.CROP_SIZE, \
                                             scale=(cfg.SCALE_FROM, \
                                                    cfg.SCALE_TO))

    def get_params(self, image):
        return self.rnd_crop.get_params(image, \
                                        self.rnd_crop.scale, \
                                        self.rnd_crop.ratio)

    def __call__(self, image, labels, scores):

        i, j, h, w = self.get_params(image)

        image = F.resized_crop(image, i, j, h, w, self.rnd_crop.size, Image.CUBIC)
        labels = F.resized_crop(labels, i, j, h, w, self.rnd_crop.size, Image.NEAREST)
        scores = F.resized_crop(scores, i, j, h, w, self.rnd_crop.size, Image.BILINEAR)

        return image, labels, scores

class MaskHFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, score):

        if random.random() < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)
            score = F.hflip(score)

        return image, mask, score

class MaskNormalise:

    def __init__(self, mean, std):
        self.norm = tf.Normalize(mean, std)

    def __toByteTensor(self, pic):
        return torch.from_numpy(np.array(pic, np.int32, copy=False))

    def __call__(self, image, labels, score):

        image = F.to_tensor(image)
        image = self.norm(image)
        labels = self.__toByteTensor(labels)
        # print('Norm:', np.array(score).shape)
        # print(np.all(np.array(score)[:, :, 0] == np.array(score)[:, :, 1]))
        # print(np.all(np.array(score)[:, :, 0] == np.array(score)[:, :, 2]))
        score = torch.from_numpy(np.array(score))
        # print('Norm:', labels.size())
        # print('Norm:', score.size())


        return image, labels, score

class MaskToTensor:

    def __call__(self, image, mask):
        gt_labels = torch.arange(0, 21)
        gt_labels = gt_labels.unsqueeze(-1).unsqueeze(-1)
        mask = mask.unsqueeze(0).type_as(gt_labels)
        mask = torch.eq(mask, gt_labels).int()
        return image, mask

class MaskColourJitter:

    def __init__(self, p=0.5, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
        self.p = p
        self.jitter = tf.ColorJitter(brightness=0.3, \
                                     contrast=0.3, \
                                     saturation=0.3, \
                                     hue=0.1)

    def __call__(self, image, mask, score):

        if random.random() < self.p:
            image = self.jitter(image)
        return image, mask, score

class RandomGaussianBlur(object):
    def __call__(self, image, labels, score):
        img = image
        mask = labels
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask, score

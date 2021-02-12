import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import cv2


def tensor_load_rgbimage(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    print(img.shape)
    tensor = torch.from_numpy(img).float()

    return tensor


def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = np.transpose(img, (0, 2, 3, 1))[0]
    cv2.imwrite("output/thing_and_stuff.png", img)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    channels = torch.chunk(tensor, 3)
    b, g, r = channels
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def subtract_imagenet_mean_batch(batch):
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))
    return batch


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch



if __name__ == '__main__':
    img = np.load("thing.npy")
    img = np.transpose(img, (0, 2, 3, 1))[0]
    cv2.imwrite("output/thing_and_stuff.png", img)

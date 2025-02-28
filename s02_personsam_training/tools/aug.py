import random
import torch
from torchvision import transforms
import torchvision.transforms.functional as tf

def random_rotate(image, mask, angel_range = 180):
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    angle = transforms.RandomRotation.get_params([-angel_range, angel_range])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    image = image.rotate(angle)
    mask = mask.rotate(angle)
    return image, mask

def random_h_flip(image, mask, prob_threshold=0.5):
    # 50%的概率应用垂直，水平翻转。
    if random.random() > prob_threshold:
        image = tf.hflip(image)
        mask = tf.hflip(mask)
    return image, mask

def random_v_flip(image, mask, prob_threshold=0.5):
    # 50%的概率应用垂直，垂直翻转。
    if random.random() > prob_threshold:
        image = tf.vflip(image)
        mask = tf.vflip(mask)
    return image, mask

def random_crop(image, mask, size = (384, 128)):
    # 随机对mask和image都进行crop增强
    i, j, h, w = transforms.RandomCrop.get_params(image,size)
    image = tf.crop(image, i, j, h, w)
    mask = tf.crop(mask, i, j, h, w)
    return image, mask

def random_erasing(image, mask,inplace=False, prob=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=[0.48145466, 0.4578275, 0.40821073]):
    # 随机对mask和image都进行crop增强
    if torch.rand(1) < prob:
        x, y, h, w, v = transforms.RandomErasing.get_params(tf.to_tensor(image), scale=scale, ratio=ratio, value=value)
        image = tf.erase(tf.to_tensor(image), x, y, h, w, v, inplace)
        mask = tf.erase(tf.to_tensor(mask), x, y, h, w, torch.zeros([1,1,1]), inplace)
    return image, mask


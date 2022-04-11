import torch
import numpy as np


def sort_file(file_path):
    f2 = open(file_path, "r")
    lines = f2.readlines()
    ret = []
    for line in lines:
        line = line[:-1]
        ret.append(line)
    ret.sort()

    with open('./output.txt', 'w') as f:
        for i in ret:
            f.write(i + '\n')


def five_point_crop(idx, d_img, config):
    new_h = config.crop_size
    new_w = config.crop_size
    b, c, h, w = d_img.shape
    if idx == 0:
        top = 0
        left = 0
    elif idx == 1:
        top = 0
        left = w - new_w
    elif idx == 2:
        top = h - new_h
        left = 0
    elif idx == 3:
        top = h - new_h
        left = w - new_w
    elif idx == 4:
        center_h = h // 2
        center_w = w // 2
        top = center_h - new_h // 2
        left = center_w - new_w // 2
    d_img_org = crop_image(top, left, config.crop_size, img=d_img)

    return d_img_org


def random_crop(d_img, config):
    b, c, h, w = d_img.shape
    top = np.random.randint(0, h - config.crop_size)
    left = np.random.randint(0, w - config.crop_size)
    d_img_org = crop_image(top, left, config.crop_size, img=d_img)
    return d_img_org


def crop_image(top, left, patch_size, img=None):
    tmp_img = img[:, :, top:top + patch_size, left:left + patch_size]
    return tmp_img


class RandCrop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size
        
    def __call__(self, sample):
        # r_img : C x H x W (numpy)
        d_img = sample['d_img_org']
        d_name = sample['d_name']

        c, h, w = d_img.shape
        new_h = self.patch_size
        new_w = self.patch_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        ret_d_img = d_img[:, top: top + new_h, left: left + new_w]
        sample = {
            'd_img_org': ret_d_img,
            'd_name': d_name
        }

        return sample


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        d_img = sample['d_img_org']
        d_name = sample['d_name']

        d_img = (d_img - self.mean) / self.var

        sample = {'d_img_org': d_img, 'd_name': d_name}
        return sample


class RandHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample['d_img_org']
        d_name = sample['d_name']
        prob_lr = np.random.random()
        # np.fliplr needs HxWxC
        if prob_lr > 0.5:
            d_img = np.fliplr(d_img).copy()
        
        sample = {
            'd_img_org': d_img,
            'd_name': d_name
        }
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample['d_img_org']
        d_name = sample['d_name']
        d_img = torch.from_numpy(d_img).type(torch.FloatTensor)
        sample = {
            'd_img_org': d_img,
            'd_name': d_name
        }
        return sample
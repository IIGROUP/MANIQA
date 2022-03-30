import torch
import numpy as np


def crop_image(top, left, patch_size, img=None):
    tmp_img = img[:, :, top:top + patch_size, left:left + patch_size]
    return tmp_img


class RandCrop(object):
    def __init__(self, patch_size, num_crop):
        self.patch_size = patch_size
        self.num_crop = num_crop
        
    def __call__(self, sample):
        # r_img : C x H x W (numpy)
        r_img, d_img = sample['r_img_org'], sample['d_img_org']
        d_name = sample['d_name']

        c, h, w = d_img.shape
        new_h = self.patch_size
        new_w = self.patch_size
        ret_r_img = np.zeros((c, self.patch_size, self.patch_size))
        ret_d_img = np.zeros((c, self.patch_size, self.patch_size))
        for _ in range(self.num_crop):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            tmp_r_img = r_img[:, top: top + new_h, left: left + new_w]
            tmp_d_img = d_img[:, top: top + new_h, left: left + new_w]
            ret_r_img += tmp_r_img
            ret_d_img += tmp_d_img
        ret_r_img /= self.num_crop
        ret_d_img /= self.num_crop

        sample = {
            'r_img_org': ret_r_img,
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
        r_img, d_img = sample['r_img_org'], sample['d_img_org']
        d_name = sample['d_name']

        r_img = (r_img - self.mean) / self.var
        d_img = (d_img - self.mean) / self.var

        sample = {'r_img_org': r_img, 'd_img_org': d_img, 'd_name': d_name}
        return sample


class RandHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        r_img, d_img = sample['r_img_org'], sample['d_img_org']
        d_name = sample['d_name']
        prob_lr = np.random.random()
        # np.fliplr needs HxWxC
        if prob_lr > 0.5:
            d_img = np.fliplr(d_img).copy()
            r_img = np.fliplr(r_img).copy()
        
        sample = {
            'r_img_org': r_img,
            'd_img_org': d_img,
            'd_name': d_name
        }
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        r_img, d_img = sample['r_img_org'], sample['d_img_org']
        d_name = sample['d_name']
        d_img = torch.from_numpy(d_img).type(torch.FloatTensor)
        r_img = torch.from_numpy(r_img).type(torch.FloatTensor)

        sample = {
            'r_img_org': r_img,
            'd_img_org': d_img,
            'd_name': d_name
        }
        return sample
import torch
import numpy as np


def random_crop(d_img, config):
    b, c, h, w = d_img.shape
    top = np.random.randint(0, h - config.crop_size)
    left = np.random.randint(0, w - config.crop_size)
    d_img_org = crop_image(top, left, config.crop_size, img=d_img)
    return d_img_org


def crop_image(top, left, patch_size, img=None):
    tmp_img = img[:, :, top:top + patch_size, left:left + patch_size]
    return tmp_img


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


def split_dataset_koniq10k(txt_file_name, split_seed=20):
    np.random.seed(split_seed)
    object_data = []
    with open(txt_file_name, 'r') as listFile:
        for line in listFile:
            dis, score = line.split()
            dis = dis
            if dis not in object_data:
                object_data.append(dis)
    
    np.random.shuffle(object_data)
    np.random.seed(20)

    l = len(object_data)
    train_name = object_data[:int(l * 0.8)]
    val_name = object_data[int(l * 0.8):]
    return train_name, val_name


def split_dataset_kadid10k(txt_file_name, split_seed=20):
    np.random.seed(split_seed)
    object_data = []
    with open(txt_file_name, 'r') as listFile:
        for line in listFile:
            dis, score = line.split()
            dis = dis[:-1]
            if dis[1:3] not in object_data:
                object_data.append(dis[1:3])
    
    np.random.shuffle(object_data)
    np.random.seed(20)

    l = len(object_data)
    train_name = object_data[:int(l * 0.8)]
    val_name = object_data[int(l * 0.8):]
    return train_name, val_name


def split_dataset_tid2013(txt_file_name, split_seed=20):
    np.random.seed(split_seed)
    object_data = []
    with open(txt_file_name, 'r') as listFile:
        for line in listFile:
            score, dis = line.split()
            if dis[1:3] not in object_data:
                object_data.append(dis[1:3])
    
    np.random.shuffle(object_data)
    np.random.seed(20)

    l = len(object_data)
    train_name = object_data[:int(l * 0.8)]
    val_name = object_data[int(l * 0.8):]
    return train_name, val_name


def split_dataset_live(txt_file_name, split_seed=20):
    np.random.seed(split_seed)
    object_data = []
    with open(txt_file_name, 'r') as listFile:
        for line in listFile:
            i1, i2, ref, dis, score, h, w = line.split()
            if ref[8:] not in object_data:
                object_data.append(ref[8:])
    
    np.random.shuffle(object_data)
    np.random.seed(20)

    l = len(object_data)
    train_name = object_data[:int(l * 0.8)]
    val_name = object_data[int(l * 0.8):]
    return train_name, val_name


def split_dataset_csiq(txt_file_name, split_seed=20):
    np.random.seed(split_seed)
    object_data = []
    with open(txt_file_name, 'r') as listFile:
        for line in listFile:
            dis, score= line.split()
            dis_name, dis_type, idx_img, _ = dis.split(".")
            if dis_name not in object_data:
                object_data.append(dis_name)
    
    np.random.shuffle(object_data)
    np.random.seed(20)

    l = len(object_data)
    train_name = object_data[:int(l * 0.8)]
    val_name = object_data[int(l * 0.8):]
    return train_name, val_name


class RandCrop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size
        
    def __call__(self, sample):
        # r_img : C x H x W (numpy)
        d_img = sample['d_img_org']
        score = sample['score']

        c, h, w = d_img.shape
        new_h = self.patch_size
        new_w = self.patch_size
        
        # For koniq10k
        if h == new_h and w == new_w:
            ret_d_img = d_img
        else:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            ret_d_img = d_img[:, top: top + new_h, left: left + new_w]

        sample = {
            'd_img_org': ret_d_img,
            'score': score
        }
        return sample


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        d_img = sample['d_img_org']
        score = sample['score']
        d_img = (d_img - self.mean) / self.var
        sample = {'d_img_org': d_img, 'score': score}
        return sample


class RandHorizontalFlip(object):
    def __init__(self, prob_aug):
        self.prob_aug = prob_aug

    def __call__(self, sample):
        d_img = sample['d_img_org']
        score = sample['score']

        p_aug = np.array([self.prob_aug, 1 - self.prob_aug])
        prob_lr = np.random.choice([1, 0], p=p_aug.ravel())

        if prob_lr > 0.5:
            d_img = np.fliplr(d_img).copy()
        
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        return sample


class RandRotation(object):
    def __init__(self, prob_aug=0.5):
        self.prob_aug = prob_aug
        self.aug_count = 0

    def __call__(self, sample):
        d_img = sample['d_img_org']
        score = sample['score']

        p_aug = np.array([self.prob_aug, 1 - self.prob_aug])
        prob_lr = np.random.choice([1, 0], p=p_aug.ravel())

        if prob_lr > 0.5:
            p = np.array([0.33, 0.33, 0.34])
            idx = np.random.choice([1, 2, 3], p=p.ravel())
            d_img = np.rot90(d_img, idx, axes=(1, 2)).copy()
            self.aug_count += 1
        
        sample = {
            'd_img_org': d_img,
            'score': score,
            'aug_count': self.aug_count
        }
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample['d_img_org']
        score = sample['score']
        d_img = torch.from_numpy(d_img).type(torch.FloatTensor)
        score = torch.from_numpy(score).type(torch.FloatTensor)
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        return sample
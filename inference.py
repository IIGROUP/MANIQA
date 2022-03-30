import os
import torch
import numpy as np
import logging
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
from utils.eval_process_image import ToTensor, Normalize, crop_image
from data.ntire2022 import NTIRE2022
from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def eval_epoch(config, net, test_loader):
    with torch.no_grad():
        net.eval()
        name_list = []
        pred_list = []
        with open(config.valid_path + '/output.txt', 'w') as f:
            for data in tqdm(test_loader):
                pred = 0
                for i in range(config.num_avg_val):
                    x_d = data['d_img_org'].cuda()
                    x_d = five_point_crop(i, d_img=x_d, config=config)
                    pred += net(x_d)

                pred /= config.num_avg_val
                d_name = data['d_name']
                pred = pred.cpu().numpy()
                name_list.extend(d_name)
                pred_list.extend(pred)
            for i in range(len(name_list)):
                f.write(name_list[i] + ',' + str(pred_list[i]) + '\n')
            print(len(name_list))
        f.close()


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


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # dataset path
        "db_name": "PIPAL",                                        
        "val_ref_path": "/mnt/data_16TB/wth22/IQA_dataset/Dis/",
        "val_dis_path": "/mnt/data_16TB/wth22/IQA_dataset/Dis/",

        # optimization
        "batch_size": 10,
        "num_avg_val": 5,
        "crop_size": 224,

        # device
        "num_workers": 8,

        # load & save checkpoint
        "valid_path": "./output/valid/last_ensemble_vit16_finetune",
        "model_path": "./output/models/last_ensemble_vit16_finetune/epoch5"
    })

    logging.info(config)
    if not os.path.exists(config.valid_path):
        os.mkdir(config.valid_path)
    
    # data load
    val_dataset = NTIRE2022(
        ref_path=config.val_ref_path,
        dis_path=config.val_dis_path,
        transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )
    net = torch.load(config.model_path)
    net = net.cuda()

    losses, scores = [], []
    eval_epoch(config, net, val_loader)
    sort_file(config.valid_path + '/output.txt')
    
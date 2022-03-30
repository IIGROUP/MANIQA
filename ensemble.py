import os
import torch
import numpy as np
import logging
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
from utils.eval_process_image import ToTensor, Normalize, crop_image, RandHorizontalFlip
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


def eval_epoch(config, net7, net4, net1, test_loader):
    with torch.no_grad():
        net7.eval()
        net4.eval()
        net1.eval()
        name_list = []
        pred_list = []
        with open(config.valid_path + '/output.txt', 'w') as f:
            for data in tqdm(test_loader):
                pred7 = 0
                pred4 = 0
                pred1 = 0
                for i in range(config.num_avg_val):
                    x_d = data['d_img_org'].cuda()
                    x_d = five_point_crop(i, d_img=x_d, config=config)
                    pred7 += net7(x_d)
                    pred4 += net4(x_d)
                    pred1 += net1(x_d)
                    
                pred = 0.55 * pred7 + 0.2 * pred4 + 0.25 * pred1

                pred /= config.num_avg_val
                d_name = data['d_name']
                pred = pred.cpu().numpy()
                name_list.extend(d_name)
                pred_list.extend(pred)
            for i in range(len(name_list)):
                f.write(name_list[i] + ',' + str(pred_list[i][0]) + '\n')
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
        "num_avg_val": 1,
        "crop_size": 224,

        # device
        "num_workers": 8,
        
        # load & save checkpoint
        "valid_path": "./output/valid/ensemble1",
        "model_path1": "./output/models/test_model_attentionIQA2/epoch1", # 1.35
        # "model_path2": "./output/models/ensemble_attentionIQA2_finetune_seed15/epoch2",  # 1.34
        # "model_path3": "./output/models/ensemble_attentionIQA2_finetune/epoch1", # 1.37
        "model_path4": "./output/models/ensemble_attentionIQA2_finetune/epoch2",  # 1.38
        # "model_path5": "./output/models/ensemble_attentionIQA2_finetune_lr5/epoch2", # 1.37
        # "model_path6": "./output/models/ensemble_attentionIQA2_finetune_lr8/epoch2"  # 1.36
        "model_path7": "./output/models/ensemble_attentionIQA2_finetune_e2/epoch4", # 1.39 train one epoch
        # "model_path8": "./output/models/ensemble_attentionIQA2_finetune_train2/epoch3", # 1.38 train two epoch
        # "model_path9": "./output/models/ensemble_attentionIQA2_finetune_train2_seed22/epoch3", # 1.32
        # "model_path10": "./output/models/ensemble_attentionIQA2_finetune_train1_falsecudnn/epoch4" # 1.39
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

    net1 = torch.load(config.model_path1)
    net1 = net1.cuda()

    net4 = torch.load(config.model_path4)
    net4 = net4.cuda()

    net7 = torch.load(config.model_path7)
    net7 = net7.cuda()

    # train & validation
    losses, scores = [], []
    eval_epoch(config, net7, net4, net1, val_loader)

    sort_file(config.valid_path + '/output.txt')
    
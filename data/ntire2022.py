import os
import torch
import numpy as np
import cv2 


class NTIRE2022(torch.utils.data.Dataset):
    def __init__(self, ref_path, dis_path, transform):
        super(NTIRE2022, self).__init__()
        self.ref_path = ref_path
        self.dis_path = dis_path
        self.transform = transform

        ref_files_data, dis_files_data = [], []
        for dis in os.listdir(dis_path):
            ref = dis
            ref_files_data.append(ref)
            dis_files_data.append(dis)
        self.data_dict = {'r_img_list': ref_files_data, 'd_img_list': dis_files_data}

    def __len__(self):
        return len(self.data_dict['r_img_list'])
    
    def __getitem__(self, idx):
        # r_img: H x W x C -> C x H x W
        r_img_name = self.data_dict['r_img_list'][idx]
        r_img = cv2.imread(os.path.join(self.ref_path, r_img_name), cv2.IMREAD_COLOR)
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        r_img = np.array(r_img).astype('float32') / 255
        r_img = np.transpose(r_img, (2, 0, 1))

        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        sample = {
            'r_img_org': r_img,
            'd_img_org': d_img,
            'd_name': d_img_name
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

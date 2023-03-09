import os
import torch
import numpy as np
import cv2 


class PIPAL22(torch.utils.data.Dataset):
    def __init__(self, dis_path, transform):
        super(PIPAL22, self).__init__()
        self.dis_path = dis_path
        self.transform = transform

        dis_files_data = []
        for dis in os.listdir(dis_path):
            dis_files_data.append(dis)
        self.data_dict = {'d_img_list': dis_files_data}

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        sample = {
            'd_img_org': d_img,
            'd_name': d_img_name
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

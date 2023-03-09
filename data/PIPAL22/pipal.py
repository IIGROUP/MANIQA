import os
import torch
import numpy as np
import cv2


class PIPAL(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, transform, keep_ratio):
        super(PIPAL, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        name_type = {}
        
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                dis = dis[:-1]
                
                # obtain the spliting parts
                name = dis[:-4]
                split_list = dis.split('_')
                img_name, dis_type, level = split_list[0], split_list[1], split_list[2]

                if img_name + '_' + dis_type not in name_type.keys():
                    name_type[img_name + '_' + dis_type] = 1
                else:
                    name_type[img_name + '_' + dis_type] += 1

        count_name_type = {}
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                dis = dis[:-1]

                name = dis[:-4]
                split_list = dis.split('_')
                img_name, dis_type, level = split_list[0], split_list[1], split_list[2]

                if img_name + '_' + dis_type not in count_name_type.keys():
                    count_name_type[img_name + '_' + dis_type] = 1
                else:
                    count_name_type[img_name + '_' + dis_type] += 1

                if count_name_type[img_name + '_' + dis_type] <= int(name_type[img_name + '_' + dis_type] * keep_ratio):
                    score = float(score)
                    dis_files_data.append(dis)
                    score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        
        score = self.data_dict['score_list'][idx]
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

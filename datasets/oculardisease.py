# -*- coding: utf-8 -*-
import os
import pdb
import math
import numpy as np
from PIL import Image
import pandas as pd
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import sys
from pathlib import Path
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../.."))
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to
# the sys.path.
sys.path.append(parent)
from datasets.dataset import MultipleDomainDataset
from engine.configs import Datasets

def RGB_loader(path):
    return Image.open(path).convert('RGB')

# python -m pdb main.py --gpu_ids 0 --data_name Ocular --data_path '/lustre06/project/6054110/absking/dataset/OcularDisease/' --num_classes 8 --data_size '[3, 224, 224]' --source-domains 25 --intermediate-domains 2 --target-domains 6 --mode train --model-func resnet18 --feature-dim 512 --epochs 80 --iterations 100 --train_batch_size 8 --eval_batch_size 8 --test_epoch -1 --algorithm SDE --seed 0 --save_path './logs/Ocular/SDE/uni' --record --mlp-depth 3 --mlp-width 512 --dropout 0. --path_weight 1 --uni
# python -m pdb main.py --gpu_ids 0 --data_name Ocular --data_path '/lustre06/project/6054110/absking/dataset/OcularDisease/' --num_classes 8 --data_size '[3, 224, 224]' --source-domains 25 --intermediate-domains 2 --target-domains 6 --mode train --model-func resnet18 --feature-dim 512 --epochs 80 --iterations 100 --train_batch_size 12 --eval_batch_size 12 --test_epoch -1 --algorithm ERM --seed 0 --save_path './logs/Ocular/ERM' --record --mlp-depth 3 --mlp-width 512 --dropout 0. --path_weight 1
# PS: grey scale works better
@Datasets.register('ocular')
class MultipleEnvironmentOcular(MultipleDomainDataset):
    def __init__(self, root, input_shape, num_classes, dataset_transform=None):
        super().__init__()
        num_domains = 33 # 12 # 33
        num_source = 27 # 10  # 25
        val_test = 6 # 2  # 8
        # self.Environments = environments
        self.Environments = np.arange(num_domains)
        self.root = root
        self.input_shape = input_shape
        self.num_classes = num_classes

        transform = transforms.Compose([
            transforms.Resize(self.input_shape[-2:]),
            transforms.Grayscale(), #
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize(self.input_shape[-2:]),
            # transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.Grayscale(),#transforms.RandomGrayscale(), #
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        df = pd.read_csv(os.path.join(root, 'full_df.csv'))  #'/lustre06/project/6054110/absking/dataset/OcularDisease/full_df.csv')
        # ROOT = os.path.join(root, )'/lustre06/project/6054110/absking/dataset/OcularDisease/'
        df.head(10)

        raw_data = df.drop(columns=['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O', 'labels'])

        raw_data.head();raw_data.columns;raw_data['Patient Age']
        # plt.hist(raw_data['Patient Age'], bins=10, edgecolor='black')
        # plt.savefig('foo.png')
        raw_data["target"].unique().size
        raw_data.shape

        targets = np.array(raw_data["target"].apply(lambda x: json.loads(x)).tolist())
        raw_data["class_name"] = np.argmax(targets, axis=1)
        raw_data["class_name"][raw_data["class_name"] > 2] = 2
        
        str_info = "class info"
        # for i in range(8):
        #     count = np.sum(raw_data["class_name"]==i)
        #     str_info += f" {i}:{count}"
        # print(str_info)
        # raw_data["class_name"] = raw_data["class_name"] .replace(classes)
        print(raw_data.head()) 

        COL = 'Patient Age'
        MAX_ = 85 #80  # df[col].max()
        MIN_ = 27 #40  # df[col].min()
        COUNT_ = num_domains # 33 #26
        ranges = np.arange(MIN_, MAX_, (MAX_ - MIN_) / COUNT_)
        ranges = np.concatenate([np.ones(1)*0,ranges])
        ranges = np.concatenate([ranges, np.ones(1)*100])
        ranges = np.concatenate([ranges[:2],ranges[3:]])

        cut_res = pd.cut(raw_data[COL], ranges)
        gb = raw_data.groupby(cut_res)
        self.datasets = []
        for d in range(len(gb.groups.keys())):
            # d - domain; gb_key - domain_range;
            gb_key = list(gb.groups.keys())[d]
            domain_df = gb.get_group(gb_key)
            domain_list = []
            print(f"{d}-th domain with {gb_key}: {len(domain_df)}")
            image_name_list = domain_df.filename.tolist()
            #, row['Patient Age'], row.class_name)
            # age_list = domain_df.filename.tolist()
            label_list = np.array(domain_df['class_name'].tolist(), dtype=np.int64)
            str_info = "class info"
            for i in range(3):
                str_info += f" {i}:{np.sum(label_list==i)}"
            print(str_info)
             
            if d >= num_source:
                self.datasets.append(SubOcular(root, image_name_list, label_list, transform))
            else:
                self.datasets.append(SubOcular(root, image_name_list, label_list, augment_transform))
        

class SubOcular(Dataset):
    def __init__(self, root, image_name_list, image_labels, dataset_transform, loader=RGB_loader):
        super(SubOcular, self).__init__()
        self.root = root
        self.image_name_list = image_name_list
        self.image_labels = image_labels
        self.transform = dataset_transform
        self.loader = loader

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'preprocessed_images/', self.image_name_list[idx])  #os.path.join(self.root, self.image_name_list[idx])
        label = self.image_labels[idx]
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_name_list)


if __name__ == "__main__":
    import pandas as pd
    import json
    import matplotlib.pyplot as plt

    

    # Input data files are available in the read-only "../data/" directory
    # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
    torch.__version__
    ROOT = '/lustre06/project/6054110/absking/dataset/OcularDisease/'
    input_shape = (224, 224, 3)
    num_classes = 8
    MultipleEnvironmentOcular(ROOT, input_shape, num_classes, dataset_transform=None)
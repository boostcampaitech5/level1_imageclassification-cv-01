import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion
import pandas as pd

def seed_everything(seed):
    # seed 고정

def get_lr(optimizer):
    # lr 설정


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size
    # 텐서보드 이미지 표시해주기


# loss, accuracy 그래프로 저장
# def save_graph(losses, accuracy, save_dir):
#     plt.figure(figsize=(10, 5))

#     plt.subplot(1,2,1)
#     plt.plot(np.array(losses), "blue")
#     _, _, y1, y2 = plt.axis()
#     plt.xlim([0, args.epochs])
#     plt.ylim([0, y2])
#     plt.xlabel("epoch")
#     plt.ylabel("loss")

#     plt.subplot(1,2,2)
#     plt.plot(np.array(accuracy), "green")
#     plt.xlim([0, args.epochs])
#     plt.ylim([0, 1])
#     plt.xlabel("epoch")
#     plt.ylabel("accuracy")

#     plt.savefig(f"{save_dir}/graph.png")

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    
    # path 설정, exp파일 저장 시 번호 추가해주는 기능


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset 가져와서 augmentation 적용,  default: MaskBaseDataset

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    # transform에 augmentation 설정

    # -- data_loader
    # train, val dataset 분리
   
    y_train=[]
    for images, labels in train_set:
        labels = int(labels)
        y_train.append(labels) # 전부 저장
    class_sample_count = [2745, 2050, 415, 3660, 4085, 545, 549, 410, 83, 732, 817, 109, 549, 410, 83, 732, 817, 109]    # 18개 클래스 별 count
    weight = 1. / np.array(class_sample_count)   # 1/n로 가중치 부여 클래스 별, 개수가 많은 클래스면 가중치 작게 설정
    samples_weight = np.array([weight[t] for t in y_train])  # 모든 레이벨에 대한 가중치 적용
    samples_weight = torch.from_numpy(samples_weight)    # 텐서형태로 바꿔줌
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)  

    # dataloader설정
    train_loader = DataLoader(
    )

    val_loader = DataLoader(
    )

    # -- model
    # default: CustomModel 적용

    model = torch.nn.DataParallel(model)
    # 병렬처리


    # -- loss & metric
    # loss함수, optimizer, scheduler 적용
    
    # -- logging
    
    
    for epoch in range(args.epochs):
        # train loop
        model.train()
        # 학습 과정 ...
        
        
        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            #  학습 과정 ...
            

            # save_graph(losses, accuracy, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: CustomAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 64)')
    parser.add_argument('--model', type=str, default='EfficientNet_MultiLabel', help='model type (default: EfficientNet_MultiLabel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='focal', help='criterion type (default: FocalLoss)')
    parser.add_argument('--lr_decay_step', type=int, default=200, help='learning rate scheduler deacy step (default: 200)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # parser.add_argument('--CEloss', type=float, default=1, help='weight of cross_entropy')
    # parser.add_argument('--FOloss', type=float, default=0, help='weight of FocalLoss')
    # parser.add_argument('--LAloss', type=float, default=0, help='weight of LabelSmoothingLoss')
    # parser.add_argument('--F1loss', type=float, default=0, help='weight of F1Loss')
    # parser.add_argument('--CEBloss', type=float, default=0, help='weight of CrossEntropyLossWithClassBalancing')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--df_path', type=str, default=os.environ.get('SM_DF_DIR','/opt/ml/input/data/train' ))
    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    df_dir = args.df_path

    df = pd.read_csv(df_dir+'/train.csv')
    fetoma=[336, 342, 374, 430, 933, 1369, 1387, 1560, 1570, 2399, 2400, 2401, 2402, 2403, 2404]
    matofe=[569, 764, 1912]
    df.loc[fetoma,'gender']='male'
    df.loc[matofe,'gender']='female'
    df.to_csv(df_dir+'/train.csv')

    train(data_dir, model_dir, args)
    

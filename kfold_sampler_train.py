import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

import torch.nn.init as init
from sklearn.model_selection import KFold


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# # loss, accuracy 그래프로 저장
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

    
def load_model(saved_model, num_classes, device):

    return model


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    # save_dir = increment_path(os.path.join(model_dir, args.name))
    # save_dir = increment_path(os.path.join(save_dir,'fold_'+str(fold+1)))

    # -- settings
 

    # -- dataset


    # -- augmentation


    # -- K-fold
    n_splits = args.kfold
    kfold = KFold(n_splits=n_splits, shuffle=True) # 데이터 불균형 해소를 위해 shuffle = True

    # -- loss & metric
    CEloss = create_criterion("cross_entropy")
    FOloss = create_criterion("focal")
    LAloss = create_criterion("label_smoothing")
    F1loss = create_criterion("f1")
    CEBloss = create_criterion("cross_entropy_class_balancing")


    k_fold = []
    val_idx_list = []

    # -- early_stopping
    patience = args.early_stop # patience 이상 val_acc가 update되지 않으면 stop
    trigger_times = 0 


    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)): # train set 중 0.2를 validation으로 사용
        y_train = []
        for idx in train_idx:
            labels = dataset[idx][1]
            y_train.append(labels)
        class_sample_count = [2745, 2050, 415, 3660, 4085, 545, 549, 410, 83, 732, 817, 109, 549, 410, 83, 732, 817, 109]    # 18개 클래스 별 count
        weight = 1. / np.array(class_sample_count)   # 1/n로 가중치 부여 클래스 별, 개수가 많은 클래스면 가중치 작게 설정
        samples_weight = np.array([weight[t] for t in y_train])  # 모든 레이벨에 대한 가중치 적용
        samples_weight = torch.from_numpy(samples_weight)    # 텐서형태로 바꿔줌
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
          
        best_val_acc = 0
        best_val_loss = np.inf
        losses = []
        accuracy = []
        print('==============[', fold+1,'fold',']==============')
        print('val_idx', val_idx)
        save_fold = increment_path(os.path.join(save_dir,'fold_'+str(fold+1)))

        # -- model
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        ).to(device)
        # train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        # criterion = create_criterion(args.criterion)  # default: cross_entropy
 
        
        #print(val_idx)
        val_idx_list.append(val_idx)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler
            )
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=val_subsampler
            )
    
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):


            # val loop
            with torch.no_grad():
     
            #save_graph(losses, accuracy, save_dir)
        k_fold.append([fold,best_val_acc])

        # -- Early stopping
        val_acc = np.sum(val_acc_items) / len(val_idx)
        if val_acc < best_val_acc:
            trigger_times += 1
            print('Trigger Times:', trigger_times)
            if trigger_times >= patience:
                print('Early stopping!')
                break
        else:
            print('trigger times: 0')
            trigger_times = 0

    # -- save validation set index
    df_val_idx = pd.DataFrame(val_idx_list)
    df_val_idx.to_csv(os.path.join(save_dir,'val_idx_list.csv'), index=False)

    # -- print k-fold result
    print(f"============== [Fold reslt] ==============")
    for fold, best_val_acc in k_fold:
        print("k-fold :", fold+1 ,f" Validation acc: {best_val_acc:4.2%}")
    
    mean_val_acc = sum([best_val_acc for _, best_val_acc in k_fold]) / args.kfold
    print("Mean Validation Accuracy = {:.2f}%".format(mean_val_acc*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='EfficientNet_MultiLabel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='focal', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=200, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    
    parser.add_argument('--kfold', type=int, default=5, help='number of splits for kfold (default: 5)')
    parser.add_argument('--early_stop', type=int, default=10, help='patience for early stopping (default: 10)')

    parser.add_argument('--CEloss', type=float, default=1, help='weight of cross_entropy')
    parser.add_argument('--FOloss', type=float, default=0, help='weight of FocalLoss')
    parser.add_argument('--LAloss', type=float, default=0, help='weight of LabelSmoothingLoss')
    parser.add_argument('--F1loss', type=float, default=0, help='weight of F1Loss')
    parser.add_argument('--CEBloss', type=float, default=0, help='weight of CrossEntropyLossWithClassBalancing')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--df_path', type=str, default=os.environ.get('SM_DF_DIR','/opt/ml/input/data/train' ))
    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    df_dir = args.df_path
    # -- change data label
    df = pd.read_csv(df_dir+'/train.csv')
    fetoma=[336, 342, 374, 430, 933, 1369, 1387, 1560, 1570, 2399, 2400, 2401, 2402, 2403, 2404]
    matofe=[569, 764, 1912]
    df.loc[fetoma,'gender']='male'
    df.loc[matofe,'gender']='female'
    df.to_csv(df_dir+'/train.csv')

    train(data_dir, model_dir, args)

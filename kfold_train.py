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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

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


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

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

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"
    
def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- K-fold
    n_splits = args.kfold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed) # 데이터 불균형 해소를 위해 shuffle = True

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    CEloss = create_criterion("cross_entropy")
    FOloss = create_criterion("focal")
    LAloss = create_criterion("label_smoothing")
    F1loss = create_criterion("f1")
    CEBloss = create_criterion("cross_entropy_class_balancing")

    # criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    losses = []
    accuracy = []
    k_fold = []
    val_idx_list = []

    # -- early_stopping
    patience = args.early_stop # patience 이상 val_acc가 update되지 않으면 stop
    trigger_times = 0 

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)): # train set 중 0.2를 validation으로 사용
        print('==============[', fold+1,'fold',']==============')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        #print(val_idx)
        val_idx_list.append(val_idx)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_subsampler
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
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = CEloss(outs, labels) * args.CEloss + FOloss(outs, labels) * args.FOloss + LAloss(outs, labels) * args.LAloss + F1loss(outs, labels) * args.F1loss + CEBloss(outs, labels) * args.CEBloss
                # loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    # loss_item = (criterion(outs, labels)).item()
                    loss_item = (CEloss(outs, labels) * args.CEloss + FOloss(outs, labels) * args.FOloss + LAloss(outs, labels) * args.LAloss + F1loss(outs, labels) * args.F1loss + CEBloss(outs, labels) * args.CEBloss).item()

                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_idx)

                losses.append(val_loss)
                accuracy.append(val_acc)

                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best_"f"{fold+1}.pth")
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{save_dir}/last_"f"{fold+1}.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                print()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
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

    # -- delete cache data
    print('delete cache memory')
    torch.cuda.empty_cache()

    train(data_dir, model_dir, args)

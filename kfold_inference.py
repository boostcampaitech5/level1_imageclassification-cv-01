import argparse
import multiprocessing
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device):

    model_path = os.path.join(saved_model, 'best.pth')

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):

    # kfold별 best.pth를 각각 불러온다.
    with torch.no_grad():
        for fold in range(args.kfold):
            model_dir = os.path.join(args.model_dir, 'fold_{}'.format(fold+1)) # define model_dir
            model = load_model(model_dir, num_classes, device).to(device) # load model
            model.eval()
            fold_preds = []
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred = model(images) # batch_size * num_classes 사이즈를 가진 tensor
                fold_preds.extend(pred.cpu())
            if fold == 0:
                preds = fold_preds
            else:
                preds = [pred + fold_pred for pred, fold_pred in zip(preds, fold_preds)]
        preds = [pred.argmax(dim=-1).item() for pred in preds]
    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 64)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (224, 224))')
    parser.add_argument('--model', type=str, default='EfficientNet_MultiLabel', help='model type (default: BaseModel)')
    parser.add_argument('--kfold', type=int, default=5, help='number of splits for kfold (default: 5)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)


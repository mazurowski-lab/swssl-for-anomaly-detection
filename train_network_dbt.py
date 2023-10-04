import argparse
import time
import torch
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import pickle
import random
import json
from PIL import Image, ImageOps
from torch import nn
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors

from utils import *
from evaluate import evaluate_image
from dataset import DBTDataset, ChestDataset
from models import Patch_Model

def twin_loss(f_patch1, f_patch2, f_neg=None, p=False, target=None, f_full1=None, f_full2=None, f_neg_full=None):
    batch_size, dimension = f_patch1.shape

    f_patch1_norm = (f_patch1 - f_patch1.mean(0)) / f_patch1.std(0)
    f_patch2_norm = (f_patch2 - f_patch2.mean(0)) / f_patch2.std(0)

    pos_score = torch.mm(f_patch1_norm.t(), f_patch2_norm) / batch_size
    diff = (pos_score - torch.eye(dimension).cuda()).pow(2)
    loss = diff.diag().sum()

    non_diag_weight = (torch.ones([dimension, dimension]) - torch.eye(dimension)) * 1e-6
    non_diag_weight = non_diag_weight.cuda()

    diff *= non_diag_weight
    loss += diff.sum()

    if f_neg is not None:
        f_patch1_norm = F.normalize(f_patch1, dim=-1)
        f_patch2_norm = F.normalize(f_patch2, dim=-1)
        f_neg_norm = F.normalize(f_neg, dim=-1)

        pair_score = torch.mm(f_patch1_norm, f_patch2_norm.t())
        pair_sim = torch.sigmoid(pair_score.diag())
        pair_loss = torch.abs(pair_sim - torch.ones(pair_score.shape[0]).cuda()).sum()
    
        neg_score = torch.mm(f_patch1_norm, f_neg_norm.t())
        neg_sim = torch.sigmoid(neg_score.diag())
        neg_loss = torch.abs(neg_sim - target).sum()
        
        loss += neg_loss + pair_loss 
        
    # Printing for debugging and tracking
    if p:
        if f_neg is not None:
            print('pair loss ', pair_loss.item())
            print('neighbor loss ', neg_loss.item())
        print('total loss ', loss.item())
        print('feature sample:')
        print(f_patch1_norm[0][:10])
        print(f_patch2_norm[0][:10])
        print(f_patch1_norm[1][:10])

    return loss

def train(model, device, args):
    # Dataloader
    if args.category == 'chest':
        train_transforms = transforms.Compose([
                   transforms.Resize((256*4, 256*4), Image.ANTIALIAS),
        ])

        test_transforms = transforms.Compose([
                   transforms.Resize((256*4, 256*4), Image.ANTIALIAS),
                   transforms.ToTensor()
        ])
        
        # Train set
        train_patch_d = ChestDataset(root = args.dataset_path, 
                                         pre_transform = train_transforms,
                                         phase = 'train',
                                         patch = True,
                                         patch_size = args.patch_size, 
                                         step_size = args.step_size)
        # Test set 
        # Need all training images with full size for generating normal feature
        # Then test on testing images with full size
        train_full_d = ChestDataset(root = args.dataset_path, 
                                    pre_transform = test_transforms,
                                    phase = 'train', 
                                    patch = False, 
                                    patch_size = args.patch_size, 
                                    step_size = args.step_size)
        test_full_d = ChestDataset(root = args.dataset_path, 
                                   pre_transform = test_transforms,
                                   phase = 'test',
                                   patch = False,
                                   patch_size = args.patch_size, 
                                   step_size = args.step_size)

    if args.category == 'dbt':
        train_transforms = transforms.Compose([
                   transforms.Resize((256*4, 256*3), Image.ANTIALIAS),])

        test_transforms = transforms.Compose([
                   transforms.Resize((256*4, 256*3), Image.ANTIALIAS),
                   transforms.ToTensor()])
        
        train_patch_d = DBTDataset(root = args.dataset_path, 
                                       pre_transform = train_transforms,
                                       phase = 'train',
                                       patch = True,
                                       patch_size = args.patch_size, 
                                       step_size = args.step_size)
        # Test set 
        # Need all training images with full size for generating normal feature
        # Then test on testing images with full size
        train_full_d = DBTDataset(root = args.dataset_path, 
                                  pre_transform = test_transforms,
                                  phase = 'train',
                                  patch = False,
                                  patch_size = args.patch_size, 
                                  step_size = args.step_size)
        test_full_d = DBTDataset(root = args.dataset_path, 
                                pre_transform = test_transforms,
                                phase = 'val',
                                patch = False,
                                patch_size = args.patch_size, 
                                step_size = args.step_size)


    train_patch_loader = DataLoader(train_patch_d, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_full_d, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_full_d, batch_size=1, shuffle=False)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

    best_score = -1
    score = evaluate_image(args, model, train_loader, test_loader, device, category=args.category)
    for epoch in range(args.epochs):
        with tqdm(total=len(train_patch_d), desc=f'Epoch {epoch + 1} / {args.epochs}', unit='img') as pbar:
            for idx, data in enumerate(train_patch_loader):
                img, img_aug, img_2, sim = data

                img = img.to(device)
                img_2 = img_2.to(device)
                img_aug = img_aug.to(device)
                sim = sim.to(device)
                
                f_patch, tmp = model(img)
                f_patch2, _ = model(img_2)
                f_aug, _ = model(img_aug)

                loss = twin_loss(f_patch, f_aug, f_neg=f_patch2, target=sim)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
            
                # tqdm Update 
                pbar.set_postfix(**{'twin loss': loss.item()})
                pbar.update(img.shape[0])
        
        # Evaluate
        if epoch > 0 and epoch % 10 == 0:
            twin_loss(f_patch, f_aug, f_neg=f_patch2, target=sim, p=1)
            score = evaluate_image(args, model, train_loader, test_loader, device, category=args.category)
            if score > best_score:
                torch.save(model.state_dict(), 'checkpoints/%s_%s_%s.pth' % (args.category, epoch, str(score)))
                best_score = score
            print('img lv curr acc %s, best acc %s' % (str(score), str(best_score)))
        
def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default='../dbt_dataset')
    parser.add_argument('--category', default='dbt')
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--load_size', default=256) # 256
    parser.add_argument('--input_size', default=256)
    parser.add_argument('--coreset_sampling_ratio', default=0.01)
    parser.add_argument('--project_root_path', default='results')
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--k', type=int, default=9)
    parser.add_argument('--learning-rate-weights', default=0.01, type=float, metavar='LR',
                        help='base learning rate for weights')
    parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                        help='base learning rate for biases and batch norm parameters')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--step_size', type=int, default=32)
    parser.add_argument('--use_tumor', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()

    model = Patch_Model(input_channel=3)
    model.to(device)
    train(model, device, args)

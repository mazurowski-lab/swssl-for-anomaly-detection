import csv
import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
from utils import Transform, create_image_name

class DBTDataset(Dataset):
    def __init__(self, root, pre_transform, phase, \
                 patch=False, patch_size=128, step_size=20):
        # Load Hyperparameter
        self.root = root
        self.phase = phase
        self.pre_transform = pre_transform
        self.patch = patch
        if patch: 
            self.patch_size = patch_size
            self.step_size = step_size
            self.ss_transform = Transform()
            self.to_im = transforms.ToTensor()
        
        # Normal Images Info
        self.normal_path = os.path.join(root, 'images-normal-unique')
        self.normal_path = os.path.join(self.normal_path, self.phase)
        self.normal_images = [f for f in os.listdir(self.normal_path)]
        self.edge_path = os.path.join(root, 'images-normal-unique-transferred')
        self.edge_path = os.path.join(self.edge_path, self.phase)

        # Tumor Images Info
        if phase != 'train':
            self.tumor_path = os.path.join(root, 'images-tumor-full')
            self.tumor_path = os.path.join(self.tumor_path, self.phase)
            self.tumor_images = [f for f in os.listdir(self.tumor_path)]

        print('%s dataset size %s' % (phase, len(self)))
    
    def __len__(self):
        if self.phase == 'train':
            return len(self.normal_images)
        else:
            return len(self.tumor_images) + len(self.normal_images)

    def __getitem__(self, idx):
        # Load Images. Tumor images are last
        if idx < len(self.normal_images):
            img_name = self.normal_images[idx]
            img_path = os.path.join(self.normal_path, img_name)
        else:
            img_path = self.tumor_images[idx - len(self.normal_images)]
            img_path = os.path.join(self.tumor_path, img_path)

        img = Image.open(img_path).convert('RGB')
        edg = Image.fromarray(255 - np.array(img.copy()))
        
        # Need patches during training, and full images during inference
        img = self.pre_transform(img)
        if not self.patch:
            return img, idx >= len(self.normal_images)
        else: 
            x_length = (img.size[0] - self.patch_size) // self.step_size - 1
            y_length = (img.size[1] - self.patch_size) // self.step_size - 1
            
            # Find a non-zero patch
            while 1:
                x_start, y_start = random.randint(0, x_length), random.randint(0, y_length)
                
                x_s = x_start * self.step_size
                y_s = y_start * self.step_size

                img_patch = img.crop((x_s,y_s,x_s+self.patch_size,y_s+self.patch_size))
                max_pixel = img_patch.getextrema()[0][1]
                if max_pixel > 0:
                    edg_patch = edg.crop((x_s,y_s,x_s+self.patch_size,y_s+self.patch_size))
                    break

            # Apply self-supervised learning augmentation
            img_11, _ = self.ss_transform(img_patch)
            img_12, _ = self.ss_transform(edg_patch)

            # Find neighbor image of patch
            while 1:
                x_start2, y_start2 = random.randint(0, x_length), random.randint(0, y_length)
                if abs(x_start2 - x_start) <= 3 and abs(y_start2 - y_start) <= 3:
                    if x_start == x_start2 and y_start == y_start2:
                        continue
                    x_s = x_start2 * self.step_size
                    y_s = y_start2 * self.step_size
                    img_patch = img.crop((x_s,y_s,x_s+self.patch_size,y_s+self.patch_size))
                    max_pixel = img_patch.getextrema()[0][1]
                    # Record difference
                    if max_pixel > 0:
                        dis_x = abs(x_start2-x_start) 
                        dis_y = abs(y_start2-y_start) 

                        sim_x = 0 if dis_x == 3 else 0.5 / (dis_x + 1)
                        sim_y = 0 if dis_y == 3 else 0.5 / (dis_y + 1)
                        sim = (sim_x + sim_y)

                        break
            # Neighbor image is applied same augmentation
            img_21 = self.to_im(img_patch)
            
            # img_11: patch with 1st augmentation
            # img_12: patch with 2nd augmentation
            # img_21: neighbor of img_1* patch with 1st augmentation
            return img_11, img_12, img_21, sim


class ChestDataset(Dataset):
    def __init__(self, root, pre_transform, phase, \
                 patch=False, patch_size=128, step_size=20):
        # Load Hyperparameter
        self.root = root
        self.phase = phase
        self.pre_transform = pre_transform
        self.patch = patch
        if patch: 
            self.patch_size = patch_size
            self.step_size = step_size
            self.ss_transform = Transform()
            self.to_im = transforms.ToTensor()
        
        # Normal Images Info
        self.normal_path = os.path.join(root, self.phase, '0.normal')
        self.normal_images = [f for f in os.listdir(self.normal_path)]

        # Tumor Images Info
        if phase != 'train':
            self.tumor_path = os.path.join(root, self.phase, '1.abnormal')
            self.tumor_images = [f for f in os.listdir(self.tumor_path)]

        print('%s dataset size %s' % (phase, len(self)))
    
    def __len__(self):
        if self.phase == 'train':
            return len(self.normal_images)
        else:
            return len(self.tumor_images) + len(self.normal_images)

    def __getitem__(self, idx):
        # Load Images. Tumor images are last
        if idx < len(self.normal_images):
            img_name = self.normal_images[idx]
            img_path = os.path.join(self.normal_path, img_name)
        else:
            img_path = self.tumor_images[idx - len(self.normal_images)]
            img_path = os.path.join(self.tumor_path, img_path)

        img = Image.open(img_path).convert('RGB')
        edg = Image.fromarray(255 - np.array(img.copy()))
        
        # Need patches during training, and full images during inference
        img = self.pre_transform(img)
        if not self.patch:
            return img, idx >= len(self.normal_images)
        else: 
            x_length = (img.size[0] - self.patch_size) // self.step_size - 1
            y_length = (img.size[1] - self.patch_size) // self.step_size - 1
            
            # Find a non-zero patch
            while 1:
                x_start, y_start = random.randint(0, x_length), random.randint(0, y_length)
                
                x_s = x_start * self.step_size
                y_s = y_start * self.step_size

                img_patch = img.crop((x_s,y_s,x_s+self.patch_size,y_s+self.patch_size))
                max_pixel = img_patch.getextrema()[0][1]
                if max_pixel > 0:
                    edg_patch = edg.crop((x_s,y_s,x_s+self.patch_size,y_s+self.patch_size))
                    break

            # Apply self-supervised learning augmentation
            img_11, _ = self.ss_transform(img_patch)
            img_12, _ = self.ss_transform(edg_patch)

            # Find neighbor image of patch
            while 1:
                x_start2, y_start2 = random.randint(0, x_length), random.randint(0, y_length)
                if abs(x_start2 - x_start) <= 3 and abs(y_start2 - y_start) <= 3:
                    if x_start == x_start2 and y_start == y_start2:
                        continue
                    x_s = x_start2 * self.step_size
                    y_s = y_start2 * self.step_size
                    img_patch = img.crop((x_s,y_s,x_s+self.patch_size,y_s+self.patch_size))
                    max_pixel = img_patch.getextrema()[0][1]
                    # Record difference
                    if max_pixel > 0:
                        dis_x = abs(x_start2-x_start) 
                        dis_y = abs(y_start2-y_start) 

                        sim_x = 0 if dis_x == 3 else 0.5 / (dis_x + 1)
                        sim_y = 0 if dis_y == 3 else 0.5 / (dis_y + 1)
                        sim = (sim_x + sim_y)

                        break
            # Neighbor image is applied same augmentation
            img_21 = self.to_im(img_patch)
            
            # img_11: patch with 1st augmentation
            # img_12: patch with 2nd augmentation
            # img_21: neighbor of img_1* patch with 1st augmentation
            return img_11, img_12, img_21, sim

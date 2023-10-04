import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
import json
from sklearn.neighbors import KernelDensity
import time

def evaluate_image(args, model, train_loader, test_loader, device, category='dbt'):
    model.eval()

    # Extract Normal Image Features
    embedding_list = []
    patch_size = args.patch_size
    sliding_step = args.step_size

    patch_embeddings = None

    with torch.no_grad():
        for idx, data in enumerate(train_loader):
            curr = time.time()
            # Fit img to size B * 3 * H * W
            img, _ = data
            img = img.to(device)

            x_length = (img.shape[2] - patch_size) // sliding_step
            y_length = (img.shape[3] - patch_size) // sliding_step

            single_patch_embeddings = None
            for i in range(x_length):
                for j in range(y_length):
                    start_x = i*sliding_step
                    start_y = j*sliding_step
                    curr_patch = img[:,:,start_x:start_x+patch_size,start_y:start_y+patch_size]
                    _, feature = model(curr_patch)
                    feature = feature.cpu().squeeze().unsqueeze(-1)
                    if single_patch_embeddings is None:
                        single_patch_embeddings = feature
                    else:
                        single_patch_embeddings = torch.cat([single_patch_embeddings, feature], dim=-1)

            batch_size = single_patch_embeddings.shape[0]
            hidden_size = single_patch_embeddings.shape[1]
            single_patch_embeddings = single_patch_embeddings.reshape(batch_size,hidden_size,-1).cpu()

            if patch_embeddings is None:
                patch_embeddings = single_patch_embeddings
            else:
                patch_embeddings = torch.cat([patch_embeddings, single_patch_embeddings], dim=0)

            if idx % 10 == 0:
                print('load feature %s/%s, instance time %s' % (str(idx), str(len(train_loader)), str(time.time() - curr)))
    print('patch embedding size :', patch_embeddings.shape)
    _, patch_dim, patch_num = patch_embeddings.shape

    # Testing
    gt_list_img_lvl = []
    pred_list_img_lvl = []
    score_patch_list = []
    
    patch_embeddings = patch_embeddings.numpy()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            curr = time.time()
            img, label = data

            img = img.to(device)
            embedding_tests = None
            for i in range(x_length):
                for j in range(y_length):
                    start_x = i*sliding_step
                    start_y = j*sliding_step
                    curr_patch = img[:,:,start_x:start_x+patch_size,start_y:start_y+patch_size]
                    _, feature = model(curr_patch)
                    feature = feature.cpu().squeeze().unsqueeze(-1)
                    if embedding_tests is None:
                        embedding_tests = feature
                    else:
                        embedding_tests = torch.cat([embedding_tests, feature], dim=-1)
            
            if idx % 50 == 0:
                print('Extraction', time.time() - curr)

            # hidden * size
            embedding_tests = embedding_tests.squeeze().cpu()
            embedding_tests = embedding_tests.reshape(hidden_size, -1).numpy()
            
            # Compute distance
            dis_all = np.linalg.norm(patch_embeddings - embedding_tests, axis=-1)
            score_patches = np.min(dis_all, axis=0)
            image_score = max(score_patches)

            pred_list_img_lvl.append(float(image_score))
            gt_list_img_lvl.append(label.numpy()[0])
            score_patch_list.append(score_patches.tolist())

            if idx % 50 == 0:
                print('evaluate %s/%s, instance time %s' % (str(idx), str(len(test_loader)), str(time.time() - curr)))
            
        pred_img_np = np.array(pred_list_img_lvl)
        gt_img_np = np.array(gt_list_img_lvl)
        img_auc = roc_auc_score(gt_img_np, pred_img_np)
        print("image-level auc-roc score : %f" % img_auc)

        file_name = 'results/performance_%s_%s.json' % (category, str(img_auc))
        with open(file_name, 'w+') as f:
            gt_list_int = [int(i) for i in gt_list_img_lvl]
            json.dump([gt_list_int, pred_list_img_lvl, score_patch_list], f)

    model.train()
    return img_auc

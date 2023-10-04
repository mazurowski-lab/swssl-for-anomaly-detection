import torch
import random
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter

def create_image_name(patient_id, study_uid, view, slice_id):
    tmp = ''
    for s in view:
        if not s.isdigit():
            tmp += s
    view = tmp
    return patient_id + '_' + study_uid + '_' + view + '_' + slice_id + '.png'

def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)
    return dist

class NN():
    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):
    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        knn = dist.topk(self.k, largest=False)
        return knn

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Invertion(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.invert(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(0.4, 1.)),
            transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0, hue=0)],
                                p=0.8),
            # Chest
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            # DBT
            #GaussianBlur(p=1.0),
            #Solarization(p=0.0),
            transforms.ToTensor(),
            # DBT
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
            # Chest
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(0.4, 1.)),
            transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0, hue=0)],
                                p=0.8),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            # DBT
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
            # Chest
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])


    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

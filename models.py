import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

class Patch_Model(nn.Module):
    def __init__(self, input_channel=3):
        super(Patch_Model, self).__init__()
        
        # Model atributes
        resnet = models.wide_resnet50_2(pretrained=True)
        if input_channel != 3:
            resnet.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

        # projector
        sizes = [2048, 2048, 2048]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, x):
        embed_full = self.backbone(x)
        embed = self.pool(embed_full).squeeze()
        if x.shape[0] == 1:
            embed = embed.unsqueeze(0)
        return self.projector(embed), embed

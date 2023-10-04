import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


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

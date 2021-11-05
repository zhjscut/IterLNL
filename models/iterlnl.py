# modified from https://github.com/mil-tokyo/MCD_DA/blob/master/classification/model/usps.py
# and https://github.com/tim-learn/SHOT/blob/master/digit/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from gorilla.nn import resnet, MultiFC


# reasonable usage of Dropout(D): 
# 1. one betweens convs and fcs, such as conv - conv - D - fc - fc
# 2. after each conv, such as conv - D - conv - D - conv - D - fc
class ExtractorRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=5, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                )
        self.in_features = 256*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class PredictorRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_params = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*4*4, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.fc_params(x)
        return x


class ExtractorGray(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.BatchNorm2d(20),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, dilation=(1,1)),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.BatchNorm2d(50),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, dilation=(1,1)),
                )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class PredictorGray(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_params = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(50*4*4, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.fc_params(x)
        return x


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.dataset == "Digit":  # simpler model
            if "usps" in [cfg.source, cfg.target]:
                self.G = ExtractorGray()
                self.F = PredictorGray()
            else:
                self.G = ExtractorRGB()
                self.F = PredictorRGB()
        else:  # resnet
            if cfg.arch in ["resnet18", "resnet34"]:
                feature_dim = 512
            elif cfg.arch in ["resnet50", "resnet101"]:
                feature_dim = 2048
            else:
                raise NotImplementedError(cfg.arch)

            self.G = resnet(cfg)
            self.F = MultiFC([feature_dim, cfg.num_classes], init=None)

        if torch.cuda.is_available():
            self.G = self.G.cuda()
            self.F = self.F.cuda()

    def forward(self, data):
        # compute the category output
        cate_output = self.F(self.G(data))

        return dict(cate = cate_output)

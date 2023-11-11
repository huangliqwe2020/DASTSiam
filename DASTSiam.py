import time
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from networks.res18.resnet import resnet50
from networks.siamrpnplus.transformer import build_transformer
from networks.siamrpnplus.RPN import *
from sklearn.svm import SVC

# import torchinfo
cfg = {'d_model':256,'dropout':0.1,'nhead':4,'dim_feedforward':1024,'num_enclayer':4,
          'num_declayer':4,'nbefore':False,'divide_norm':False}


class SiamRPN(nn.Module):
    def __init__(self):
        super(SiamRPN, self).__init__()
        self.tbackbone = resnet50(last_layer='layer3')
        self.sbackbone = resnet50(last_layer='layer3')
        self.tAugmentation = TransformerAugmentation()
        self.rpnfusion = RPNfusion()

    def forward(self, examplars, search_region):

        _, _, zConv4 = self.tbackbone(
            examplars[0])
        _, _, srcConv4 = self.tbackbone(
            examplars[1])
        _, _, src1Conv4 = self.tbackbone(
            examplars[2])
        _, _, xconv4 = self.sbackbone(
            search_region)

        zb, _, _, _ = zConv4.size()

        zconvOut4,xconvOut4 = self.tAugmentation(zConv4,srcConv4,src1Conv4,xconv4)

        loc,cls1,cls2=self.rpnfusion(zconvOut4,xconvOut4,zb)

        return loc, cls1,cls2


class TransformerAugmentation(nn.Module):
    def __init__(self):
        super(TransformerAugmentation, self).__init__()
        self.transformer = build_transformer(cfg)
        self.retif2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.flag = 0
    def forward(self, zConv4,srcConv4, src1Conv4, xconv4):

        zConv4 = self.downsample(zConv4)
        # srcConv4 = self.downsample(srcConv4)
        if self.flag ==0:
            srcConv4 = self.downsample(srcConv4)
            self.flag = 1
        src1Conv4 = self.downsample(src1Conv4)
        xconv4 = self.downsample(xconv4)

        zb, zc, zw, zh = zConv4.size()

        xb, xc, xw, xh = xconv4.size()

        zconv4 = zConv4.view(zb, zc, -1).permute(2, 0, 1)

        srcConv4 = srcConv4.view(zb, zc, -1).permute(2, 0, 1)

        src1Conv4 = src1Conv4.view(zb, zc, -1).permute(2, 0, 1)

        xconv4 = xconv4.view(xb, xc, -1).permute(2, 0, 1)

        xconvOut4, zconvOut4 = self.transformer(xconv4, zconv4, srcConv4, src1Conv4, zb, zc, zw, zh)

        xconvOut4 = xconvOut4.permute(1, 2, 0).view(zb, zc, xw, xh)

        zconvOut4 = zconvOut4.permute(1, 2, 0).view(zb, zc, zw, zh)

        xconv4 = xconv4.permute(1, 2, 0).view(zb, zc, xw, xh)
        #
        xconvOut4 = self.retif2(xconvOut4) + xconv4

        return zconvOut4,xconvOut4

class RPNfusion(nn.Module):
    def __init__(self):
        super(RPNfusion, self).__init__()
        self.conv4_6_RPN = RPN()
        self.convloc = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=3, stride=1),
        )
        self.convcls = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.convcls1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, stride=1),
        )
        self.convcls2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1),
        )

    def forward(self,zconvOut4,xconvOut4,zb):
        conv4_6_cls_prediction, conv4_6_bbox_regression_prediction = self.conv4_6_RPN(
            zconvOut4, xconvOut4, zb)

        cls = self.convcls(conv4_6_cls_prediction)
        loc = self.convloc(conv4_6_bbox_regression_prediction)
        cls1 = self.convcls1(cls)
        cls2 = self.convcls2(cls)

        return loc,cls1,cls2

if __name__ =="__main__":

  model = SiamRPN().cuda()
  z = torch.randn((10,3,127,127)).cuda()
  zfs = [z,z,z]
  x = torch.randn((10,3,287,287)).cuda()
  a = model(zfs,x)
  print("qweqwe")



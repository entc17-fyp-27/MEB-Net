from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch


__all__ = ['DenseNet', 'densenet']


class DenseNet(nn.Module):

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(DenseNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        densenet_model = torchvision.models.densenet121(pretrained=True)

        self.base = nn.Sequential(densenet_model.features)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            # out_planes = densenet_model.classifier.in_features
            # print(out_planes)
            out_planes = 2048

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                #self.feat_bn_ub = nn.BatchNorm1d(1024)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                #self.feat_bn_ub = nn.BatchNorm1d(2048)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)
        #print(densenet_model.classifier.in_features)

        # if not pretrained:
        #     self.reset_params()

    def forward(self, x, feature_withbn=False):
        

        x = self.base(x)
       
        x = F.relu(x,inplace=True)
        #print(x.size()) 32,1024,8,4

        h=x.size(2)
        x_split=[]
        xd_split=[x[:, :, h // 2 * s: h // 2 * (s+1), :] for s in range(2)]
        for xx in xd_split:
            xx = self.gap(xx)
            xx = torch.cat([xx,xx], dim=1)
            x_split.append(xx.view(xx.size(0), -1))

        x = self.gap(x)
        #print(x.size()) 32,1024,1,1
        x = torch.cat([x,x], dim=1)  #####why?????!!!!!!!!
        # x = torch.cat([x, torch.zeros(x.shape[0],128,1,1).cuda()], dim=1)
        x = x.view(x.size(0), -1)
        #print(x.size()) 32,2048
        if self.cut_at_pooling:
            
            return [x,x_split[0],x_split[1]]
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
            bn_x_up = self.feat(self.feat(x_split[0]))
            bn_x_bot = self.feat(self.feat(x_split[1]))
        else:
            bn_x = self.feat_bn(x)
            bn_x_up = self.feat_bn(x_split[0])
            bn_x_bot = self.feat_bn(x_split[1])

        if self.training is False:
            bn_x = F.normalize(bn_x)
            bn_x_up = F.normalize(bn_x_up)
            bn_x_bot = F.normalize(bn_x_bot)
            #during Extraction
            #print("x : ", bn_x.size())
            #print("up : ", bn_x_up.size())
            #print("bot : ", bn_x_bot.size())

            return [bn_x,bn_x_up,bn_x_bot]
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
            bn_x_up = F.normalize(bn_x_up)
            bn_x_bot = F.normalize(bn_x_bot)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)
            bn_x_up = F.relu(bn_x_up)
            bn_x_bot = F.relu(bn_x_bot)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)
            bn_x_up = self.drop(bn_x_up)
            bn_x_bot = self.drop(bn_x_bot)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return x, bn_x

        if feature_withbn:
            return bn_x,bn_x_up,bn_x_bot, prob
        return x,x_split[0],x_split[1], prob



def densenet(**kwargs):
    return DenseNet(50, **kwargs)


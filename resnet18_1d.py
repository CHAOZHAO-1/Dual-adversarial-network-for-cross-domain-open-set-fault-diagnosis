import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torch
import itertools
import mmd
import torch.nn as nn
from utils import *
from torch.autograd import Variable
import  numpy as np
import torch.nn.functional as F




def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

class GradReverse(torch.autograd.Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)



class CNN_1D(nn.Module):

    def __init__(self, num_classes=31):
        super(CNN_1D, self).__init__()
        # self.sharedNet = resnet18(False)
        # self.cls_fc = nn.Linear(512, num_classes)

        self.sharedNet = CNN()
        self.cls_fc = nn.Linear(256, num_classes)


    def forward(self, source):

        # source= source.unsqueeze(1)

        source = self.sharedNet(source)
        source=self.cls_fc(source)
        return source



class CNN_1Dfea(nn.Module):

    def __init__(self, num_classes=31):
        super(CNN_1Dfea, self).__init__()


        # self.sharedNet = CNN()
        # self.cls_fc = nn.Linear(256, num_classes)
        # self.domain_fc = AdversarialNetwork(in_feature=256)
        # self.au_domain_fc = AdversarialNetwork(in_feature=256)

        self.sharedNet = CNN()
        self.cls_fc = nn.Linear(256, num_classes)
        self.domain_fc = AdversarialNetwork(in_feature=256)
        self.outlier_fc = AdversarialNetwork(in_feature=256)



    def forward(self, source):

        source_fea = self.sharedNet(source)
        source_lab =self.cls_fc(source_fea)

        source_lab=torch.max(source_lab, 1)[1]

        # source_domain_lab = self.au_domain_fc(source_fea)

        source_domain_lab = self.domain_fc(source_fea)

        return source_fea,source_lab,source_domain_lab




class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, num_classes=10):
        super(CNN, self).__init__()


        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64,stride=1),  # 32, 24, 24
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )  # 32, 12,12     (24-2) /2 +1


        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16,stride=1),  # 128,8,8
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))# 128, 4,4

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5,stride=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5,stride=1),  # 128,8,8
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4)
        )


        # self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        x = x.unsqueeze(1)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)

        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)


        x = x.view(x.size(0), -1)

        # x = self.layer5(x)

        # x = self.fc(x)

        return x


class Classifier(nn.Module):
    def __init__(self, in_feature,num_class):
        super(Classifier, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 128)
        self.ad_layer2 = nn.Linear(128, num_class)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)


    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        # print(x.size())
        # x = self.relu2(x)
        # x = self.dropout2(x)
        # x = self.sigmoid(x)
        return x





class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 128)
        self.ad_layer2 = nn.Linear(128, 2)
        # self.ad_layer1.weight.data.normal_(0, 0.1)
        # self.ad_layer2.weight.data.normal_(0, 0.3)
        # self.ad_layer1.bias.data.fill_(0.0)
        # self.ad_layer2.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        # print(x.size())
        # x = self.relu2(x)
        # x = self.dropout2(x)
        # x = self.softmax(x)
        return x

    def output_num(self):
        return 1




class M18(nn.Module):

    def __init__(self, num_classes=31):
        super(M18, self).__init__()

        self.sharedNet = CNN()
        self.cls_fc = nn.Linear(256, num_classes)
        self.domain_fc = AdversarialNetwork(in_feature=256)
        self.au_domain_fc = AdversarialNetwork(in_feature=256)



    def forward(self, x,constant = 1, adaption = False):

        feature = self.sharedNet(x)
        domain_pred=self.domain_fc(feature)
        au_domain_pred=self.au_domain_fc(feature)

        if adaption == True:

            x = self.cls_fc(feature)

            feature = grad_reverse(feature, constant)

            gra_x = self.cls_fc(feature)
            return gra_x, domain_pred, au_domain_pred, feature,x

        if adaption == False:
            x = self.cls_fc(feature)
            return x, domain_pred, au_domain_pred, feature



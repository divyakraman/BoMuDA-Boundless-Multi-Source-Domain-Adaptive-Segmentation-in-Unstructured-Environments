import pdb

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

BatchNorm = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class Net1(nn.Module):
    def __init__(self, inplanes = 57, planes = 19, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Net1, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, inplanes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(inplanes)

        self.seg1 = nn.Conv2d(inplanes, planes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        
        out = self.relu(out)

        out = self.seg1(out)
        return self.softmax(out)

class Net2(nn.Module):
    def __init__(self, inplanes = 57, planes = 19, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Net2, self).__init__()
        self.conv1 = conv3x3(inplanes, 64, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.conv2 = conv3x3(64, 64,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(64)
        self.conv3 = conv3x3(64, 64,
                             padding=dilation[1], dilation=dilation[1])
        self.bn3 = BatchNorm(64)


        self.seg1 = nn.Conv2d(64, planes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)

        residual = out 

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leaky_relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        
        out = self.leaky_relu(out)

        out = self.seg1(out)
        return self.softmax(out)


class Net3(nn.Module):
    def __init__(self, inplanes = 57, planes = 19, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Net3, self).__init__()
        self.conv1 = conv3x3(inplanes, 64, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.conv2 = conv3x3(64, 64,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(64)
        self.conv3 = conv3x3(64, 64,
                             padding=dilation[1], dilation=dilation[1])
        self.bn3 = BatchNorm(64)

        self.conv4 = conv3x3(64, 64,
                             padding=dilation[1], dilation=dilation[1])
        self.bn4 = BatchNorm(64)
        self.conv5 = conv3x3(64, 64,
                             padding=dilation[1], dilation=dilation[1])
        self.bn5 = BatchNorm(64)


        self.seg1 = nn.Conv2d(64, planes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)

        residual = out 

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leaky_relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += residual

        residual = out 

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.leaky_relu(out)
        out = self.conv5(out)
        out = self.bn5(out)

        out += residual

        
        out = self.leaky_relu(out)

        out = self.seg1(out)
        return self.softmax(out)

        

    

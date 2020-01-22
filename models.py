import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from ops import AdaIN
from collections import namedtuple
from copy import deepcopy


class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        vgg_features = vgg19(pretrained=True).features
        self.relu1_1 = vgg_features[:2]
        self.intr1 = vgg_features[2:4]
        self.relu2_1 = vgg_features[5:7]
        self.intr2 = vgg_features[7:9]
        self.relu3_1 = vgg_features[10:12]
        self.intr3 = vgg_features[12:18]
        self.relu4_1 = vgg_features[19:21]
        self.intr4 = vgg_features[21:27]
        self.relu5_1 = vgg_features[28:30]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input, layer=None):
        relu1_1 = self.relu1_1(input)
        if layer == 'relu1_1':
            return relu1_1
        intr1 = F.max_pool2d(self.intr1(relu1_1), kernel_size=2, stride=2, padding=0)
        relu2_1 = self.relu2_1(intr1)
        if layer == 'relu2_1':
            return relu2_1
        intr2 = F.max_pool2d(self.intr2(relu2_1), kernel_size=2, stride=2, padding=0)
        relu3_1 = self.relu3_1(intr2)
        if layer == 'relu3_1':
            return crelu3_1
        intr3 = F.max_pool2d(self.intr3(relu3_1), kernel_size=2, stride=2, padding=0)
        relu4_1 = self.relu4_1(intr3)
        if layer == 'relu4_1':
            return relu4_1
        vgg_output = namedtuple('vgg_output', ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
        return vgg_output(relu1_1, relu2_1, relu3_1, relu4_1)


vgg_decoder_relu5_1 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, 3),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, 3),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, 3),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, 3)
)


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(*list(deepcopy(vgg_decoder_relu5_1[13:]).children()))
        self.relu4_1 = self.net[:13]
        self.relu3_1 = self.net[13:20]
        self.relu2_1 = self.net[20:27]
        self.relu1_1 = self.net[27:]

    def forward(self, x, s_features):
        d_cs3 = self.relu4_1(x)
        F_sf3 = AdaIN(d_cs3, s_features.relu3_1)
        d_cs2 = self.relu3_1(F_sf3)
        F_sf2 = AdaIN(d_cs2, s_features.relu2_1)
        d_cs1 = self.relu2_1(F_sf2)
        F_sf1 = AdaIN(d_cs1, s_features.relu1_1)
        output = self.relu1_1(F_sf1)
        return output

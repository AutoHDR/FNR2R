import torch
import torch.nn as nn
from torchvision.models import vgg13_bn
import math
import torch.nn.functional as F
import torchvision.models as models
import copy

__all__ = ['vgg13bn_unet', 'vgg16bn_unet']


class SDFT(nn.Module):

    def __init__(self, color_dim, channels, kernel_size = 3):
        super().__init__()
        
        # generate global conv weights
        fan_in = channels * kernel_size ** 2
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.scale = 1 / math.sqrt(fan_in)
        self.modulation = nn.Conv2d(color_dim, channels, 1)
        self.weight = nn.Parameter(torch.randn(1, channels, channels, kernel_size, kernel_size))

    def forward(self, fea, color_style):
        # for global adjustation
        B, C, H, W = fea.size()
        # print(fea.shape, color_style.shape)
        style = self.modulation(color_style).view(B, 1, C, 1, 1)
        weight = self.scale * self.weight * style
        # demodulation
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(B, C, 1, 1, 1)

        weight = weight.view(
            B * C, C, self.kernel_size, self.kernel_size
        )

        fea = fea.view(1, B * C, H, W)
        fea = F.conv2d(fea, weight, padding=self.padding, groups=B)
        fea = fea.view(B, C, H, W)

        return fea
    
class UpBlock(nn.Module):
    def __init__(self, color_dim, in_channels, out_channels, kernel_size = 3, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)     
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels // 2 + in_channels // 8, out_channels, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.conv_s = nn.Conv2d(in_channels//2, out_channels, 1, 1, 0)
        # generate global conv weights
        self.SDFT = SDFT(color_dim, out_channels, kernel_size)

    def forward(self, x1, x2, color_style):
        # print(x1.shape, x2.shape, color_style.shape)
        xc1 = self.up(x1)
        x1_s = self.conv_s(xc1)

        x = torch.cat([xc1, x2[:, ::4, :, :]], dim=1)
        x = self.conv_cat(x)
        x = self.SDFT(x, color_style)
        x = x + x1_s
        return x

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class VGGUnetN256(nn.Module):
    """Unet with VGG-13 (with BN), VGG-16 (with BN) encoder.
    """
    def __init__(self, encoder, *, pretrained=False, out_channels=2):
        super().__init__()
        self.encoder = encoder(pretrained=pretrained).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])
        self.bottleneck = nn.Sequential(*self.encoder[34:])

        self.conv_bottleneck = double_conv(512, 512)

        self.NormFeat = NormalEncoder_FPN256()

        self.up1 = UpBlock(256, 1024, 512, 3, True)
        self.up2 = UpBlock(256, 1024, 256, 3, True)
        self.up3 = UpBlock(256, 512, 128, 3, True)
        self.up4 = UpBlock(256, 256, 64, 3, True)
        self.up5 = UpBlock(256, 128, 64, 3, True)
        self.conv11 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, exemplarNorm):
        normfeat = self.NormFeat(exemplarNorm)

        block1 = self.block1(x) #64*256*256
        block2 = self.block2(block1) #128*128*128
        block3 = self.block3(block2) #256*64*64
        block4 = self.block4(block3) #512*32*32
        block5 = self.block5(block4) #512*16*16
        bottleneck = self.bottleneck(block5) #512*8*8
        
        feat_1 = self.conv_bottleneck(bottleneck) #512*8*8

        x6 = self.up1(feat_1, block5, normfeat) # [B, 512, 16, 16]
        x7 = self.up2(x6, block4, normfeat) # [B, 512, 16, 16]
        x8 = self.up3(x7, block3, normfeat) # [B, 512, 16, 16]
        x9 = self.up4(x8, block2, normfeat) # [B, 512, 16, 16]
        x10 = self.up5(x9, block1, normfeat) # [B, 512, 16, 16]

        output = self.conv11(x10)
        return torch.tanh(output)

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class NormalEncoder_FPN256(nn.Module):
    def __init__(self, color_dim=3, out_dim=256):
        super(NormalEncoder_FPN256, self).__init__()
        ChanFeatures = 64
        self.conv0 = nn.Sequential(
                        BasicConv(color_dim, ChanFeatures, kernel_size=3, stride=1, relu=True),
                        BasicConv(ChanFeatures, ChanFeatures, kernel_size=3, stride=1, relu=True))
        self.conv1 = nn.Sequential(
                        BasicConv(ChanFeatures, ChanFeatures, kernel_size=5, stride=2, relu=True),
                        double_conv(ChanFeatures, ChanFeatures))
        self.conv2 = nn.Sequential(
                        BasicConv(ChanFeatures, ChanFeatures, kernel_size=5, stride=2, relu=True),
                        double_conv(ChanFeatures, ChanFeatures))
        self.conv3 = nn.Sequential(
                        BasicConv(ChanFeatures, ChanFeatures, kernel_size=5, stride=2, relu=True),
                        double_conv(ChanFeatures, ChanFeatures))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))#, # 1x1
        self.toplayer = nn.Conv2d(ChanFeatures, out_dim, 1)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv0(x) # (B, 128, 512, 512)
        x = self.conv1(x) # (B, 128, 256, 256)
        x = self.conv2(x) # (B, 128, 128, 128)
        x = self.conv3(x) # (B, 128, 64,  64)
        x = self.pool(x)  # (B, 128, 1,   1)
        x = self.toplayer(x) # (B, 512, 1, 1)
        return x



class UShapedNet(nn.Module):
    # initializers
    def __init__(self, inputdim=3, d=64):
        super(UShapedNet, self).__init__()
        # Unet encoder
        self.conv1 = nn.Conv2d(inputdim, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        # self.conv8_bn = nn.BatchNorm2d(d * 8)

        # Unet decoder
        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        e8 = self.conv8(F.leaky_relu(e7, 0.2))
        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        o = torch.tanh(d8)

        return o

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()



def vgg13bn_unet256(output_dim: int=3, pretrained: bool=False):
    return VGGUnetN256(vgg13_bn, pretrained=pretrained, out_channels=output_dim)




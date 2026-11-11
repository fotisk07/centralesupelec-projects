import numpy as np
import torch
from torch.nn import Module, Conv2d, ReLU, BatchNorm2d, MaxPool2d, ConvTranspose2d


class conv_block(Module):
    def __init__(self, in_c, out_c, config):
        super().__init__()
        self.conv1 = Conv2d(in_c, out_c, kernel_size=3,
                            padding=1)
        self.bn1 = BatchNorm2d(out_c)
        self.conv2 = Conv2d(out_c, out_c, kernel_size=3,
                            padding=1)
        self.bn2 = BatchNorm2d(out_c)
        self.relu = ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class encoder_block(Module):
    def __init__(self, in_c, out_c, config):
        super().__init__()
        self.conv = conv_block(in_c, out_c, config)
        self.pool = MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(Module):
    def __init__(self, in_c, out_c, config):
        super().__init__()
        self.up = ConvTranspose2d(
            in_c, out_c, kernel_size=2, stride=2, padding=0, device=config.device)
        self.conv = conv_block(3*out_c, out_c)

    def forward(self, inputs, skip1, skip2):
        x = self.up(inputs)
        x = torch.cat([x, skip1, skip2], axis=1)
        x = self.conv(x)
        return x


class Unet(Module):
    def __init__(self, config):
        super().__init__()

        self.e11 = conv_block(1, 32, config)
        self.e21 = conv_block(1, 32, config)

        self.e12 = conv_block(32, 64, config)
        self.e22 = conv_block(32, 64, config)

        self.bottle_neck = conv_block(128, 512, config)

        self.d1 = conv_block(384, 192, config)
        self.d2 = conv_block(128, 64, config)

        self.outputs = conv_block(64, 1, config)

        self.max1 = MaxPool2d((2, 2))
        self.max2 = MaxPool2d((2, 2))

        self.up1 = ConvTranspose2d(512, 256, kernel_size=2,
                                   stride=2, padding=0)

        self.up2 = ConvTranspose2d(
            192, 64, kernel_size=2, stride=2, padding=0)

    def forward(self, input1, input2):

        x11 = self.e11(input1)
        x21 = self.e21(input2)

        x11_pool = self.max1(x11)
        x21_pool = self.max2(x21)

        x12 = self.e12(x11_pool)
        x22 = self.e22(x21_pool)

        x12_pool = self.max1(x12)
        x22_pool = self.max1(x22)

        x3 = self.bottle_neck(torch.cat([x12_pool, x22_pool], dim=1))

        upi = self.up1(x3)

        x4 = torch.cat([x12, x22, self.up1(x3)], dim=1)

        x5 = self.d1(x4)

        upi = self.up2(x5)
        x6 = torch.cat([x11, x21, self.up2(x5)], dim=1)

        x7 = self.d2(x6)

        return self.outputs(x7).permute(0, 2, 3, 1)


def get_model(config):
    return Unet(config)

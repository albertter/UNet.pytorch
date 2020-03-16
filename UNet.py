import torch.nn as nn
import torch
from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.conv1 = DoubleConv(in_channels, 64)

        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)

        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)

        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)

        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride = 2)
        self.conv6 = DoubleConv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, out_channels, kernel_size = 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        x5 = self.conv5(self.pool4(x4))

        x6 = self.up6(x5)
        x6 = self.conv6(torch.cat([x6, x4], dim = 1))
        #
        x7 = self.up7(x6)
        x7 = self.conv7(torch.cat([x7, x3], dim = 1))

        x8 = self.up8(x7)
        x8 = self.conv8(torch.cat([x8, x2], dim = 1))

        x9 = self.up9(x8)
        x9 = self.conv9(torch.cat([x9, x1], dim = 1))
        x10 = self.conv10(x9)

        return x10


if __name__ == '__main__':
    model = UNet(in_channels = 3, out_channels = 2)
    # x = torch.randn(1, 1, 572, 572)
    #
    # y=model(x)
    # print(y.shape)
    summary(model, input_size = (3, 512, 512))

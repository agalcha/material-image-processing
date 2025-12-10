import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (conv => BN => ReLU) * 2
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then DoubleConv
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c, out_c),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upscaling then DoubleConv.

    Assumes:
      - input x1 has in_c channels
      - after ConvTranspose2d, x1 has in_c // 2 channels
      - skip connection x2 has in_c // 2 channels
      - after concatenation, channels = in_c
      - DoubleConv(in_c, out_c) reduces to out_c channels
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # up-conv: in_c -> in_c // 2
        self.up = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)
        # conv: (in_c // 2 + skip_channels) == in_c
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x1, x2):
        # x1: from deeper layer
        # x2: from skip connection
        x1 = self.up(x1)

        # handle any mismatch in spatial size due to pooling/conv rounding
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2]
        )

        # concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)   

        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)    # 64 channels
        x2 = self.down1(x1) # 128
        x3 = self.down2(x2) # 256
        x4 = self.down3(x3) # 512
        x5 = self.down4(x4) # 1024

        # Decoder path with skip connections
        x = self.up1(x5, x4)  # 1024 -> 512
        x = self.up2(x,  x3)  # 512  -> 256
        x = self.up3(x,  x2)  # 256  -> 128
        x = self.up4(x,  x1)  # 128  -> 64

        logits = self.outc(x) 
        return logits

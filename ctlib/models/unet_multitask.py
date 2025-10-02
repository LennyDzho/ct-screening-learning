import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )

class UNetMultiTask(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(base*4, base*8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = conv_block(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = conv_block(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_block(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = conv_block(base*2, base)

        # выход №1: карта локализации (heatmap)
        self.head_heat = nn.Conv2d(base, 1, 1)

        # выход №2: классификация (злокачественность)
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc = nn.Linear(base*16, 1)

    def forward(self, x):
        e1 = self.enc1(x)          # B,base,H,W
        p1 = self.pool1(e1)        # H/2
        e2 = self.enc2(p1)         # base*2
        p2 = self.pool2(e2)        # H/4
        e3 = self.enc3(p2)         # base*4
        p3 = self.pool3(e3)        # H/8
        e4 = self.enc4(p3)         # base*8
        p4 = self.pool4(e4)        # H/16
        bn = self.bottleneck(p4)   # base*16

        u4 = self.up4(bn)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        heat = self.head_heat(d1)

        # cls из bottleneck
        g = self.cls_pool(bn).flatten(1)
        cls_logit = self.cls_fc(g)

        return {
            "heat": heat,             # B,1,H,W
            "cls_logit": cls_logit,   # B,1
        }

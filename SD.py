import torch
import torch.nn as nn
from FCSN_ENC import FCSN_ENC



class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class SD_test(nn.Module):
    def __init__(self):
        super(SD_test, self).__init__()
        self.inc = inconv(1024, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.linear = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        h = x
        x1 = self.inc(x);       #print(x1.shape)
        x2 = self.down1(x1);    #print(x2.shape)
        x3 = self.down2(x2);    #print(x3.shape)
        x4 = self.down3(x3);    #print(x4.shape)
        x5 = self.down4(x4);    #print(x5.shape)

        h = nn.AvgPool2d((1,h.size()[3]), stride=(1,h.size()[3]), ceil_mode=True)(x5)

        h = h.view(1, -1)
        
        h = self.linear(h)

        h = self.sigmoid(h).view(-1)

        return h



class SD(nn.Module):
    def __init__(self):
        super(SD, self).__init__()

        self.FCSN_ENC = FCSN_ENC()
        self.linear = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x

        h = self.FCSN_ENC(h); print(h.shape)

        h = nn.AvgPool2d((1,h.size()[3]), stride=(1,h.size()[3]), ceil_mode=True)(h)

        h = h.view(1, -1)
        
        h = self.linear(h)

        h = self.sigmoid(h).view(-1)

        return h

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SD_test()
    model.to(device)
    model.eval()

    inp = torch.randn(1, 1024, 1, 245, requires_grad=True).to(device)
    #mask = torch.randn(1, 1, 1, 2).to(device); print(mask)

    #inp_view = inp.view(1,3,2); print(inp_view)
    #mask_view = mask.view(1,1,2); print(mask_view)

    #print(inp_view*mask_view)

    #scalar = torch.randn(1); print(scalar)
    #print(torch.mean(scalar))


    out = model(inp)
    #print(out.shape)
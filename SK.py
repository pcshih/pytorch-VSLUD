import torch
import torch.nn as nn
import torch.nn.functional as F
#from FCSN import FCSN

import random


class SK(nn.Module):
    def __init__(self, n_class=2):
        super(SK, self).__init__()
        # conv1 input shape (batch_size, Channel, H, W) -> (1,1024,1,T)
        self.conv1_1 = nn.Conv2d(1024, 64, (1,3), padding=(0,100))
        self.sn1_1 = nn.utils.spectral_norm(self.conv1_1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.conv1_2 = nn.Conv2d(64, 64, (1,3), padding=(0,1))
        self.sn1_2 = nn.utils.spectral_norm(self.conv1_2)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.pool1 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, (1,3), padding=(0,1))
        self.sn2_1 = nn.utils.spectral_norm(self.conv2_1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.conv2_2 = nn.Conv2d(128, 128, (1,3), padding=(0,1))
        self.sn2_2 = nn.utils.spectral_norm(self.conv2_2)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.pool2 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, (1,3), padding=(0,1))
        self.sn3_1 = nn.utils.spectral_norm(self.conv3_1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.conv3_2 = nn.Conv2d(256, 256, (1,3), padding=(0,1))
        self.sn3_2 = nn.utils.spectral_norm(self.conv3_2)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.conv3_3 = nn.Conv2d(256, 256, (1,3), padding=(0,1))
        self.sn3_3 = nn.utils.spectral_norm(self.conv3_3)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.pool3 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, (1,3), padding=(0,1))
        self.sn4_1 = nn.utils.spectral_norm(self.conv4_1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.conv4_2 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.sn4_2 = nn.utils.spectral_norm(self.conv4_2)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.conv4_3 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.sn4_3 = nn.utils.spectral_norm(self.conv4_3)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.pool4 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.sn5_1 = nn.utils.spectral_norm(self.conv5_1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.conv5_2 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.sn5_2 = nn.utils.spectral_norm(self.conv5_2)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.conv5_3 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.sn5_3 = nn.utils.spectral_norm(self.conv5_3)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.pool5 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, (1,7))
        self.sn6 = nn.utils.spectral_norm(self.fc6)
        self.in6 = nn.InstanceNorm2d(4096)
        self.relu6 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.drop6 = nn.Dropout2d(p=0.5)

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, (1,1))
        self.sn7 = nn.utils.spectral_norm(self.fc7)
        self.in7 = nn.InstanceNorm2d(4096)
        self.relu7 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.drop7 = nn.Dropout2d(p=0.5)

        self.score_fr = nn.Conv2d(4096, n_class, (1,1))
        self.sn_score_fr = nn.utils.spectral_norm(self.score_fr)
        self.bn_score_fr = nn.BatchNorm2d(n_class)
        self.in_score_fr = nn.InstanceNorm2d(n_class)
        self.relu_score_fr = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.score_pool4 = nn.Conv2d(512, n_class, (1,1))
        self.sn_score_pool4 = nn.utils.spectral_norm(self.score_pool4)
        self.bn_score_pool4 = nn.BatchNorm2d(n_class)
        self.relu_bn_score_pool4 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, (1,4), stride=(1,2))
        self.sn_upscore2 = nn.utils.spectral_norm(self.upscore2)
        self.bn_upscore2 = nn.BatchNorm2d(n_class)
        self.relu_upscore2 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)

        self.upscore16 = nn.ConvTranspose2d(
            n_class, n_class, (1,32), stride=(1,16))
        self.sn_upscore16 = nn.utils.spectral_norm(self.upscore16)
        self.bn_upscore16 = nn.BatchNorm2d(n_class)
        self.relu_upscore16 = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2)
        self.sigmoid_upscore16 = nn.Sigmoid()
        self.tanh_upscore16 = nn.Tanh()

        self.relu_add = nn.ReLU()#nn.LeakyReLU(0.2)

        self.softmax = nn.Softmax(dim=1)

        self.conv_reconstuct1 = nn.Conv2d(n_class, 1024, (1,1))
        self.bn_reconstruct1 = nn.BatchNorm2d(1024)
        self.relu_reconstuct1 = nn.ReLU(inplace=True)

        self.conv_reconstuct2 = nn.Conv2d(1024, 1024, (1,1))
        self.bn_reconstruct2 = nn.BatchNorm2d(1024)
        self.relu_reconstuct2 = nn.ReLU(inplace=True)


    def forward(self, x):
        # input
        h = x
        in_x = x
        # conv1
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))       #;print(h.shape)
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))       #;print(h.shape)
        h = self.pool1(h)                                   #;print(h.shape)
        # conv2
        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))       #;print(h.shape)
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))       #;print(h.shape)
        h = self.pool2(h)                                   #;print(h.shape)
        # conv3
        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))       #;print(h.shape)
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))       #;print(h.shape)
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))       #;print(h.shape)
        h = self.pool3(h)                                   #;print(h.shape)
        # conv4
        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))       #;print(h.shape)
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))       #;print(h.shape)
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))       #;print(h.shape)
        h = self.pool4(h)                                   #;print(h.shape)
        pool4 = h
        # conv5
        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))       #;print(h.shape)
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))       #;print(h.shape)
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))       #;print(h.shape)
        h = self.pool5(h)                                   #;print(h.shape)
        # conv6
        h = self.relu6(self.fc6(h))                         #;print(h.shape)
        h = self.drop6(h)                                   #;print(h.shape)
        # conv7
        h = self.relu7(self.fc7(h))                         #;print(h.shape)
        h = self.drop7(h)                                   #;print(h.shape)
        # conv8
        h = self.in_score_fr(self.score_fr(h)) # original should be bn_score_fr, in order to handle the one frame input i.e. [1,1024,1,1] input
        # deconv1
        h = self.upscore2(h)
        upscore2 = h
        # get score_pool4c to do skip connection
        h = self.bn_score_pool4(self.score_pool4(pool4))
        h = h[:, :, :, 5:5+upscore2.size()[3]]
        score_pool4c = h
        # skip connection
        h = upscore2+score_pool4c
        # deconv2
        h = self.upscore16(h)
        h = h[:, :, :, 27:27+x.size()[3]]; #print("before softmax:", h)

        # h
        h_softmax = self.softmax(h); #print("after softmax:", h_softmax)

        # get simulated 0/1 vector
        mask = h_softmax[:,1,:].view(1,1,1,-1); #print("mask:", mask) # [1,1,1,T] use key frame score to be the mask

        h_mask = h*mask; #print("h_mask:", h_mask)

        h_reconstruct = self.relu_reconstuct1(self.bn_reconstruct1(self.conv_reconstuct1(h_mask))) # [1,1024,1,T]
        x_select = in_x*mask

        # merge with input features
        h_merge = h_reconstruct + x_select # [1,1024,1,T]
        h_merge_reconstruct = self.relu_reconstuct2(self.bn_reconstruct2(self.conv_reconstuct2(h_merge))) # [1,1024,1,T]
        

        return h_merge_reconstruct,mask,h   # [1,1024,1,T],[1,1,1,T],[1,2,1,T]

class SK_old(nn.Module):
    def __init__(self):
        super(SK_old, self).__init__()

        self.FCSN = FCSN(n_class=2)

        self.conv_1 = nn.Conv2d(2, 1024, (1,1))
        self.batchnorm_1 = nn.BatchNorm2d(1024)
        self.relu_1 = nn.ReLU(inplace=True)
        
        self.tanh_h_select = nn.Tanh()
        self.relu_summary = nn.ReLU(inplace=True)#nn.RReLU()
        self.tanh_summary = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.conv_reconstuct2 = nn.Conv2d(1024, 1024, (1,1))
        self.bn_reconstruct2 = nn.BatchNorm2d(1024)
        self.relu_reconstuct2 = nn.ReLU(inplace=True)


    def forward(self, x):
        h = x   # [1,1024,1,T]
        x_temp = x  # [1,1024,1,T]

        h = self.FCSN(h); print(h) # [1,2,1,T]

        ###old###
        # values, indices = h.max(1, keepdim=True)
        # # 0/1 vector, we only want key(indices=1) frame
        # column_mask = (indices==1).view(-1).nonzero().view(-1).tolist() 

        # # if S_K doesn't select more than one element, then random select two element(for the sake of diversity loss)
        # if len(column_mask)<2:
        #     print("S_K does not select anything, give a random mask with 2 elements")
        #     column_mask = random.sample(list(range(h.shape[3])), 2)

        # index = torch.tensor(column_mask, device=torch.device('cuda:0'))
        # h_select = torch.index_select(h, 3, index)
        # x_select = torch.index_select(x_temp, 3, index)
        ###old###

        ###new###
        #index_mask = self.sigmoid(h[:,1]-h[:,0]).view(1,1,1,-1)
        diverse_h = h*100
        h_softmax = self.softmax(diverse_h) # [1,2,1,T]
        index_mask = h_softmax[:,1,:].view(1,1,1,-1)
        #index_mask = self.sigmoid(h[:,1]-h[:,0]).view(1,1,1,-1)
        #index_mask = (indices==1).type(torch.float32)
        # if S_K doesn't select more than one element, then random select two element(for the sake of diversity loss)
        # if (len(index_mask.view(-1).nonzero().view(-1).tolist()) < 2):
        #     print("S_K does not select anything, give a random mask with 2 elements")
        #     index_mask = torch.zeros([1,1,1,h.shape[3]], dtype=torch.float32, device=torch.device('cuda:0'))
        #     for idx in random.sample(list(range(h.shape[3])), 2):
        #         index_mask[:,:,:,idx] = 1.0

        h_select = h*index_mask
        x_select = x_temp*index_mask
        ###new###


        h_select = self.relu_1(self.conv_1(h_select))


        summary = x_select+h_select

        summary = self.relu_reconstuct2(self.bn_reconstruct2(self.conv_reconstuct2(summary))) # [5,1024,1,320]

        #summary = self.relu_summary(summary)
        
        #return summary,column_mask
        return summary,index_mask


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

class SK_test(nn.Module):
    def __init__(self, n_channels=1024, n_classes=2):
        super(SK_test, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.softmax = nn.Softmax(dim=1)


        self.conv_reconstuct1 = nn.Conv2d(n_classes, 1024, (1,1))
        self.bn_reconstruct1 = nn.BatchNorm2d(1024)
        self.relu_reconstuct1 = nn.ReLU(inplace=True)

        self.conv_reconstuct2 = nn.Conv2d(1024, 1024, (1,1))
        self.bn_reconstruct2 = nn.BatchNorm2d(1024)
        self.relu_reconstuct2 = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x

        x1 = self.inc(x);       #print(x1.shape)
        x2 = self.down1(x1);    #print(x2.shape)
        x3 = self.down2(x2);    #print(x3.shape)
        x4 = self.down3(x3);    #print(x4.shape)
        x5 = self.down4(x4);    #print(x5.shape)
        x = self.up1(x5, x4);   #print(x.shape)
        x = self.up2(x, x3);    #print(x.shape)
        x = self.up3(x, x2);    #print(x.shape)
        x = self.up4(x, x1);    #print(x.shape)
        x = self.outc(x);       #print(x.shape)

        h_softmax = self.softmax(x)

        mask = h_softmax[:,1,:].view(1,1,1,-1); #print("mask:", mask) # [1,1,1,T] use key frame score to be the mask

        h_mask = x*mask; #print("h_mask:", h_mask)

        h_reconstruct = self.relu_reconstuct1(self.bn_reconstruct1(self.conv_reconstuct1(h_mask))) # [1,1024,1,T]
        x_select = h*mask

        # # merge with input features
        h_merge = h_reconstruct + x_select # [1,1024,1,T]
        h_merge_reconstruct = self.relu_reconstuct2(self.bn_reconstruct2(self.conv_reconstuct2(h_merge))) # [1,1024,1,T]


        return h_merge_reconstruct,mask,x   # [1,1024,1,T],[1,1,1,T],[1,2,1,T]
         


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = SK_test(1024, 2)
    #model.eval()
    model.to(device)
    inp = torch.randn(1, 1024, 1, 100, requires_grad=True).to(device)

    a,b,c = model(inp)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    #print(out.shape)
    #print(out)
    #print(mask)


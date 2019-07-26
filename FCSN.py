import torch
import torch.nn as nn

class FCSN(nn.Module):
    def __init__(self, n_class=2):
        super(FCSN, self).__init__()

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

    def forward(self, x):
        # input
        h = x
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
        h = h[:, :, :, 27:27+x.size()[3]].contiguous()

        return h

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = FCSN(n_class=2)
    model.to(device)
    model.eval()
    data = torch.randn(1, 1024, 1, 1).to(device) # [1,1024,1,input_frames] 
    out = model(data)
    print(out.shape) #[1,2,1,input_frames]
    
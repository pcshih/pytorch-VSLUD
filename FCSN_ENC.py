import torch
import torch.nn as nn

class FCSN_ENC(nn.Module):

    def __init__(self):
        super(FCSN_ENC, self).__init__()
        # conv1 (input shape (batch_size X Channel X H X W))
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

        return h


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = FCSN_ENC()
    model.to(device)
    #model.eval()

    inp = torch.randn(1, 1024, 1, 1).to(device)
    out = model(inp)
    print(out.shape)
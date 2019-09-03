import torch
import torch.nn as nn
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

        # h*100 getting larger differences(tricks!!)
        h_softmax = self.softmax(h*100); #print("after softmax:", h_softmax)

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


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SK()
    #model.eval()
    model.to(device)
    inp = torch.randn(1, 1024, 1, 5, requires_grad=True).to(device)

    summary,mask,_ = model(inp)
    print(summary.shape)
    print(mask)


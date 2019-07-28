import torch
import torch.nn as nn
from FCSN import FCSN

import random

class SK(nn.Module):
    def __init__(self):
        super(SK, self).__init__()

        self.FCSN = FCSN(n_class=2)

        self.conv_1 = nn.Conv2d(2, 1024, (1,1))
        self.batchnorm_1 = nn.BatchNorm2d(1024)
        self.relu_1 = nn.ReLU(inplace=True)
        
        self.tanh_h_select = nn.Tanh()
        self.relu_summary = nn.ReLU(inplace=True)#nn.RReLU()
        self.tanh_summary = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        h = x
        x_temp = x

        h = self.FCSN(h)

        

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
        index_mask = self.sigmoid(h[:,1]-h[:,0]).view(1,1,1,-1)
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

        summary = self.relu_summary(summary)
        
        #return summary,column_mask
        return summary,index_mask         


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SK()
    #model.eval()
    model.to(device)
    inp = torch.randn(1, 1024, 1, 5).to(device)

    summary,mask = model(inp)
    print(summary.shape)
    print(mask)


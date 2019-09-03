import torch
import torch.nn as nn
from FCSN_ENC import FCSN_ENC


class SD(nn.Module):
    def __init__(self):
        super(SD, self).__init__()

        self.FCSN_ENC = FCSN_ENC()
        self.linear = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x

        h = self.FCSN_ENC(h); 

        h = nn.AvgPool2d((1,h.size()[3]), stride=(1,h.size()[3]), ceil_mode=True)(h)

        h = h.view(1, -1)
        
        h = self.linear(h)

        h = self.sigmoid(h).view(-1)

        return h

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #model = SD()
    #model.to(device)
    #model.eval()

    inp = torch.randn(1, 3, 1, 2, requires_grad=True).to(device); print(inp)
    mask = torch.randn(1, 1, 1, 2).to(device); print(mask)

    inp_view = inp.view(1,3,2); print(inp_view)
    mask_view = mask.view(1,1,2); print(mask_view)

    print(inp_view*mask_view)

    scalar = torch.randn(1); print(scalar)
    print(torch.mean(scalar))


    #out = model(inp)
    #print(out.shape)
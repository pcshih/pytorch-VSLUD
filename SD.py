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

    model = SD()
    model.to(device)
    #model.eval()

    inp = torch.randn(1, 1024, 1, 1).to(device)
    out = model(inp)
    print(out.shape)
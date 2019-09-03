import torch
import torch.nn as nn
import torchvision
# torchvision0.2.2
#from googlenet import googlenet
import time

class FeatureExtractor(nn.Module):
    def __init__(self):
        # supposed input format(N,C,L) C:#features L:#frames
        super(FeatureExtractor, self).__init__()
        # GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # torchvision0.3.0
        self.googlenet = torchvision.models.googlenet(pretrained=True)
        # use eval mode to do feature extraction
        self.googlenet.eval()

        # we only want features no grads
        for param in self.googlenet.parameters():
            param.requires_grad = False

        # feature extractor
        self.model = nn.Sequential(*list(self.googlenet.children())[:-2])

        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def forward(self, x):
        # put data in to device
        x = x.to(self.device)

        h = self.model(x)

        h = h.view(h.size()[0],1024)
        h = h.transpose(1,0)

        return h



if __name__ == '__main__':
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net = FeatureExtractor()
    
        
    #net = nn.DataParallel(net)
    
    #net.to(device)
    
    #data = torch.randn((20, 3, 299, 299)) # (N,C,299,299) inceptionv3 input otherwise (N,C,224,224)
    data = torch.randn((616, 3, 224, 224))
    #tic = time.time()
    #data = data.to(device)
    result = net(data)
    print(result.requires_grad)
    #toc = time.time()

    #print(toc-tic) GPU:0.1sec CPU:15.4sec
    
    #print(net)
    #print(net(data).size())
    #print(net(data))
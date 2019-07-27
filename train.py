import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from tensorboardX import SummaryWriter

import time
import tqdm
import random
from SK import SK
from SD import SD

random.seed(time.time())


print("loading training data...")
video = torch.load("datasets/reorganized_training_dataset_summe_video.tar")
summary = torch.load("datasets/reorganized_training_dataset_summe_summary.tar")
print("loading training data ended")

PATH_record = "loss_record.tar"
PATH_model = "saved_models"

EPOCH = 700

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ref: https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
def weights_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        #init.xavier_normal_(m.weight.data)
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        #init.xavier_normal_(m.weight.data)
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


S_K = SK().to(device)
S_D = SD().to(device)


optimizerS_K = optim.Adam(S_K.parameters(), lr=0.00001)
optimizerS_D = optim.SGD(S_D.parameters(), lr=0.0002)


# configure training record
writer = SummaryWriter()


# mode=0 -> first train
# mode=1 -> continue train
mode = 0

if mode==0:
    print("first train")

    time_list = []
    S_K_iter_loss_list = []
    reconstruct_iter_loss_list = []
    diversity_iter_loss_list = []
    S_D_real_iter_loss_list = []
    S_D_fake_iter_loss_list = []
    S_D_total_iter_loss_list = []
    
    #S_K.apply(weights_init)
    #S_D.apply(weights_init)
    S_K.train()
    S_D.train()
elif mode==1:
    print("continue train")
    checkpoint_loss = torch.load(PATH_record)
    time_list = checkpoint_loss['time_list']; #print(time_list)

    iteration = len(time_list)-1
    PATH_model_load = "{}{}{:0>7d}{}".format(PATH_model, "/iter_", iteration, ".tar"); #print(PATH_model_load)
    checkpoint_model = torch.load(PATH_model_load)
    S_K.load_state_dict(checkpoint_model['S_K_state_dict'])
    S_D.load_state_dict(checkpoint_model['S_D_state_dict'])
    optimizerS_K.load_state_dict(checkpoint_model['optimizerS_K_state_dict'])
    optimizerS_D.load_state_dict(checkpoint_model['optimizerS_D_state_dict'])
    S_K.train()
    S_D.train()

    S_K_iter_loss_list = checkpoint_loss['S_K_iter_loss_list']
    reconstruct_iter_loss_list = checkpoint_loss['reconstruct_iter_loss_list']
    diversity_iter_loss_list = checkpoint_loss['diversity_iter_loss_list']
    S_D_real_iter_loss_list = checkpoint_loss['S_D_real_iter_loss_list']
    S_D_fake_iter_loss_list = checkpoint_loss['S_D_fake_iter_loss_list']
    S_D_total_iter_loss_list = checkpoint_loss['S_D_total_iter_loss_list']

    
    # draw previous loss
    for idx in range(len(time_list)):
        writer.add_scalar("loss/S_K", S_K_iter_loss_list[idx], idx, time_list[idx])
        writer.add_scalar("loss/reconstruction", reconstruct_iter_loss_list[idx], idx, time_list[idx]) 
        writer.add_scalar("loss/diversity", diversity_iter_loss_list[idx], idx, time_list[idx]) 
        writer.add_scalar("loss/S_D_real", S_D_real_iter_loss_list[idx], idx, time_list[idx])
        writer.add_scalar("loss/S_D_fake", S_D_fake_iter_loss_list[idx], idx, time_list[idx])
        writer.add_scalar("loss/S_D_total", S_D_total_iter_loss_list[idx], idx, time_list[idx])
else:
    print("please select mode 0 or 1")



criterion = nn.BCELoss()


for epoch in range(EPOCH):
    # random feature index
    random.shuffle(video["feature"])
    random.shuffle(summary["feature"])


    tqdm_range = tqdm.trange(len(video["feature"]))
    for i in tqdm_range: # video["feature"] -> [[1,1024,1,A], [1,1024,1,B]...]
        tqdm_range.set_description(" Epoch: {:0>5d}, Running current iter {:0>3d} ...".format(epoch+1, i+1))

        vd = video["feature"][i]
        sd = summary["feature"][i]

        ##############
        # update S_K #
        ##############
        S_K.zero_grad()

        #S_K_summary,column_mask = S_K(vd)
        S_K_summary,index_mask = S_K(vd)
        output = S_D(S_K_summary)
        label = torch.full((1,), 1, device=device)

        # adv. loss
        errS_K = criterion(output, label)

        ###old reconstruct###
        # index = torch.tensor(column_mask, device=device)
        # select_vd = torch.index_select(vd, 3, index)
        # reconstruct_loss = torch.norm(S_K_summary-select_vd, p=2)**2
        # reconstruct_loss /= len(column_mask)
        ###old reconstruct###

        ###new reconstruct###
        reconstruct_loss = torch.sum((S_K_summary-vd)**2 * index_mask) / torch.sum(index_mask)
        ###new reconstruct###


        # diversity
        S_K_summary_reshape = S_K_summary.view(S_K_summary.shape[1], S_K_summary.shape[3])
        norm_div = torch.norm(S_K_summary_reshape, 2, 0, True)
        S_K_summary_reshape = S_K_summary_reshape/norm_div
        loss_matrix = S_K_summary_reshape.transpose(1, 0).mm(S_K_summary_reshape)
        diversity_loss = loss_matrix.sum() - loss_matrix.trace()
        #diversity_loss = diversity_loss/len(column_mask)/(len(column_mask)-1)
        diversity_loss = diversity_loss/(torch.sum(index_mask))/(torch.sum(index_mask)-1)

        S_K_total_loss = errS_K+reconstruct_loss+diversity_loss # for summe dataset beta=1
        S_K_total_loss.backward()

        # update
        optimizerS_K.step()

        ##############
        # update S_D #
        ##############
        S_D.zero_grad()

        # real summary #
        output = S_D(sd)
        label.fill_(1)
        err_S_D_real = criterion(output, label)
        err_S_D_real.backward()

        # fake summary #
        S_K_summary,_ = S_K(vd)
        output = S_D(S_K_summary.detach())
        label.fill_(0)
        err_S_D_fake = criterion(output, label)
        err_S_D_fake.backward()

        S_D_total_loss = err_S_D_real+err_S_D_fake

        optimizerS_D.step()
        
        # record
        time_list.append(time.time())
        S_K_iter_loss_list.append(errS_K)
        reconstruct_iter_loss_list.append(reconstruct_loss)
        diversity_iter_loss_list.append(diversity_loss)
        S_D_real_iter_loss_list.append(err_S_D_real)
        S_D_fake_iter_loss_list.append(err_S_D_fake)
        S_D_total_iter_loss_list.append(S_D_total_loss)

        iteration = len(time_list)-1
        
        if ((iteration+1)%(79*5)==0): # save every 5 epoch
            PATH_model_save = "{}{}{:0>7d}{}".format(PATH_model, "/iter_", iteration, ".tar")
            S_K_state_dict = S_K.state_dict()
            optimizerS_K_state_dict = optimizerS_K.state_dict()
            S_D_state_dict = S_D.state_dict()
            optimizerS_D_state_dict = optimizerS_D.state_dict()

            torch.save({
                    "S_K_state_dict": S_K_state_dict,
                    "optimizerS_K_state_dict": optimizerS_K_state_dict,
                    "S_D_state_dict": S_D_state_dict,
                    "optimizerS_D_state_dict":  optimizerS_D_state_dict
                    }, PATH_model_save)

            print("model is saved in {}".format(PATH_model_save))

            torch.save({
                    "S_K_iter_loss_list": S_K_iter_loss_list,
                    "reconstruct_iter_loss_list": reconstruct_iter_loss_list,
                    "diversity_iter_loss_list": diversity_iter_loss_list,
                    "S_D_real_iter_loss_list": S_D_real_iter_loss_list,
                    "S_D_fake_iter_loss_list": S_D_fake_iter_loss_list,
                    "S_D_total_iter_loss_list": S_D_total_iter_loss_list,
                    "time_list": time_list
                    }, PATH_record)

            print("loss record is saved in {}".format(PATH_record))
        

        # send to tensorboard
        writer.add_scalar("loss/S_K", S_K_iter_loss_list[iteration], iteration, time_list[iteration])   # tag, Y, X -> 當Y只有一個時
        writer.add_scalar("loss/reconstruction", reconstruct_iter_loss_list[iteration], iteration, time_list[iteration]) 
        writer.add_scalar("loss/diversity", diversity_iter_loss_list[iteration], iteration, time_list[iteration]) 
        writer.add_scalar("loss/S_D_real", S_D_real_iter_loss_list[iteration], iteration, time_list[iteration])
        writer.add_scalar("loss/S_D_fake", S_D_fake_iter_loss_list[iteration], iteration, time_list[iteration])
        writer.add_scalar("loss/S_D_total", S_D_total_iter_loss_list[iteration], iteration, time_list[iteration])


writer.close()
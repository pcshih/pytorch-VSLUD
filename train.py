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
video = torch.load("datasets/video_frame_pool5.tar")
summary = torch.load("datasets/summary_frame_pool5.tar")
print("loading training data ended")

PATH_record = "loss_record.tar"
PATH_model = "saved_models"

EPOCH = 700

# reconstruction error coefficient
reconstruction_error_coeff = 0.0005
# diversity error coefficient
diversity_error_coeff = 1.5

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
optimizerS_D = optim.SGD(S_D.parameters(), lr=0.00004)


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
        S_K_summary,index_mask,_ = S_K(vd)
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
        #reconstruct_loss = torch.sum((S_K_summary-vd)**2 * index_mask) / torch.sum(index_mask) # [1,1024,1,S]-[1,1024,1,T]
        ###new reconstruct###


        # diversity
        # S_K_summary = index_mask*S_K_summary
        # S_K_summary_reshape = S_K_summary.view(S_K_summary.shape[1], S_K_summary.shape[3])
        # norm_div = torch.norm(S_K_summary_reshape, 2, 0, True)
        # S_K_summary_reshape = S_K_summary_reshape/norm_div
        # loss_matrix = S_K_summary_reshape.transpose(1, 0).mm(S_K_summary_reshape)
        # diversity_loss = loss_matrix.sum() - loss_matrix.trace()
        # #diversity_loss = diversity_loss/len(column_mask)/(len(column_mask)-1)
        # diversity_loss = diversity_loss/(torch.sum(index_mask))/(torch.sum(index_mask)-1)

        ######################## LOSS FROM FCSN #########################
        # 2D 1D conversion
        outputs_reconstruct = S_K_summary.view(1,1024,-1) # [1,1024,1,T] -> [1,1024,T] 
        mask = index_mask.view(1,1,-1) # [1,1,1,T] -> [1,1,T] 
        feature = vd.view(1,1024,-1) # [1,1024,1,T] -> [1,1024,T] 

        # reconst. loss改成分批再做平均
        feature_select = feature*mask
        outputs_reconstruct_select = outputs_reconstruct*mask
        feature_diff_1 = torch.sum((feature_select-outputs_reconstruct_select)**2, dim=1)
        feature_diff_1 = torch.sum(feature_diff_1, dim=1)

        mask_sum = torch.sum(mask, dim=2)
        mask_sum = torch.sum(mask_sum, dim=1)

        reconstruct_loss = torch.mean(feature_diff_1/mask_sum)
        

        # diversity loss
        batch_size, feat_size, frames = outputs_reconstruct.shape

        outputs_reconstruct_norm = torch.norm(outputs_reconstruct, p=2, dim=1, keepdim=True)

        normalized_outputs_reconstruct = outputs_reconstruct/outputs_reconstruct_norm

        normalized_outputs_reconstruct_reshape = normalized_outputs_reconstruct.permute(0, 2, 1)

        similarity_matrix = torch.bmm(normalized_outputs_reconstruct_reshape, normalized_outputs_reconstruct)

        mask_trans = mask.permute(0,2,1)
        mask_matrix = torch.bmm(mask_trans, mask)
        # filter out non key
        similarity_matrix_filtered = similarity_matrix*mask_matrix

        diversity_loss = 0
        acc_batch_size = 0
        for j in range(batch_size):
            batch_similarity_matrix_filtered = similarity_matrix_filtered[j,:,:]
            batch_mask = mask[j,:,:]
            if batch_mask.sum() < 2:
                #print("select less than 2 frames", batch_mask.sum())
                batch_diversity_loss = 0
            else:
                batch_diversity_loss = (batch_similarity_matrix_filtered.sum()-batch_similarity_matrix_filtered.trace())/(batch_mask.sum()*(batch_mask.sum()-1))
                acc_batch_size += 1

            diversity_loss += batch_diversity_loss

        if acc_batch_size>0:
            diversity_loss /= acc_batch_size
            #print(acc_batch_size)
        else:
            diversity_loss = 0


        S_K_total_loss = errS_K + reconstruction_error_coeff*reconstruct_loss + diversity_error_coeff*diversity_loss # for summe dataset beta=1
        #S_K_total_loss = errS_K+reconstruct_loss
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
        S_K_summary,_,_ = S_K(vd)
        output = S_D(S_K_summary.detach()); #print(S_K_summary)
        label.fill_(0)
        err_S_D_fake = criterion(output, label)
        err_S_D_fake.backward()

        S_D_total_loss = err_S_D_real+err_S_D_fake

        optimizerS_D.step()
        
        # record
        time_list.append(time.time())
        S_K_iter_loss_list.append(errS_K)
        reconstruct_iter_loss_list.append(reconstruction_error_coeff*reconstruct_loss)
        diversity_iter_loss_list.append(diversity_error_coeff*diversity_loss)
        S_D_real_iter_loss_list.append(err_S_D_real)
        S_D_fake_iter_loss_list.append(err_S_D_fake)
        S_D_total_iter_loss_list.append(S_D_total_loss)

        iteration = len(time_list)-1
        
        # if ((iteration+1)%(150*5)==0): # save every 5 epoch
        #     PATH_model_save = "{}{}{:0>7d}{}".format(PATH_model, "/iter_", iteration, ".tar")
        #     S_K_state_dict = S_K.state_dict()
        #     optimizerS_K_state_dict = optimizerS_K.state_dict()
        #     S_D_state_dict = S_D.state_dict()
        #     optimizerS_D_state_dict = optimizerS_D.state_dict()

        #     torch.save({
        #             "S_K_state_dict": S_K_state_dict,
        #             "optimizerS_K_state_dict": optimizerS_K_state_dict,
        #             "S_D_state_dict": S_D_state_dict,
        #             "optimizerS_D_state_dict":  optimizerS_D_state_dict
        #             }, PATH_model_save)

        #     print("model is saved in {}".format(PATH_model_save))

        #     torch.save({
        #             "S_K_iter_loss_list": S_K_iter_loss_list,
        #             "reconstruct_iter_loss_list": reconstruct_iter_loss_list,
        #             "diversity_iter_loss_list": diversity_iter_loss_list,
        #             "S_D_real_iter_loss_list": S_D_real_iter_loss_list,
        #             "S_D_fake_iter_loss_list": S_D_fake_iter_loss_list,
        #             "S_D_total_iter_loss_list": S_D_total_iter_loss_list,
        #             "time_list": time_list
        #             }, PATH_record)

        #     print("loss record is saved in {}".format(PATH_record))
        

        # send to tensorboard
        writer.add_scalar("loss/S_K", S_K_iter_loss_list[iteration], iteration, time_list[iteration])   # tag, Y, X -> 當Y只有一個時
        writer.add_scalar("loss/reconstruction", reconstruct_iter_loss_list[iteration], iteration, time_list[iteration]) 
        writer.add_scalar("loss/diversity", diversity_iter_loss_list[iteration], iteration, time_list[iteration]) 
        writer.add_scalar("loss/S_D_real", S_D_real_iter_loss_list[iteration], iteration, time_list[iteration])
        writer.add_scalar("loss/S_D_fake", S_D_fake_iter_loss_list[iteration], iteration, time_list[iteration])
        writer.add_scalar("loss/S_D_total", S_D_total_iter_loss_list[iteration], iteration, time_list[iteration])


writer.close()
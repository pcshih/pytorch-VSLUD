import h5py
import collections
import random
import time

import torch

random.seed(time.time())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# only train for summe first
create_evaluation_dataset = "summe"

dir_base = "datasets"
file_training_video = "{}/reorganized_training_dataset_{}_video.tar".format(dir_base, create_evaluation_dataset)
file_training_summary = "{}/reorganized_training_dataset_{}_summary.tar".format(dir_base, create_evaluation_dataset)

dataset_name_list = ["summe", "tvsum", "youtube", "ovp"]

dataset_reorganized = collections.OrderedDict()

for dataset_name in dataset_name_list:
    dataset = h5py.File("{}/eccv16_dataset_{}_google_pool5.h5".format(dir_base,dataset_name), 'r')
    keys = list(dataset.keys())
    
    # about 20% to be testing set
    if dataset_name=="summe":
        # mimic random selection
        random.shuffle(keys)
        keys = keys[6:]

    for key in keys:
        attributes = collections.OrderedDict()

        new_key = "{}_{}".format(dataset_name, key)

        feature_video_cuda = torch.from_numpy(dataset[key]["features"][...]).to(device)
        feature_video_cuda = feature_video_cuda.transpose(1,0).view(1,1024,1,feature_video_cuda.shape[0])
        attributes["video_features"] = feature_video_cuda; #print(torch.isnan(feature_video_cuda).nonzero().view(-1))

        gt_summary = torch.from_numpy(dataset[key]["gtsummary"][...]).to(device)
        column_index = gt_summary.nonzero().view(-1)
        feature_summary_cuda = torch.from_numpy(dataset[key]["features"][...]).to(device)
        feature_summary_cuda = feature_summary_cuda.transpose(1,0).view(1,1024,1,feature_summary_cuda.shape[0])
        feature_summary_cuda = torch.index_select(feature_summary_cuda, 3, column_index)
        attributes["summary_features"] = feature_summary_cuda; #print(torch.isnan(feature_summary_cuda).nonzero().view(-1))

        dataset_reorganized[new_key] = attributes; #print(new_key, dataset_reorganized[new_key]["video_features"].shape, dataset_reorganized[new_key]["summary_features"].shape)



dataset_reorganized_keys_list = list(dataset_reorganized.keys()); #print(len(dataset_reorganized_keys_list))

# randomized to mimic random selection
random.shuffle(dataset_reorganized_keys_list)
# 50% video, 50% summary
half_index = len(dataset_reorganized_keys_list)//2


video_feature_data_list = []
summary_feature_data_list = []
for idx,video_name in enumerate(dataset_reorganized_keys_list):
    if(idx<half_index):
        video_feature_data_list.append(dataset_reorganized[video_name]["video_features"]); #print(dataset_reorganized[video_name]["video_features"].shape)
    else:
        summary_feature_data_list.append(dataset_reorganized[video_name]["summary_features"]); #print(dataset_reorganized[video_name]["summary_features"].shape)


# save training data
torch.save({"feature":video_feature_data_list}, file_training_video); #print(len(video_feature_data_list))
torch.save({"feature":summary_feature_data_list}, file_training_summary); #print(len(summary_feature_data_list))

# print log
print("video training file save in {}".format(file_training_video))
print("summary training file save in {}".format(file_training_summary))
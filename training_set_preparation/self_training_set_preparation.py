import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from FeatureExtractor import FeatureExtractor

import subprocess
import tqdm



class PreProcess():
    """
    video_path_list: 原始影片放在哪個資料夾
    root_path_list: downsample完的frame要放在哪個資料夾
    save_path_list: 抽完的features要放在哪個資料夾
    """
    def __init__(self, video_path_list, root_path_list, save_path_list):
        self.resize = 224
        self.video_path = video_path_list
        self.root_path = root_path_list
        self.save_path = save_path_list
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = FeatureExtractor()
        self.net.eval()
        

    def pre_process(self):
        for index in range(len(self.save_path)): # or root_path, save_path
            # extract frame first
            arg_1 = self.video_path[index]
            arg_2 = self.root_path[index]
            print("START extracting frames")
            subprocess.call(["./extract_frame.sh", arg_1, arg_2])   # extract frame first
            print("END extracting frames")
            # 使用內建的預處理
            # dataset[0][0] 代表第一個資料的tensor
            # dataset[0][1] 代表地個資料的class
            dataset = dset.ImageFolder(root=self.root_path[index],
                            transform=transforms.Compose([
                                transforms.Resize((self.resize,self.resize)), # googlenet(inception v1) accepts size 224x224 image
                                transforms.ToTensor(), # HWC->CHW [0,255]->[0.0,1.0]
                                transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)) # [0.0,1.0] -> [-1.0,1.0] official 那組是從imageNet出來的
                            ]))
            
            feature_data_list = [] # 裡面存每個video sub frames的feature i.e. [[1,1024,1,v1], [1,1024,1,v2]......]
            frame_list = [] # 每個影片裡面有多少個frame i.e. [v1, v2......]
            name_list = dataset.classes # 存每部影片的名字 i.e. [Gordon, James......]


            # get the frame_count of each video -> frame_list
            tqdm_range = tqdm.trange(len(dataset.classes))
            for video_idx in tqdm_range:
                tqdm_range.set_description(" Extracting Features from {}".format(name_list[video_idx]))
                temp_list = [i for i in dataset.imgs if i[1] == video_idx] # 用來數這部影片有多少個frame
                frame_list.append(len(temp_list))  # 得到該video的frame數量並儲存, frame_list=[1143, 2242, ......]

                video_images = torch.randn((len(temp_list),3,self.resize,self.resize)) # 宣告用來存每部video frame的空間

                # 先把一部影片的所有frame寫入video_images，之後要用googlenet抽特徵
                for frame in range(frame_list[video_idx]):
                    if video_idx==0: # 如果是第一步影片時，idx從0~frame_list[video_idx]-1
                        video_images[frame][:][:][:] = dataset[frame][0].view(1,3,self.resize,self.resize)
                    else:
                        video_images[frame][:][:][:] = dataset[frame+frame_list[video_idx-1]][0].view(1,3,self.resize,self.resize)

                # 因為GPU沒辦法處理一次整個影片的frames做feature extraction，故要切
                video_images_subs = torch.split(video_images, 1500, dim=0) # 多少張切成一個區塊，第一個區塊會存1500張照片
                video_images_subs = list(video_images_subs) # 轉成list[[1500,3,224,224], [rest,3,224,224]]

                # 處理每個區塊的feature
                for idx,sub in enumerate(video_images_subs):
                    sub_gpu = sub.to(self.device); #print(sub_gpu.shape)
                    sub_feature_data = self.net(sub_gpu)  # [1024,T]

                    if(idx == 0):
                        cat_sub_feature = sub_feature_data
                        #print(cat_sub_feature.shape)
                    else:
                        cat_sub_feature = torch.cat((cat_sub_feature,sub_feature_data),1)
                        #print(cat_sub_feature.shape)
                    
                    # release gpu memory
                    sub_gpu = sub_gpu.cpu()
                    sub_feature_data = sub_feature_data.cpu()
                    torch.cuda.empty_cache()
                

                cat_sub_feature = cat_sub_feature.view(1,1024,1,cat_sub_feature.size()[1]); 

                print(cat_sub_feature)
                feature_data_list.append(cat_sub_feature); #print(cat_sub_feature.requires_grad)

                # release gpu memory
                cat_sub_feature = cat_sub_feature.cpu()
                torch.cuda.empty_cache()


            torch.save({"feature":feature_data_list, "name_list":name_list, "frame_list":frame_list}, self.save_path[index])

            # print save result
            for i,feature in enumerate(feature_data_list):
                print(name_list[i], frame_list[i], feature.shape)


    def test(self):
        for index in range(len(self.video_path)):
            arg_1 = self.video_path[index]
            arg_2 = self.root_path[index]
            subprocess.call(["./extract_frame.sh", arg_1, arg_2])   # extract frame first


if __name__ == '__main__':
    #print(len(dataset.classes), dataset.classes)
    #print(dataset.class_to_idx)
    #print(dataset.__len__())

    #checkpoint = torch.load(PATH)
    #print(len(checkpoint["training_data"]))


    # 要抽取feature的影片位置
    #"/media/data/PTec131b/VideoSum/training_data/video",
    #"/media/data/PTec131b/VideoSum/training_data/summary",
    #"/media/data/PTec131b/VideoSum/testing_data/video"
    video_path_list = [
                        "/media/data/PTec131b/VideoSum/training_data/video",
                        "/media/data/PTec131b/VideoSum/training_data/summary"
                        #"/media/data/PTec131b/VideoSum/testing_data/video"
                      ]
    # 暫存的影片frame先放在哪裡
    #"/media/data/PTec131b/VideoSum/training_data/video_frame", 
    #"/media/data/PTec131b/VideoSum/training_data/summary_frame",
    #"/media/data/PTec131b/VideoSum/testing_data/video_frame"
    root_path_list = [
                        "/media/data/PTec131b/VideoSum/training_data/video_frame_wang",
                        "/media/data/PTec131b/VideoSum/training_data/summary_frame_wang"
                        #"/media/data/PTec131b/VideoSum/testing_data/video_frame"
                     ]
    # 影片抽取完feature要放在哪
    #"/media/data/PTec131b/VideoSum/training_data/video_frame/video_frame.tar",
    #"/media/data/PTec131b/VideoSum/training_data/summary_frame/summary_frame.tar",
    #"/media/data/PTec131b/VideoSum/testing_data/video_frame/video_frame.tar"
    save_path_list = [
                        "/media/data/PTec131b/VideoSum/training_data/video_frame_wang/video_frame_pool5.tar",
                        "/media/data/PTec131b/VideoSum/training_data/summary_frame_wang/summary_frame_pool5.tar"
                        #"/media/data/PTec131b/VideoSum/testing_data/video_frame/video_frame.tar"
                     ]
    process = PreProcess(video_path_list, root_path_list, save_path_list)
    #process.test()
    process.pre_process()
    #a = "{}{}{:0>5d}{}".format("./saved_models", "/iter_", 5, ".tar")
    #print(a)
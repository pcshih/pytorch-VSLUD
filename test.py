import torch
from SK import *
from SD import *
import subprocess
import tqdm
import cv2
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

S_K = SK_test().to(device)
S_D = SD_test().to(device)

PATH_model_load = "saved_models/iter_0014999.tar" 
checkpoint_model = torch.load(PATH_model_load)
S_K.load_state_dict(checkpoint_model['S_K_state_dict'])
S_D.load_state_dict(checkpoint_model['S_D_state_dict'])

S_K.eval()
S_D.eval()
#S_K.FCSN.eval()
#S_D.FCSN_ENC.eval()

video = torch.load("datasets/test_video_frame_pool5.tar")

video_path="/media/data/PTec131b/VideoSum/testing_data/video"
video_frame_path="/media/data/PTec131b/VideoSum/testing_data/video_frame"
video_processed_path="/media/data/PTec131b/VideoSum/testing_data/video_processed_0029999"



def merge_frame(name, video_processed_path, video_frame_path, video_mask, frames_count):
    """
    name: 影片名
    video_processed_path: sample過後的影片、挑選完key frame的影片及concate前兩者的影片要放在哪
    video_mask: key frame mask
    frames_count: 這部影片總過有幾個frame
    """
    width = 1280
    height = 720
    fps = 2
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    black_frame = np.zeros((height,width,3), dtype=np.uint8)

    file_name_2fps = "{}/{}_2fps.mp4".format(video_processed_path,name)
    file_name_selected = "{}/{}_selected.mp4".format(video_processed_path,name)
    file_name_concate = "{}/{}_concate.mp4".format(video_processed_path,name)


    # out_2fps = cv2.VideoWriter(file_name_2fps,    
    #                         fourcc,
    #                         fps,
    #                         (width, height))
    # out_selected = cv2.VideoWriter(file_name_selected,    
    #                         fourcc,
    #                         fps,
    #                         (width, height))
    out_concate = cv2.VideoWriter(file_name_concate,    
                            fourcc,
                            fps,
                            (width*2, height))


    for i in range(frames_count):
        # 2fps
        frame_path = "{}/{}/{}_{:0>4d}.jpg".format(video_frame_path, name, name, i+1)
        frame_2fps = cv2.imread(frame_path)
        #out_2fps.write(frame_2fps)

        # selected+concate
        if (i>=len(video_mask)):
            frame = black_frame
        else:
            frame_path = "{}/{}/{}_{:0>4d}.jpg".format(video_frame_path, name, name, video_mask[i]+1)
            frame = cv2.imread(frame_path)
            #out_selected.write(frame)
        
        frame_concate = np.concatenate((frame_2fps, frame), axis=1)
        out_concate.write(frame_concate)

    #out_2fps.release()
    #out_selected.release()
    out_concate.release()





def test():
    tqdm_range = tqdm.trange(len(video["feature"]))

    for i in tqdm_range: # video["feature"] -> [[1,1024,1,A], [1,1024,1,B]...]
        vd = video["feature"][i].to(device)
        name = video["name_list"][i]
        frames_count = video["frame_list"][i]

        _,video_mask,_ = S_K(vd)

        print(i, video_mask.view(-1))

        #subprocess.call(["./merge_original_frame.sh", 
        #                 video_frame_path, 
        #                 video_processed_path,
        #                 name])
        
        # merge frames
        #video_mask_list = video_mask.view(-1).nonzero().view(-1).tolist(); print(video_mask_list)
        #merge_frame(name, video_processed_path, video_frame_path, video_mask_list, frames_count)


if __name__ == '__main__':
    test()
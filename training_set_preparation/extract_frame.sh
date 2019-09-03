#!/bin/zsh

# no space in "=" left or right
# seems cannot use "~" to get user path
#video_path="/media/data/PTec131b/VideoSum/training_data/video"
#video_frame_path="/media/data/PTec131b/VideoSum/training_data/video_frame"



#video_path="/media/data/PTec131b/VideoSum/training_data/video_wang"
#video_frame_path="/media/data/PTec131b/VideoSum/training_data/video_frame_wang"

video_path=${1}
video_frame_path=${2}

skip_start_time=10
skip_end_time=20

for file in $(ls ${video_path}) # 加冒號最後會多一個冒號!!!
do
    #echo ${file}
    #echo "Extracting frames of ${file} ..."
    dir_name="${video_frame_path}/${file%????}"  #% ?有幾個代表去除最後幾個字母
    
    # 如果有處理過的不再處理
    if [ -d ${dir_name} ]
    then
        continue
    else
        mkdir ${dir_name}
    fi

    file_path="${video_path}/${file}"
    # cut -c 只要13~20個字
    # awk -F: 以":"作為分隔，一般是以" "作為分割
    duration=$(ffmpeg -i ${file_path} 2>&1 | grep "Duration" | cut -c 13-20 | awk -F: '{ print $1*3600+$2*60+$3 }')
    interval=$((duration-skip_start_time-skip_end_time)) # residue of duration

    
    frame_path="${dir_name}/${file%????}_%04d.jpg"
    # skip front and last
    #ffmpeg -i ${file_path} -ss ${skip_start_time} -t ${interval} -vf fps=2 ${frame_path} 2>&1 | grep "Input"   # sample one frame every 0.5 sec(2fps), -ss 前幾秒丟掉
    # no skip
    ffmpeg -i ${file_path} -vf fps=2 ${frame_path} 2>&1 | grep "Input"
done
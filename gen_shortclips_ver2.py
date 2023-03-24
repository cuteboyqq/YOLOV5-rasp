#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:00:47 2023

@author: ali
"""
import os
import glob
import cv2
import shutil
CONF_TH=0.70
SHIFT_BACK_FRAME = 40
SHIFT_FORWARD_FRAME = 40
SET_W=1280
SET_H=720
SET_FPS=20
#the particular label.txt name is videoname_fcnt.txt
#for example: 0_5.txt , 0_150.txt ..,etc
def Analysis_path(path):
    file=path.split("/")[-1]
    file_name=file.split(".")[0]
    #f_cnt = file_name.split("_")[-1]
    return file_name

def Analysis_file(file):
    file_name = file.split(".")[0]
    frame_count = file_name.split("_")[-1]
    return file_name,frame_count

def Generate_Short_Clips(root_data_dir=r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp2/0_imgs",
                         anomaly_img=None,
                         shift_left=40,
                         shift_right=40,
                         save_ano_clip_dir=None
                         ):
    
    img_list=glob.glob(os.path.join(root_data_dir,'0_imgs','*.jpg'))
    
    
    #Get the left boundary
    print("anomaly_img : {}".format(anomaly_img))
    frame_count = Analysis_path(anomaly_img)
    frame_count = int(frame_count)
    if frame_count - shift_left>0:
        left_boundary_frame =  frame_count - shift_left
    else:
        left_boundary_frame = 1
       
    #Get the right boundary
    if frame_count + shift_right < len(img_list):
        right_boundary_frame =  frame_count + shift_right
    else:
        right_boundary_frame = len(img_list) - 2
        
    #save_ano_clip_dir=os.path.join(save_ano_clip_dir,"anomaly_clips")
    if not os.path.exists(save_ano_clip_dir):
        os.makedirs(save_ano_clip_dir)
                    
    anomaly_video = str(left_boundary_frame) + "_" + str(right_boundary_frame) + "_" + str(frame_count) + ".avi"
    #anomaly_video = str(left_boundary_frame) + "_" + str(frame_count) + ".avi"
    save_ano_clips_path = os.path.join(save_ano_clip_dir,anomaly_video)
    print(save_ano_clips_path)
    w,h=SET_W,SET_H
    fps=SET_FPS
    anomaly_video_writer = cv2.VideoWriter(save_ano_clips_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    
    #Save the stream at range [left_boundary_frame] and [right_boundary_frame]
    for i in range(left_boundary_frame,right_boundary_frame+1):
        ano_img = "0_" + str(i) + ".jpg"
        if os.path.exists(os.path.join(root_data_dir,"0_imgs")):
            ano_img_path = os.path.join(root_data_dir,"0_imgs",ano_img)
            ano_im = cv2.imread(ano_img_path)
            anomaly_video_writer.write(ano_im)
    
    
    
    anomaly_video_writer.release()
    

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-datadir','--data-dir',help="folder 0_imgs path",default= r"/home/ali/Desktop/YOLOV5-rasp/runs/detect/exp2/0_imgs")
    parser.add_argument('-rootdatadir','--root-datadir',help="folder of root data",default=r"/home/ali/Desktop/YOLOV5-rasp/runs/detect/exp2")
    parser.add_argument('-anomalyimg','--anomaly-img',help="anomaly image file ex:0_50.jpg",default="0_20.jpg")
    parser.add_argument('-shiftleft','--shift-left',type=int,help="the left frame range of the stream",default=40)
    parser.add_argument('-shiftright','--shift-right',type=int,help="the right frame range of the stream",default=40)
    parser.add_argument('--saveanoclipdir','--save-anoclipdir',help="the directory of save anomaly clips",default=r"/home/ali/Desktop/YOLOV5-rasp/runs/detect/exp2/anomaly_clips")
    
    return parser.parse_args()
               

if __name__=="__main__":
    '''
    data_dir = r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp2/0_imgs"
    root_data_dir = r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp2"
    save_ano_clip_dir = r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp2/anomaly_clips"
    shift_left = 40
    shift_right = 40 
    anomaly_img = "0_145.jpg"
    '''
    
    args = get_args()
    
    data_dir = args.data_dir
    root_data_dir = args.root_datadir
    save_ano_clip_dir = args.saveanoclipdir
    shift_left = args.shift_left
    shift_right = args.shift_right
    anomaly_img = args.anomaly_img
    
    print(f'data_dir = {data_dir}')
    print("root_data_dir = {}".format(root_data_dir))
    print("save_ano_clip_dir = {}".format(save_ano_clip_dir))
    print("shift_left = {}".format(shift_left))
    print("shift_right = {}".format(shift_right))
    print("anomaly_img = {}".format(anomaly_img))
    
    
    Generate_Short_Clips(root_data_dir,anomaly_img,shift_left,shift_right,save_ano_clip_dir)
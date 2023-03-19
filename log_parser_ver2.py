#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:57:31 2023

@author: ali
"""
from utils.plots import Annotator, colors, save_one_box
import cv2
import os
import shutil
def Analysis_path(path):
    file = path.split(os.sep)[-1]
    file_name = file.split(".")[0]
    file_dir = os.path.dirname(path)
    print("file = ",file)
    print("file_name = ",file_name)
    print("file_dir = ",file_dir)
    return file,file_name,file_dir

def parse_log_txt(log_dir="runs/detect/"):
    #parsing log file
    #log format is : [save_txt_path] [class] x y w h
    #generate each frame label.txt and save at file_dir
    log_path = log_dir + "log.txt"
    with open(r"C:\GitHub_Code\cuteboyqq\YOLO\YOLOV5-rasp\runs\detect\exp22\log.txt") as f:
        for line in f.readlines():
            save_txt_path = line.split(" ")[0]
            label = line.split(" ")[1:]
            label_str = ' '.join(x for x in label)
            print(line.split(" ")[0])
            
            
            file,file_name,file_dir = Analysis_path(save_txt_path)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            with open(save_txt_path,'a') as l_f:
                l_f.write(label_str)
    
    return file_dir
                


    
def Analysis_path(path):
    file = path.split(os.sep)[-1]
    file_name = file.split(".")[0]
    file_dir = os.path.dirname(path)
    print("file = ",file)
    print("file_name = ",file_name)
    print("file_dir = ",file_dir)
    return file,file_name,file_dir

#c_file,c_file_name,c_file_dir = Analysis_path(class_path)
SET_FPS = 20
SET_W = 1280
SET_H = 720
from datetime import datetime
import numpy as np
SAVE_RAW_STREAM=True
def video_extract_frame(path,
                        skip_frame,
                        txt_dir,
                        class_path,
                        img_size,
                        yolo_infer_txt,
                        log_dir):
    
    #Alister add 2023-02-28
    if SAVE_RAW_STREAM:
        now = datetime.now()
        s_time = datetime.strftime(now,'%y-%m-%d_%H-%M-%S')
        s_time = str(s_time)
        save_path = log_dir + "0_result_offline"+ ".avi"
        w,h=SET_W,SET_H
        fps=SET_FPS
        vw = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
        #save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
    
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 1
    file,filename,file_dir = Analysis_path(path)
    print(file," ",filename," ",file_dir)
    save_folder_name =  filename + "_imgs"
    save_dir = os.path.join(file_dir,save_folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    #Copy class.txt to save_dir
    shutil.copy(class_path,save_dir)
    
    while True:
        if success:
            namess = "2023-03-17"
            annotator = Annotator(image, line_width=2, example=str(namess))
            if count%skip_frame==0:
                
                #====extract video frame====
                filename_ = filename + "_" + str(count) + ".jpg"
                img_path = os.path.join(save_dir,filename_)
                #image = cv2.resize(image,(img_size, int(img_size*9/16) ))
                #image = cv2.resize(image,(img_size,img_size))
                
                #=====Start parsing label.txt, try to get the xyxy and cls informations======================
                filename_txt_ = filename + "_" + str(count) + ".txt"
                txt_path = os.path.join(txt_dir,filename_txt_)
                if os.path.exists(txt_path):
                    with open(txt_path,'r') as f:
                        for line in f.readlines():
                            print(line)
                            #Line format is [class] x y w h
                            #Try to get xywh
                            xyxy = line.split(" ")[1:5]
                            label = line.split(" ")[0]
                            conf = line.split(" ")[5]
                            conf_f = float(conf)
                            xyxy_str = ' '.join(x for x in xyxy)
                            label_str = ' '.join(x for x in label)
                            conf_str = ' '.join(x for x in conf)
                            print("label_str = {}".format(label_str))
                            print("xyxy_str = {}".format(xyxy_str))
                            print("conf_str = {}".format(conf_str))
                            
                            #Not implemented
                            save_airesult=True
                            hide_labels = False
                            hide_conf = False
                            names = ["line","noline","others","frontline"]
                            if save_airesult:
                                #=====End parsing label.txt, try to get the xyxy and cls informations======================
                                c = int(label_str)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf_f:.2f}')
                                #annotator.box_label(xyxy, label, color=colors(c, True))
                                #if c==0 and filter_line_label==False: #noline (test)
                                if c==0: #noline (test)
                                    if conf<0.70:
                                        annotator.box_label(xyxy, label+" anomaly" , color=(255,0,128))
                                    else:
                                        annotator.box_label(xyxy, label+" normal" , color=(255,0,0))
                                elif not c==0:
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                    #===========================================================
                
            #==============================================
            if SAVE_RAW_STREAM:
                #names="test_2023_03_03"
                #image = np.ascontiguousarray(image)  # contiguous
                #annotator = Annotator(im0[0], line_width=3, example=str(names))
                #if not SET_H is 720:
                    #image = image[..., ::-1]
                #annotator.time_label(frame_count=self.count,txt_color=(0,255,128))
                vw.write(image)
            #==============================================
                
                cv2.imwrite(img_path,image)
                print("save image complete",img_path)
                
                #cv2.imwrite(img_path,image)
                if yolo_infer_txt:
                    #=====Copy .txt file=======
                    filename_txt_ = filename + "_" + str(count) + ".txt"
                    txt_path = os.path.join(txt_dir,filename_txt_)
                    if os.path.exists(txt_path):
                        shutil.copy(txt_path,save_dir)
                    
                #cv2.imwrite("/home/ali/datasets-old/TL4/frame%d.jpg" % count, image)     # save frame as JPEG file    
                print('save frame ',count)
        else:
            print('Video capture failed, break')
            continue
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('-videopath','--video-path',help="input video path",default="/home/ali/Argos_Project/Factory_Project/Argos_Raw_Stream/2023-03-04/Big_Screen/Light_ON_OFF/20230304_BIGSCREEN_LIGHT_OFF_AND_ON.mp4")
    #parser.add_argument('-videopath','--video-path',help="input video path",default="/home/ali/Argos_Project/Factory_Project/Screen_record_Argos_Stream/20230314_203715.mp4")
    #parser.add_argument('-videopath','--video-path',help="input video path",default="/home/ali/factory_video/Argos_Record/2023-03-15/SmallScreen_Record/filterline_version2/YoloV4-f320-4cls/20230315-215557-H264-00.mp4")
    #parser.add_argument('-videopath','--video-path',help="input video path",default="/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/2023-03-17/0.avi")
    parser.add_argument('-videopath','--video-path',help="input video path",default=r"C:\GitHub_Code\cuteboyqq\YOLO\YOLOV5-rasp\runs\detect\exp22\0.avi")
    parser.add_argument('-skipf','--skip-f',type=int,help="number of skp frame",default=1)
    parser.add_argument('-imgsize','--img-size',type=int,help="size of images",default=640)
    parser.add_argument('-yoloinfer','--yolo-infer',action='store_true',help="have yolo infer txt")
    #parser.add_argument('-yolotxt','--yolo-txt',help="yolo infer label txt dir",default="/home/ali/factory_video/2023-03-04/labels")
    parser.add_argument('-yolotxt','--yolo-txt',help="yolo infer label txt dir",default="/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/2023-03-1712/labels")
    parser.add_argument('-classtxt','--class-txt',help="class.txt path",default="C:\GitHub_Code\cuteboyqq\YOLO\classes.txt")
    
    return parser.parse_args()
    
if __name__=="__main__":
    
    args=get_args()
    video_path = args.video_path
    skip_frame = args.skip_f
    yolo_txt_dir = args.yolo_txt
    class_path = args.class_txt
    yolo_infer = args.yolo_infer
    img_size = args.img_size
    print("video_path =",video_path)
    print("skip_frame = ",skip_frame)
    print("yolo_txt_dir = ",yolo_txt_dir)
    print("class_path = ",class_path)
    print("yolo_infer = ",yolo_infer)
    print("img_size = ",img_size)
    log_dir="C:/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp22/"
    #yolo_txt_dir = parse_log_txt(log_dir)
    yolo_txt_dir = r"C:\GitHub_Code\cuteboyqq\YOLO\YOLOV5-rasp\runs\detect\exp22\labels"
    
    video_extract_frame(video_path, 
                        skip_frame, 
                        yolo_txt_dir, 
                        class_path, 
                        img_size,
                        True,
                        log_dir)
    
    
        
        
        

        


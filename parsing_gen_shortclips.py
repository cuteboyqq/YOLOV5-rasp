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
#the particular label.txt name is videoname_fcnt.txt
#for example: 0_5.txt , 0_150.txt ..,etc
def Analysis_path(path):
    file=path.split("/")[-1]
    file_name=file.split(".")[0]
    f_cnt = file_name.split("_")[-1]
    return f_cnt


def Parsing_Result_Imgs_Labels(data_dir,root_data_dir):
    print(root_data_dir)
    if os.path.exists(os.path.join(root_data_dir,"anomaly_clips")):
        shutil.rmtree(os.path.join(root_data_dir,"anomaly_clips"))
    if not os.path.exists(os.path.join(root_data_dir,"anomaly_clips")):
        os.makedirs(os.path.join(root_data_dir,'anomaly_clips'))
    #delete classes.txt first
    if os.path.exists(os.path.join(data_dir,"classes.txt")):
        os.remove(os.path.join(data_dir,"classes.txt"))    
    
    #get the label list and img list
    img_path_list = glob.glob(os.path.join(data_dir,'*.jpg'))
    label_path_list = glob.glob(os.path.join(data_dir,'*.txt'))
    
    #the frame range of the short clips
    start_frame_num = -1
    end_frame_num = -1
    first_time = True
    #go through all label.txt
    for i in range(len(label_path_list)):
        print("{} {}".format(i,label_path_list[i]))
        #go through label.txt content
        f_cnt = Analysis_path(label_path_list[i])
        with open(label_path_list[i],'r') as f:
            boundry_updated=False
            for line in f.readlines():
                print("fr {}: {}".format(f_cnt,line))
                #start paring each label
                clas = line.split(" ")[0]
                xyxy = line.split(" ")[1:5]
                conf = line.split(" ")[5]
                
                #print("class {}".format(clas))
                #print("xyxy {}".format(xyxy))
                #print("conf {}".format(conf))
                
                #Try to get the anomaly label.txt
                if float(conf)<CONF_TH and int(clas)==0:
                    print("fr {}".format(f_cnt))
                    print("anomaly class {}".format(clas))
                    print("anomaly xyxy {}".format(xyxy))
                    print("anomaly conf {}".format(conf))
                    
                    #get the clip frame num boundary that needs to save
                    if first_time:
                        first_time=False
                        #check the boundary
                        if int(f_cnt) - SHIFT_BACK_FRAME>=1:
                            start_frame_num = int(f_cnt) - SHIFT_BACK_FRAME
                        elif int(f_cnt) - SHIFT_BACK_FRAME<=0:
                            start_frame_num = 1
                            
                        if int(f_cnt) + SHIFT_FORWARD_FRAME<=len(label_path_list):
                            end_frame_num =int(f_cnt) + SHIFT_FORWARD_FRAME
                        else:
                            end_frame_num = len(label_path_list) -1
                        boundry_updated=True
                    else:#not first time
                        #check the whether f_cnt is in the range of (start_frame_num,end_frame_num)
                        #if yes, do nothing
                        #if no, update the start_frame_num and end_frame_num
                        if int(f_cnt) <= end_frame_num and int(f_cnt)>=start_frame_num:
                            boundry_updated=False
                        else:
                            #check the boundary
                            if int(f_cnt) - SHIFT_BACK_FRAME>=1:
                                start_frame_num = int(f_cnt) - SHIFT_BACK_FRAME
                            elif int(f_cnt) - SHIFT_BACK_FRAME<=0:
                                start_frame_num = 1
                                
                            if int(f_cnt) + SHIFT_FORWARD_FRAME<=len(label_path_list):
                                end_frame_num =int(f_cnt) + SHIFT_FORWARD_FRAME
                            else:
                                end_frame_num = len(label_path_list) -1
                            boundry_updated=True
                            
                            
                    SET_W=1280
                    SET_H=720
                    SET_FPS=20
                    if boundry_updated:
                        #start saving frames into a stream
                        print("boundry_updated:{}".format(boundry_updated))
                        print("start_frame_num:{}".format(start_frame_num))
                        print("end_frame_num:{}".format(end_frame_num))
                        file = str(start_frame_num) + "_" + str(end_frame_num) + "_" + f_cnt + ".avi"
                        save_dir = "/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp2/anomaly_clips"
                        save_path = os.path.join(save_dir,file)
                        print(save_path)
                        w,h=SET_W,SET_H
                        fps=SET_FPS
                        
                        
                        
                        if start_frame_num<end_frame_num:
                            vw = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
                            #start parsing frame and save to stream
                            for i in range(start_frame_num,end_frame_num+1):
                                img = "0_"+str(i)+".jpg"
                                img_path=os.path.join(data_dir,img)
                                im=cv2.imread(img_path)
                                vw.write(im)
                            
                            vw.release()
                            
                        
                        
                        
                        
                    #start_frame_num = f_cnt - SHIFT_BACK_FRAME
                    #end_frame_num =f_cnt + SHIFT_FORWARD_FRAME
                    
                    print("start_frame_num:{}".format(start_frame_num))
                    print("end_frame_num:{}".format(end_frame_num))
                    
                    
                    #save the image.jpg at range(start_frame_num,end_frame_num)
                    
                    
                #else:
                    
                    #print("class {}".format(clas))
                    #print("xyxy {}".format(xyxy))
                    #print("conf {}".format(conf))
    
    
    #return NotImplemented





if __name__=="__main__":
    data_dir = r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp2/0_imgs"
    root_data_dir = r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp2"
    Parsing_Result_Imgs_Labels(data_dir,root_data_dir)
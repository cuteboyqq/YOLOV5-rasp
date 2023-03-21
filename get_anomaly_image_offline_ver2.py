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
RANGE = 80
#the particular label.txt name is videoname_fcnt.txt
#for example: 0_5.txt , 0_150.txt ..,etc
def Analysis_path(path):
    path_dir = os.path.dirname(path)
    file=path.split("/")[-1]
    file_name=file.split(".")[0]
    f_cnt = file_name.split("_")[-1]
    return f_cnt,file,file_name,path_dir


def Parsing_Result_Imgs_Labels(root_data_dir):
    print(root_data_dir)
    if os.path.exists(os.path.join(root_data_dir,"anomaly_clips")):
        shutil.rmtree(os.path.join(root_data_dir,"anomaly_clips"))
    if not os.path.exists(os.path.join(root_data_dir,"anomaly_clips")):
        os.makedirs(os.path.join(root_data_dir,'anomaly_clips'))
    #delete classes.txt first
    if os.path.exists(os.path.join(root_data_dir,"0_imgs","classes.txt")):
        os.remove(os.path.join(root_data_dir,"0_imgs","classes.txt"))    
    
    #get the label list and img list
    img_path_list = glob.glob(os.path.join(root_data_dir,"0_imgs",'*.jpg'))
    label_path_list = glob.glob(os.path.join(root_data_dir,"0_imgs",'*.txt'))
    
    #the frame range of the short clips
    start_frame_num = -1
    end_frame_num = -1
    first_time = True
    #go through all label.txt
    boundry_have_anomaly=False
    for i in range(len(label_path_list)):
        print("{} {}".format(i,label_path_list[i]))
        #go through label.txt content
        if i%RANGE==0 and boundry_have_anomaly==True:
            boundry_have_anomaly=False
            
        #f_cnt,file,file_name,path_dir = Analysis_path(label_path_list[i])
        
        file_txt = "0_" + str(i) + ".txt"
        file_path = os.path.join(root_data_dir,"0_imgs",file_txt)
        if os.path.exists(file_path):
            with open(file_path,'r') as f:
                for line in f.readlines():
                    print("{}".format(line))
                    #start paring each label
                    clas = line.split(" ")[0]
                    xyxy = line.split(" ")[1:5]
                    conf = line.split(" ")[5]
                    
                    #print("class {}".format(clas))
                    #print("xyxy {}".format(xyxy))
                    #print("conf {}".format(conf))
                    
                    #Try to get the anomaly label.txt
                    if float(conf)<CONF_TH and int(clas)==0 and boundry_have_anomaly==False:
                        #print("fr {}".format(f_cnt))
                        print("anomaly class {}".format(clas))
                        print("anomaly xyxy {}".format(xyxy))
                        print("anomaly conf {}".format(conf))
                        #get img
                        
                        im = "0_" + str(i) + ".jpg"
                        new_im = str(i) + ".jpg"
                        path_dir = os.path.join(root_data_dir,"0_imgs")
                        im_path = os.path.join(path_dir,im)
                        ano_img = cv2.imread(im_path)
                        save_path = os.path.join(root_data_dir,"anomaly_img_offline",new_im)
                        if not os.path.exists(os.path.join(root_data_dir,"anomaly_img_offline")):
                            os.makedirs(os.path.join(root_data_dir,"anomaly_img_offline"))
                        
                        boundry_have_anomaly=True
                                                           
                                                           
                        cv2.imwrite(save_path,ano_img)
                   

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-rootdatadir','--root-datadir',help="the directory of inference result dataset",default=r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp2")
    
    return parser.parse_args()    

if __name__=="__main__":
    #data_dir = r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp2/0_imgs"
    #root_data_dir = r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp2"
    args = get_args()
    root_data_dir = args.root_datadir
    Parsing_Result_Imgs_Labels(root_data_dir)
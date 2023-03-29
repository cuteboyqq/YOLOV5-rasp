# YOLOv5 ğŸš€ by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             'path/*.jpg'   # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
import threading
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from datetime import datetime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import threading
import time
import numpy as np
import queue
import time

Total_postprocess_time = 0
frame_count_global = 1
#=========Alister add 20223-02-28 Start to write multiprocess==================Failed
from multiprocessing import Process
#import multiprocessing.Queue as process_queue #wrong
from multiprocessing import Queue
get_frame_proc_queue = Queue()
my_proc_queue = Queue()
#================================================================================
SAVE_AI_RESULT_STREAM=True
USE_SEM4=True
USE_SEM5=False
USE_TIME=False
FPS_SET=10
SET_W=1280
SET_H=720
fr_cnt=0
set_time_2 = 0.001
set_time_1 = 0.001
set_time_3 = 0.001
#im_global=None
#path_global=None
#im0s_global=None
s_global=None
vid_cap_global=None
#pred_global=None
model_global=None
#Alister add 2023-03-08
during_get_frame_global=None
during_model_inference_global=None

#Alister add 2023-03-09
#get_frame_list_global=[]


anomaly_img_count=0
sem1 = threading.Semaphore(0)
sem2 = threading.Semaphore(0)
sem3 = threading.Semaphore(0)
sem4 = threading.Semaphore(1)
sem5 = threading.Semaphore(1)

# Alister 2023-03-11 add
get_frame_sem = threading.Semaphore(1)
model_inference_sem = threading.Semaphore(1)


# 建立佇列
queue_size1=10
queue_size2=10
get_frame_queue = queue.Queue(queue_size1)
my_queue = queue.Queue(queue_size2)
parameter_queue = queue.Queue(queue_size2)
get_frame_and_model_infer_queue = queue.Queue()
MULTI_PROCESS=False
MULTI_THREAD=True
frame_cnt = 1
THREE_THREADS=True #if False, using two threads
SHOW_QUEUE_TIME_LOG = False


#@smart_inference_mode()
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default= r'/home/ali/Desktop/YOLOV5-rasp/runs/train/f192_2022-12-29-4cls/weights/best.pt', help='model path(s)')
    #parser.add_argument('--source', type=str, default=r'/home/ali/factory_video/ori_video_ver2.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=r'/home/ali/Desktop/YOLOV5-rasp/data/factory_new2.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[192], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--save-airesult', action='store_true', help='save result images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

#
def load_model(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
               device='',
               dnn=False,
               data=ROOT / 'data/coco128.yaml',
               half=False,
               imgsz=(192, 192)):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    #raise NotImplemented
    return model, stride, pt, imgsz, device, names
#    
def load_dataloader(source=0,
                    nosave=False,
                    imgsz=(192,192),
                    stride = 16,
                    pt = ''
                    ):
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
    
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    #raise NotImplemented
    
    return dataset, bs, source
#    

        
    #raise NotImplemented


            

def Get_Frame(dataset):
    
    #global im_global
    #global path_global
    #global im0s_global
    #global s_global
    #global vid_cap_global
    
    #path, im, im0s, vid_cap, s = dataset
    #print(dataset)
    '''
    dataset, bs, source = load_dataloader(source=source,
                                    nosave=False,
                                    imgsz=imgsz,
                                    stride=stride,
                                    pt=pt
                                    )
    '''
    
    global s_global
    global vid_cap_global
    dataset.__iter__()
    while True:
        
        #for path, im, im0s, vid_cap, s in dataset:
        #if USE_SEM4:    
            #sem4.acquire() #sem4=0
            #for path, im, im0s, vid_cap, s in dataset:
            #path, im, im0s, vid_cap, s = dataset
        
        get_frame_time = time.time()
        
        data = dataset.__next__()
        path, im, im0s, vid_cap, s = data
        #print("[Get_Frame] im shape {} :".format(im.shape))
        im = torch.from_numpy(im).to(device)
        #print("[Get_Frame] im torch.from_numpy shape {} :".format(im.shape))
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            #print("im is None")
            im = im[None]  # expand for batch dim
        #print("im.shape: {}".format(im.shape))
        #im_global = im
        #print("im_global shape : {}".format(im_global.shape))
        #path_global = path
        #im0s_global = im0s
        #s_global = s
        #vid_cap_global = vid_cap
            
        #=========Alister 2023-03-09 add global list=========
        #global get_frame_list_global
        #get_frame_list_global.append(im)
        #print("[Get_Frame]len(get_frame_list_global) : {}".format(len(get_frame_list_global)))
        #======Alister 2023-02-28 add queue================
        #========put get frame result to queue=============
        q1_time = time.time()
        #print("start get_frame_sem.acquire()")
        #======================
        #get_frame_sem.acquire()
        #======================
        get_frame_queue.put([im,path,im0s,s,vid_cap], block=True, timeout=None)#Root Cause : Here cost a lot of time 2023-03-02
        #get_frame_queue.put([im,path,im0s])#Root Cause : Here cost a lot of time 2023-03-02
        #print("start get_frame_sem.release()")
        #=======================
        #get_frame_sem.release()
        #=======================
        #time.sleep(0.001)
        during_q1_put = time.time() - q1_time
        if SHOW_QUEUE_TIME_LOG:
            print("[TIME_LOG]during_q1_put : {} ms".format(during_q1_put*1000))
        #print("1")
        
        
        global frame_cnt
        #==========Alister add 2023-03-02============
        if frame_cnt==1:
            #parameter_queue.put([s,vid_cap]) #Failed
            s_global = s
            vid_cap_global = vid_cap
        
        
        if frame_cnt%9999==0:
            frame_cnt=2
            
        frame_cnt+=1
        #print("[Get_Frame]get im done")
        #sem1.release() #sem1=1
        #print("[Get_Frame]sem1 after release: {}".format(sem1))
        #print(im_global.shape)
        #return im, path, s, im0s, vid_cap
        #return im_global
        #if USE_TIME:
            #time.sleep(set_time_1)
        
        during_get_frame = time.time() - get_frame_time
        # Alister aff 2023-03-08
        global during_get_frame_global
        during_get_frame_global = during_get_frame
        print("[Get_Frame]during_get_frame : {} ms".format(during_get_frame*1000))
        
        



def model_inference(model,visualize,save_dir,path,augment):
    #global pred_global
    #global model_global
    #global im_global
    #global get_frame_list_global
    pred_list = []
    
    #list_global_pop_im = None
    while True:
        model_inference_time = time.time()
        
        #=========Alister 2023-03-09 add global list=========
        
        #get_frame_list_global.reverse()
        #print("[model_inference]len(get_frame_list_global) : {}".format(len(get_frame_list_global)))
        #if len(get_frame_list_global)>0:
            #list_global_pop_im = get_frame_list_global.pop(0)
            #print("list_global_pop_im : {}".format(list_global_pop_im))
    
        
        #============get frame queue=============================
        q1_before_get = time.time()
        #if not get_frame_queue.empty():
        #print("start get_frame_sem.acquire()")
        #======================
        #get_frame_sem.acquire()
        #======================
        print("[before get] get_frame_queue.qsize() = {}".format(get_frame_queue.qsize()))
        #if not get_frame_queue.empty():
            #get_frame_data_from_queue = get_frame_queue.get(0)
        #else:
        get_frame_data_from_queue = get_frame_queue.get()
            
        print("[after get] get_frame_queue.qsize() = {}".format(get_frame_queue.qsize()))
    
        im_queue,path_queue,im0s_queue,s_queue,vid_cap_queue = get_frame_data_from_queue
        #print("start get_frame_sem.release()")
        #======================
        #get_frame_sem.release()
        #======================
        #im_queue,path_queue,im0s_queue = get_frame_data_from_queue
        #if len(get_frame_list_global) == 0:
            #list_global_pop_im = im_queue
        #print("[model_inference] sem1 befroe acquire: {}".format(sem1))
        during_q1_get = time.time() - q1_before_get
        if SHOW_QUEUE_TIME_LOG:
            print("[TIME_LOG]during_q1_get : {} ms".format(during_q1_get*1000))
        #sem1.acquire() #sem1=0
        #if USE_SEM5:
            #sem5.acquire()
        #sem5.acquire()
        #print("[model_inference] sem1 after acquire: {}".format(sem1))
        
        # Directories
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        #pred = model(im_queue, augment=augment, visualize=visualize)
        #if not list_global_pop_im==None:
            #print("[model_inference]im using list_global~~~~")
            #pred = model(list_global_pop_im, augment=augment, visualize=visualize)
        #else:
        pred = model(im_queue, augment=augment, visualize=visualize)
      
        #pred_global = pred
        
        #pred_list.append(pred)
        #pred_global = pred
        #=================Alister add 2023-02-28======================== 
        #======put model inference result to queue======================
        
        mi_qput_start_time = time.time()
        #model_inference_sem.acquire()
        my_queue.put([im_queue,path_queue,s_queue,vid_cap_queue,pred,im0s_queue],block=True, timeout=None)
        #my_queue.put([im_queue,path_queue,pred,im0s_queue])
        #print("[model_inference]pred_global = {}".format(pred_global))
        #return pre2
        #model_inference_sem.release()
        #time.sleep(0.001)
        during_mi_qput = time.time() - mi_qput_start_time
        if SHOW_QUEUE_TIME_LOG:
            print("[TIME_LOG]during_mi_qput : {} ms".format(during_mi_qput*1000))
        #print("2")
        
        during_model_inference = time.time() - model_inference_time
        global during_model_inference_global
        during_model_inference_global = during_model_inference
        print("[model_inference]during_model_inference : {} ms".format(during_model_inference*1000))
        
        #if USE_TIME:
            #time.sleep(set_time_2)
        
        #print("[model_inference] sem2 start release: {}".format(sem2))
        #sem2.release() #sem2=1
        #print("[model_inference] sem2 release done: {}".format(sem2))
        #if USE_SEM4:
            #sem4.release() #sem4=1
        #


        #

def nms(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det):
    pred_nms = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    #during_nms = time.time() - nms_time
    #print("during_nms : {} ms".format(during_nms*1000))
    return pred_nms

# will use this func in Run_inference function
def Process_Prediction(pred=None,
                       source = 0,
                       path='',
                       im0s='',
                       dataset = None,
                       s='',
                       save_dir='',
                       im = None,#im_global
                       save_crop = False,
                       line_thickness = 3,
                       names = '',
                       save_txt=False,
                       save_conf=False,
                       save_img=False,
                       view_img=False,
                       hide_labels=False,
                       hide_conf=False,
                       dt=None,
                       vid_cap=None,
                       vid_path=None,
                       vid_writer=None,
                       save_ai_result=False):
    
    
    #global vid_cap_global
    save_anomaly_img = False
    global anomaly_img_count
    global fr_cnt
    anomaly_img_count+=1
    seen, windows = 0, []
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    fr_cnt+=1
    for i, det in enumerate(pred):  # per image
        post_process_detail_time = time.time()
        seen += 1
        if webcam:  # batch_size >= 1
            print("webcam i = {}".format(i))
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            
            #s += f'{i}: '
        else:
            print("video")
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        #s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            #det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            #Alister add 2023-02-21 filter line label
            #===========================================================================================================================================================
            filter_line_label = False
            X_TH = 200
            Y_TH = 200
            frontline_x = None
            frontline_y = None
            
            line_x = None
            line_y = None
            #Filter line overlapped with frontline
            start_filter_line = time.time()
            #==========================================Filter line======================================================================================================
            for *xyxy, conf, cls in reversed(det):
                xyxy = torch.tensor(xyxy).view(-1, 4)
                b = xyxy2xywh(xyxy).long()  # boxes
                c = int(cls)  # integer class
                
                if c==3 or c==2 or c==1: #front line or others or noline
                    frontline_x = b[0,0]
                    frontline_y = b[0,1]
                if c==0: #line
                    line_x = b[0,0]
                    line_y = b[0,1]
                
                if not line_x==None  and not line_y==None and not frontline_x==None:
                    x_distance =  abs(frontline_x-line_x)
                    y_distance =  abs(frontline_y-line_y)
                    print("x_distance : {}".format(x_distance))
                    print("y_distance : {}".format(y_distance))
                    #input()
                    if float(x_distance) < X_TH and float(y_distance) < Y_TH:
                        filter_line_label = True
                        print("filter_line_label filter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfilter_line_labelfiltelfilter_line_labelfilter_line_label: {}".format(filter_line_label))
                        #input()
            print("filter_line_label : {}".format(filter_line_label))
            #===============================================================================================================================================
                
            #Alister add 2023-02-21 add time label
            #annotator.time_label(frame_count=fr_cnt,txt_color=(128,256,0))
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                #Alister add 2023-02-21    save anomaly images    
                c = int(cls)  # integer class
                if c==0 and conf<0.70 and filter_line_label==False: #test c==1 
                    save_anomaly_img = True
                    now = datetime.now()
                    s_time = datetime.strftime(now,'%y-%m-%d-%H-%M-%S')
                    #anomaly_img_count+=1
                    
                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    #annotator.box_label(xyxy, label, color=colors(c, True))
                    if c==0 and filter_line_label==False: #noline (test)
                        if conf<0.70:
                            annotator.box_label(xyxy, label+" anomaly" , color=(0,0,255))
                        else:
                            annotator.box_label(xyxy, label+" normal" , color=(255,0,0))
                    elif not c==0:
                        annotator.box_label(xyxy, label, color=colors(c, True))
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        # Stream results
        im0 = annotator.result()
        #print("im0 : {}".format(im0.shape))
        #print("[Process_Prediction]before view img {}".format(view_img))
        during_post_process_detail_time = time.time() - post_process_detail_time
        print("[Process_Prediction]during_post_process_detail_time : {} ms".format(during_post_process_detail_time*1000))
        if view_img:
            if platform.system() == 'Linux' and p not in windows:
                #print("[Process_Prediction] in if ")
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                #cv2.namedWindow("test")  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                #cv2.resizeWindow("test", im0.shape[1], im0.shape[0])
                #print("[Process_Prediction] end if ")
            #print("[Process_Prediction] before imshow")
            cv2.imshow(str(p), im0)
            #cv2.imshow("test", im0)
            cv2.waitKey(1)  # 1 millisecond
            #print("[Process_Prediction] after imshow")
        #print("[Process_Prediction]after view img")
        
    #raise NotImplemented
    # Print time (inference-only)
        # Save results (image with detections)
        #print("save img : {}".format(save_img))
        save_img_time = time.time()
        if save_ai_result:
        
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                            vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            print("vidcap w:{} h:{}".format(w,h))
                        else:  # stream
                            fps, w, h = FPS_SET, im0.shape[1], im0.shape[0]
                            #fps, w, h = FPS_SET, 1600, 900
                            print("stream w:{} h:{}".format(w,h))
                            #fps, w, h = FPS_SET, SET_W, SET_H
                            #save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            #vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            
                        save_path = str(Path(save_path).with_suffix('.avi'))  # force *.mp4 suffix on results videos
                        #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
                        #print("start print save path")
                        #save_path = str(Path(save_path).with_suffix('.avi'))  # force *.mp4 suffix on results videos
                        #print("save_path : {}".format(save_path))
                        #vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
                        #vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('H', '2', '6', '4'), fps, (w, h),True)
                    vid_writer[i].write(im0)
                    
        if save_anomaly_img:
            img_dir = os.path.dirname(save_path)
            img_name = s_time + '.jpg'
            folder_name = 'anomaly_img'
            if not os.path.exists(os.path.join(img_dir,folder_name)):
                os.makedirs(os.path.join(img_dir,folder_name))
            img_path = os.path.join(img_dir,folder_name,img_name)
            #im0_resize = cv2.resize(im0,(1920,1080))
            #cv2.imwrite(img_path, im0_resize)
            cv2.imwrite(img_path, im0)
        
        during_save_img = time.time() - save_img_time
        print("[Process_Prediction]during_save_img: {} ms".format(during_save_img*1000))
        
        '''
        Save_Result(save_img=save_img,
                        dataset=dataset,
                        save_path=save_path,
                        im0=im0,
                        i=i,
                        vid_cap=vid_cap
                        )
        '''
    #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{(dt[0].dt + dt[1].dt + dt[2].dt) * 1E3:.1f}ms")
    return save_path, im0


def PostProcess(
        pred, conf_thres, iou_thres, classes, agnostic_nms, max_det,
                source,
                path,
                im0s,
                dataset,
                s,
                save_dir,
                im,
                save_crop,
                line_thickness,
                names,
                save_txt,
                save_conf,
                save_img,
                view_img,
                hide_labels,
                hide_conf,
                dt,
                vid_cap,
                vid_path,
                vid_writer,
                save_ai_result):
    #global pred_global
    #global im0s_global
    #global path_global
    #global im_global
    #global pred_global
    
    #cv2.namedWindow("test", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    #cv2.resizeWindow("test", im0.shape[1], im0.shape[0])
    #pred_global_new = []
    global s_global
    global vid_cap_global 
    while True:
        post_process_time = time.time()
        #print("[PostProcess] sem2 before acquire")
        
        #sem2.acquire() #sem2=0
        #=======Alister add 2023-02-27=============
        #model_inference_sem.acquire()
        print("[before get] my_queue.qsize() = {}".format(my_queue.qsize()))
        #if not my_queue.empty():
            #q_data=my_queue.get(0)
        #else:
        q_data=my_queue.get()
        print("[after get] my_queue.qsize() = {}".format(my_queue.qsize()))
        im_from_queue,path_from_queue,s_from_queue,vid_cap_from_queue,pred_from_queue,im0s_from_queue = q_data
        #im_from_queue,path_from_queue,pred_from_queue,im0s_from_queue = q_data
        #model_inference_sem.release()
        #Alister add 2023-03-02 #Failed
        #====================================================
        #parameter_data = parameter_queue.get()
        #s_from_queue,vid_cap_from_queue = parameter_data
        #====================================================
        
        #my_queue.task_done()
        #==========================================
        #print("[PostProcess]pred_global: {}".format(pred_global))
        #pred_global.reverse()
        #print("[PostProcess]pred_global.reverse() : {}".format(pred_global))
        
        
        #print("[PostProcess] sem2 after acquire")
        nms_time = time.time()
        #print("[PostProcess] before nms")
        #pred_in = pred_global.pop()
        #print("[PostProcess]pred_in: {}".format(pred_in))
        pred = nms(pred_from_queue, conf_thres, iou_thres, classes, agnostic_nms, max_det)
        during_nms = time.time() - nms_time
        print("[PostProcess]during_nms : {} ms".format(during_nms*1000))
        #print("[PostProcess] after nms")
        #print("pred : {}".format(pred))
        
        #print("[PostProcess] before Process_Prediction")
        save_path, im0 = Process_Prediction(
                            pred=pred,
                            source = source,
                            path=path_from_queue,
                            im0s=im0s_from_queue,
                            dataset = dataset,
                            s=s_from_queue,#s_from_queue,
                            save_dir=save_dir,
                            im =im_from_queue,
                            save_crop=False,
                            line_thickness=3,
                            names=names,
                            save_txt=save_txt,
                            save_conf=save_conf,
                            save_img=save_img,
                            view_img=view_img,
                            hide_labels=hide_labels,
                            hide_conf=hide_conf,
                            dt=dt,
                            vid_cap=vid_cap_from_queue,
                            vid_path=vid_path,
                            vid_writer=vid_writer,
                            save_ai_result=save_ai_result)
        #print("[PostProcess] after Process_Prediction")
        #print("3")
        #===============================================================================================================================
        during_post_process = time.time() - post_process_time
        #print("=======================================================================================")
        
        #Alister add 2023-03-08
        global during_get_frame_global
        global during_model_inference_global
        
        print("[Global]get_frame:{}ms".format(during_get_frame_global*1000))
        print("[Global]model_inference:{}ms".format(during_model_inference_global*1000))
        Max_time = 0
        if during_model_inference_global>during_get_frame_global:
            Max_time = during_model_inference_global
        else:
            Max_time = during_get_frame_global
        
        if during_post_process>Max_time:
            Max_time = during_post_process
        
        global Total_postprocess_time
        global frame_count_global
        Total_postprocess_time+=Max_time
        #Total_postprocess_time+=during_post_process
        Avg_postprocess_time = Total_postprocess_time/frame_count_global
        FPS = int(1000.0/float(Avg_postprocess_time*1000))
        frame_count_global+=1
        print("[PostProcess]during_post_process:{}ms".format(during_post_process*1000))
        print("[Global]=============================================Max_time:{}ms=======================================".format(Max_time*1000))
        print("=======f:{}=Total time:{}ms==========[PostProcess]Avg_one_frame_time: {} ms====FPS:{}==================".format(frame_count_global,
                                                                                                                         Total_postprocess_time*1000,
                                                                                                                         Avg_postprocess_time*1000,
                                                                                                                               FPS))
        #===============================================================================================================================
        #during_post_process = time.time() - post_process_time
        #print("[PostProcess]during_post_process: {} ms".format(during_post_process*1000))
        #if USE_SEM5:
            #sem5.release()
        #if USE_TIME:
            #time.sleep(set_time_3)
    


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    #main(opt)
    weights = opt.weights
    data = opt.data
    imgsz = opt.imgsz
    save_conf = opt.save_conf
    nosave = opt.nosave
    view_img = opt.view_img
    hide_labels = opt.hide_labels
    hide_conf = opt.hide_conf
    source = opt.source
    save_ai_result = opt.save_airesult
    
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    
    model, stride, pt, imgsz, device, names = load_model(weights=weights,  # model.pt path(s)
                                                   device='',
                                                   dnn=False,
                                                   data=data,
                                                   half=False,
                                                   imgsz=imgsz)
    
    
    #model_global = model
    
    dataset, bs, source = load_dataloader(source=source,
                                    nosave=False,
                                    imgsz=imgsz,
                                    stride=stride,
                                    pt=pt
                                    )
    
    # Block until all tasks are done.
    #my_queue.join()
    #================================
    '''
    # 建立一個子執行緒
    t = threading.Thread(target = job)
    
    # 執行該子執行緒
    t.start()
    
    # 主執行緒繼續執行自己的工作
    #for i in range(3):
      #print("Main thread:", i)
      #time.sleep(1)
    
    # 等待 t 這個子執行緒結束
    t.join()
    '''
    #https://stackoverflow.com/questions/31508574/semaphores-on-python
    #================================
    name = opt.name
    project = opt.project
    exist_ok = opt.exist_ok
    save_txt = opt.save_txt
    visualize = opt.visualize
    augment = opt.augment
    conf_thres = opt.conf_thres
    iou_thres = opt.iou_thres
    classes = opt.classes
    agnostic_nms = opt.agnostic_nms
    max_det = opt.max_det
    save_crop = opt.save_crop
    line_thickness = opt.line_thickness
    
    bs = len(dataset)  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
        #==================================================================================================
    if MULTI_THREAD:
        #================multi thread code=================================================================
        print("Thread count: {}".format(threading.active_count()))
        print("threading.enumerate() :{}".format(threading.enumerate() ))
        #for path, im, im0s, vid_cap, s in dataset:
        #path_global = None
        if THREE_THREADS: #using 3 threads
            #=================
            #thread Get_Frame
            #=================
            print("before t1")
            t1 = threading.Thread(target = Get_Frame ,args=(dataset, ))
            with dt[0]:
                t1.start()    
            print("after t1.start()")
            
            #global path_global
            #======================
            #thread model_inference
            #=======================
            print("before t2")
            t2 = threading.Thread(target = model_inference,args=(model, visualize,save_dir,None,augment,))
            with dt[1]:
                t2.start()
            print("after t2.start()")
            #
            
            #========================
            #thread PostProcess
            #========================
            print("before t3")
            t3 = threading.Thread(target = PostProcess, args=(
                                                                None,#pred_global
                                                              conf_thres, 
                                                              iou_thres, 
                                                              classes, 
                                                              agnostic_nms, 
                                                              max_det,
                                                            source,
                                                            None, #path_global
                                                            None, #im0s_global
                                                            dataset,
                                                            None, #s_global
                                                            save_dir,
                                                            None,#im_global
                                                            save_crop,
                                                            line_thickness,
                                                            names,
                                                            save_txt,
                                                            save_conf,
                                                            save_img,
                                                            view_img,
                                                            hide_labels,
                                                            hide_conf,
                                                            dt,
                                                            None,#vid_cap_global
                                                            vid_path,
                                                            vid_writer,
                                                            save_ai_result,) )
            
            print("Thread count: {}".format(threading.active_count()))
            with dt[2]:
                t3.start()
            print("after t3.start()")
        
        
            #===============================================================================================================
    
    #vid_cap_global.release()
    #vid_writer.release()
    
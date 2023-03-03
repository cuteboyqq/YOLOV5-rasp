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
FPS_SET=22
SET_W=1280
SET_H=720
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
anomaly_img_count=0
sem1 = threading.Semaphore(0)
sem2 = threading.Semaphore(0)
sem3 = threading.Semaphore(0)
sem4 = threading.Semaphore(1)
sem5 = threading.Semaphore(1)

# 建立佇列
get_frame_queue = queue.Queue(30)
my_queue = queue.Queue(30)
parameter_queue = queue.Queue(30)
MULTI_PROCESS=False
MULTI_THREAD=True
frame_cnt = 1
#@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
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
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                annotator.time_label()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        annotator.time_label()
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
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
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.avi'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default= r'/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/train/f192_2022-12-29-4cls/weights/best.pt', help='model path(s)')
    #parser.add_argument('--source', type=str, default=r'/home/ali/factory_video/ori_video_ver2.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=r'/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/data/factory_new2.yaml', help='(optional) dataset.yaml path')
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
def Run_inference(model='',
                  pt='',
                  bs=64,
                  imgsz=(192,192),
                  dataset=None,
                  device='',
                  visualize=False,
                  project = ROOT / 'runs/detect',
                  name='exp',
                  names='',
                  source = '',
                  exist_ok=False,
                  save_txt=False,
                  augment=False,
                  conf_thres=0.25,
                  classes=None,
                  iou_thres=0.45,
                  agnostic_nms=False,
                  max_det=1000,
                  save_conf=False,
                  save_img=False,
                  view_img=False,
                  hide_labels=False,
                  hide_conf=False):
    
    bs = len(dataset)  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = Get_Frame(im)
            #im = torch.from_numpy(im).to(device)
            #im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            #im /= 255  # 0 - 255 to 0.0 - 1.0
            #if len(im.shape) == 3:
                #im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model_inference(visualize,save_dir,path,im,augment)
            #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            #pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = nms(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det)
            #pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        save_path, im0 = Process_Prediction(pred=pred,
                            source = source,
                            path=path,
                            im0s=im0s,
                            dataset = dataset,
                            s=s,
                            save_dir=save_dir,
                            im =im,
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
                            vid_cap=vid_cap,
                            vid_path=vid_path,
                            vid_writer=vid_writer)
        
    #raise NotImplemented

def Get_Frame_and_model_Inference(my_queue,
                                  model,
                                  dataset,
                                  visualize,
                                  save_dir,
                                  path,
                                  augment):
    #global im_global
    #global path_global
    #global im0s_global
    #global s_global
    #global vid_cap_global
    #global pred_global
    #global model_global
    #global im_global
    #path, im, im0s, vid_cap, s = dataset
    #print(dataset)
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
        
        raw_im = im
        
        im = torch.from_numpy(im).to(device)
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
        
        
        
        print("1")
        
        during_get_frame = time.time() - get_frame_time
        print("during_get_frame : {} ms".format(during_get_frame*1000))
        #sem1.release() #sem1=1
        #print("[Get_Frame]get im done")
        #print("[Get_Frame]sem1 after release: {}".format(sem1))
        #print(im_global.shape)
        #return im, path, s, im0s, vid_cap
        #return im_global
        if USE_TIME:
            time.sleep(set_time_1)
            
        #sem1.acquire() #sem1=0
        #if USE_SEM5:
            #sem5.acquire()
        #sem5.acquire()
        #print("[model_inference] sem1 after acquire: {}".format(sem1))
        model_inference_time = time.time()
        # Directories
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        
        #pred_global = pred
        
        #pred_list.append(pred)
        #pred_global = pred
        #=================Alister add 2023-02-27======================
        my_queue.put([im,path,s,vid_cap,pred,im0s])
        #print("[model_inference]pred_global = {}".format(pred_global))
        #return pre2
        
        print("2")
        
        during_model_inference = time.time() - model_inference_time
        print("during_model_inference : {} ms".format(during_model_inference*1000))
        
        SAVE_RAW_STREAM = False
        if SAVE_RAW_STREAM:
            #===========================================save raw stream ==============================================================
            save_img_time = time.time()
            output_path = '/home/ali/Desktop/2023-02-20.avi'
            '''
            vc = cv2.VideoCapture(s)
            
            vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = vc.get(cv2.CAP_PROP_FPS)
            print("w : {}, h:{}".format(w,h))
            ret, frame = vc.read()
            '''
            
            
            if vid_cap:  # video
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #w = 1280
                #h = 720
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # stream
                fps, w, h = FPS_SET, raw_im.shape[2], raw_im.shape[3]
                #fps, w, h = 20, 1280,720
                print("w : {}, h:{}".format(w,h))
            
            #vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            #vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            #ret, frame = vc.read()
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #fps = vid_cap.get(cv2.CAP_PROP_FPS)
            '''
            if platform.system() == 'Linux':
                #print("[Process_Prediction] in if ")
                #windows.append(p)
                cv2.namedWindow('raw_im', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                #cv2.namedWindow("test")  # allow window resize (Linux)
                cv2.resizeWindow('raw_im', raw_im.shape[1], raw_im.shape[0])
                cv2.imshow('raw_im', raw_im)
                cv2.waitKey(1)  # 1 millisecond
            '''
            fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
            # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
            # fourcc = cv2.VideoWriter_fourcc('H', 'E', 'V', 'C')
            vw = cv2.VideoWriter('/home/ali/Desktop/2023-02-20.avi', fourcc, fps, (w, h), True)
            #while ret:
            vw.write(raw_im)
                #ret, frame = vc.read()
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #cv2.imshow('frame', frame)
                #if cv2.waitKey(5) & 0xFF == ord('q'):
                    #cv2.destroyAllWindows()
                    #return -1

                    
            during_save_img = time.time() - save_img_time
            print("[Get_Frame_and_model_Inference]during_save_img: {} ms".format(during_save_img*1000))
            #==========================================================================================================================
        
        
        if USE_TIME:
            time.sleep(set_time_2)
        
        #print("[model_inference] sem2 start release: {}".format(sem2))
        #sem2.release() #sem2=1
        #print("[model_inference] sem2 release done: {}".format(sem2))
        #if USE_SEM4:
            #sem4.release() #sem4=1
            
            

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
        #======Alister 2023-02-28 add queue================
        #========put get frame result to queue=============
        q1_time = time.time()
        get_frame_queue.put([im,path,im0s,s,vid_cap])#Root Cause : Here cost a lot of time 2023-03-02
        #get_frame_queue.put([im,path,im0s])#Root Cause : Here cost a lot of time 2023-03-02
        #get_frame_queue.put(path)#Root Cause : Here cost a lot of time 2023-03-02
        during_q1_put = time.time() - q1_time
        print("[TIME_LOG]during_q1_put : {} ms".format(during_q1_put*1000))
        #print("1")
        
        during_get_frame = time.time() - get_frame_time
        print("[Get_Frame]during_get_frame : {} ms".format(during_get_frame*1000))
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

def Get_Frame_proc(dataset):
    
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
        im = torch.from_numpy(im).to(device)
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
        #======Alister 2023-02-28 add queue================
        #========put get frame result to queue (process queue)=============
        get_frame_proc_queue.put([im,path,im0s]) #Root Cause : Here cost a lot of time 2023-03-02
        #Root Cause : Here cost a lot of time Root Cause : Here cost a lot of time Root Cause : Here cost a lot of time
        print("1")
        
        during_get_frame = time.time() - get_frame_time
        print("during_get_frame : {} ms".format(during_get_frame*1000))
        
        #print("[Get_Frame]get im done")
        #sem1.release() #sem1=1
        #print("[Get_Frame]sem1 after release: {}".format(sem1))
        #print(im_global.shape)
        #return im, path, s, im0s, vid_cap
        #return im_global
        if USE_TIME:
            time.sleep(set_time_1)


def model_inference(model,visualize,save_dir,path,augment):
    #global pred_global
    #global model_global
    #global im_global
    pred_list = []
    while True:
        #============get frame queue=============================
        q1_before_get = time.time()
        get_frame_data_from_queue = get_frame_queue.get()
        im_queue,path_queue,im0s_queue,s_queue,vid_cap_queue = get_frame_data_from_queue
        #im_queue,path_queue,im0s_queue = get_frame_data_from_queue
        #path_queue = get_frame_data_from_queue
        #print("[model_inference] sem1 befroe acquire: {}".format(sem1))
        during_q1_get = time.time() - q1_before_get
        print("[TIME_LOG]during_q1_get : {} ms".format(during_q1_get*1000))
        #sem1.acquire() #sem1=0
        #if USE_SEM5:
            #sem5.acquire()
        #sem5.acquire()
        #print("[model_inference] sem1 after acquire: {}".format(sem1))
        model_inference_time = time.time()
        # Directories
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im_queue, augment=augment, visualize=visualize)
      
        #pred_global = pred
        
        #pred_list.append(pred)
        #pred_global = pred
        #=================Alister add 2023-02-28======================== 
        #======put model inference result to queue======================
        #my_queue.put([im_queue,path_queue,s_queue,vid_cap_queue,pred,im0s_queue])
        mi_qput_start_time = time.time()
        my_queue.put([im_queue,path_queue,s_queue,vid_cap_queue,pred,im0s_queue])
        #my_queue.put([im_queue,path_queue,pred,im0s_queue])
        #print("[model_inference]pred_global = {}".format(pred_global))
        #return pre2
        during_mi_qput = time.time() - mi_qput_start_time
        print("[TIME_LOG]during_mi_qput : {} ms".format(during_mi_qput*1000))
        #print("2")
        
        during_model_inference = time.time() - model_inference_time
        print("[model_inference]during_model_inference : {} ms".format(during_model_inference*1000))
        
        #if USE_TIME:
            #time.sleep(set_time_2)
        
        #print("[model_inference] sem2 start release: {}".format(sem2))
        #sem2.release() #sem2=1
        #print("[model_inference] sem2 release done: {}".format(sem2))
        #if USE_SEM4:
            #sem4.release() #sem4=1
        #

def model_inference_proc(model,visualize,save_dir,path,augment):
    #global pred_global
    #global model_global
    #global im_global
    pred_list = []
    while True:
        #============get frame queue (process queue)=============================
        get_frame_data_from_proc_queue = get_frame_proc_queue.get()
        im_queue,path_queue,im0s_queue,s_queue,vid_cap_queue = get_frame_data_from_proc_queue
        #print("[model_inference] sem1 befroe acquire: {}".format(sem1))
        
        #sem1.acquire() #sem1=0
        #if USE_SEM5:
            #sem5.acquire()
        #sem5.acquire()
        #print("[model_inference] sem1 after acquire: {}".format(sem1))
        model_inference_time = time.time()
        # Directories
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im_queue, augment=augment, visualize=visualize)
      
        #pred_global = pred
        
        #pred_list.append(pred)
        #pred_global = pred
        #=================Alister add 2023-02-28======================== 
        #======put model inference result to queue (process queue)======================
        my_proc_queue.put([im_queue,path_queue,s_queue,vid_cap_queue,pred,im0s_queue])
        #print("[model_inference]pred_global = {}".format(pred_global))
        #return pre2
        
        print("2")
        
        during_model_inference = time.time() - model_inference_time
        print("during_model_inference : {} ms".format(during_model_inference*1000))
        
        #if USE_TIME:
            #time.sleep(set_time_2)
        
        #print("[model_inference] sem2 start release: {}".format(sem2))
        #sem2.release() #sem2=1
        #print("[model_inference] sem2 release done: {}".format(sem2))
        #if USE_SEM4:
            #sem4.release() #sem4=1
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
                       vid_writer=None):
    
    
    #global vid_cap_global
    save_anomaly_img = False
    global anomaly_img_count
    anomaly_img_count+=1
    seen, windows = 0, []
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    for i, det in enumerate(pred):  # per image
        post_process_detail_time = time.time()
        seen += 1
        if webcam:  # batch_size >= 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            #s += f'{i}: '
        else:
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
            annotator.time_label()
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
        if SAVE_AI_RESULT_STREAM:
        
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
                            vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, SET_W)
                            vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SET_H)
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


def PostProcess(my_queue,
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
                vid_writer):
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
        #print("[PostProcess] sem2 before acquire")
        
        #sem2.acquire() #sem2=0
        #=======Alister add 2023-02-27=============
        q_data=my_queue.get()
        im_from_queue,path_from_queue,s_from_queue,vid_cap_from_queue,pred_from_queue,im0s_from_queue = q_data
        #im_from_queue,path_from_queue,pred_from_queue,im0s_from_queue = q_data
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
        
        post_process_time = time.time()
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
                            vid_writer=vid_writer)
        #print("[PostProcess] after Process_Prediction")
        #print("3")
        #===============================================================================================================================
        during_post_process = time.time() - post_process_time
        #print("=======================================================================================")
        global Total_postprocess_time
        global frame_count_global
        Total_postprocess_time+=during_post_process
        Avg_postprocess_time = Total_postprocess_time/frame_count_global
        FPS = int(1000.0/float(Avg_postprocess_time*1000))
        frame_count_global+=1
        print("==================[PostProcess]during_post_process: {} ms======================".format(during_post_process*1000))
        print("=======f:{}=Total time:{}==========[PostProcess]Avg_postprocess_time: {} ms====FPS:{}==================".format(frame_count_global,
                                                                                                                         Total_postprocess_time,
                                                                                                                         Avg_postprocess_time*1000,
                                                                                                                               FPS))
        #===============================================================================================================================
        #during_post_process = time.time() - post_process_time
        #print("[PostProcess]during_post_process: {} ms".format(during_post_process*1000))
        #if USE_SEM5:
            #sem5.release()
        #if USE_TIME:
            #time.sleep(set_time_3)
            
            
def PostProcess_proc(my_queue,
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
                vid_writer):
    #global pred_global
    #global im0s_global
    #global path_global
    #global im_global
    #global pred_global
    
    #cv2.namedWindow("test", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    #cv2.resizeWindow("test", im0.shape[1], im0.shape[0])
    #pred_global_new = []
    while True:
        #print("[PostProcess] sem2 before acquire")
        
        #sem2.acquire() #sem2=0
        #=======Alister add 2023-02-27 (process queue)=============
        q_proc_data=my_proc_queue.get()
        im_from_queue,path_from_queue,s_from_queue,vid_cap_from_queue,pred_from_queue,im0s_from_queue = q_proc_data
        my_queue.task_done()
        #==========================================
        #print("[PostProcess]pred_global: {}".format(pred_global))
        #pred_global.reverse()
        #print("[PostProcess]pred_global.reverse() : {}".format(pred_global))
        
        post_process_time = time.time()
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
                            s=s_from_queue,
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
                            vid_cap=vid_cap,
                            vid_path=vid_path,
                            vid_writer=vid_writer)
        #print("[PostProcess] after Process_Prediction")
        print("3")
        
        during_post_process = time.time() - post_process_time
        print("[PostProcess]during_post_process: {} ms".format(during_post_process*1000))
        if USE_SEM5:
            sem5.release()
        if USE_TIME:
            time.sleep(set_time_3)
    
    

# will use this func in Run_inference function
def Save_Result(save_img=True,
                dataset=None,
                save_path="",
                im0=None,
                i=0,
                vid_cap=None
                ):
    bs = len(dataset)  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
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
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im0)
    
    #raise NotImplemented

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
    if MULTI_PROCESS:
        #================Alister add 2023-02-28 multi process code================================================================
        print("before p1")
        '''
        p1 = Process(target = Get_Frame_and_model_Inference ,args=(my_queue, 
                                                                            dataset,
                                                                            visualize,
                                                                            save_dir,
                                                                            None,augment,))
        with dt[0]:
            p1.start()    
        print("after p1.start()")
        '''
        #=================
        #process Get_Frame
        #=================
        p1 = Process(target = Get_Frame_proc ,args=(dataset, ))
        with dt[0]:
            p1.start()    
        print("after p1.start()") 
        #======================
        #process model_inference
        #=======================
        print("before p2")
        p2 = threading.Thread(target = model_inference_proc,args=(model,visualize,save_dir,None,augment,))
        with dt[1]:
            p2.start()
        print("after p2.start()")
        #======================
        #process PostProcess
        #=======================
        print("before p3")
        p3 = Process(target = PostProcess_proc, args=(my_proc_queue,
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
                                                        vid_writer,) )
        
        #print("Thread count: {}".format(threading.active_count()))
        with dt[2]:
            p3.start()
        print("after p3.start()")
        #==================================================================================================
    if MULTI_THREAD:
        #================multi thread code=================================================================
        print("Thread count: {}".format(threading.active_count()))
        print("threading.enumerate() :{}".format(threading.enumerate() ))
        #for path, im, im0s, vid_cap, s in dataset:
        #path_global = None
        
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
        
        
        '''
        #===========combine Get_frame & model_inference============================================
        print("before t1")
        t1 = threading.Thread(target = Get_Frame_and_model_Inference ,args=(my_queue,
                                                                            model,
                                                                            dataset,
                                                                            visualize,
                                                                            save_dir,
                                                                            None,augment,))
        with dt[0]:
            t1.start()    
        print("after t1.start()")
        '''
        #========================
        #thread PostProcess
        #========================
        print("before t3")
        t3 = threading.Thread(target = PostProcess, args=(my_queue,
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
                                                        vid_writer,) )
        
        print("Thread count: {}".format(threading.active_count()))
        with dt[2]:
            t3.start()
        print("after t3.start()")
        
        
        #t1.join()
        #t2.join()
        #t3.join()
        #===============================================================================================================
    
    #vid_cap_global.release()
    #vid_writer.release()
    '''
    #for path, im, im0s, vid_cap, s in dataset:
    with dt[0]:
        #im, path, s, im0s, vid_cap = Get_Frame(im,dataset)
        Get_Frame(dataset)
    with dt[1]:
        model_inference(visualize,save_dir,path_global,augment)
    with dt[2]:
        PostProcess(pred_global, conf_thres, iou_thres, classes, agnostic_nms, max_det,
                        source,
                        path_global,
                        im0s_global,
                        dataset,
                        s_global,
                        save_dir,
                        im_global,
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
                        vid_cap_global,
                        vid_path,
                        vid_writer)
    
        
        
        pred = nms(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det)
    
        save_path, im0 = Process_Prediction(pred=pred,
                            source = source,
                            path=path,
                            im0s=im0s,
                            dataset = dataset,
                            s=s,
                            save_dir=save_dir,
                            im =im,
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
                            vid_cap=vid_cap,
                            vid_path=vid_path,
                            vid_writer=vid_writer)
    '''
    #LOGGER.info(f"{s_global}{(dt[0].dt + dt[1].dt + dt[2].dt) * 1E3:.1f}ms")
        
    
    '''
    Run_inference(model=model,
                      pt=pt,
                      bs=bs,
                      imgsz=imgsz,
                      dataset=dataset,
                      device=device,
                      visualize=False,
                      project = ROOT / 'runs/detect',
                      name='exp',
                      names=names,
                      source=source,
                      exist_ok=False,
                      save_txt=False,
                      augment=False,
                      conf_thres=0.25,
                      classes=None,
                      iou_thres=0.45,
                      agnostic_nms=False,
                      max_det=1000,
                      save_conf=save_conf,
                      save_img=save_img,
                      view_img=view_img,
                      hide_labels=hide_labels,
                      hide_conf=hide_conf)
    '''
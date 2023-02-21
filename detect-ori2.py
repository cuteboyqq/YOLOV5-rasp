# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
'''
python detect.py --weights /home/ali/Desktop/GANomaly-tf2/export_model/2023-01-03/multiple_models/best-int8_edgetpu.tflite
--img-size 192 --data data/factory_new2.yaml
--GANOMALY --LOSS 2.0 --isize 32
--source /home/ali/Desktop/Produce_1221_720P_60FPS_SHORT.mp4  --name 2023-01-11 --view-img
'''
'''python detect.py --weights /home/ali/Desktop/GANomaly-tf2/export_model/2022-12-21/multiple_models/best-int8_edgetpu.tflite --data data/factory_new2.yaml --img-size 192 --GANOMALY --LOSS 2.5 --isize 32 --source 0 --view-img'''
import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box, get_crop_image
from utils.torch_utils import select_device, smart_inference_mode

from models.model_GANomaly_edgetpu import *

import datetime
import time
import threading
import queue

@smart_inference_mode()
def run(weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
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
        vid_stride=1,  # video frame-rate stride
        isize=32,
        nz=100,
        nc=3,
        ndf=64,
        ngf=64,
        w_adv=1,
        w_con=50,
        w_enc=1,
        extralayers=0,
        encdims=None,
        dataset=r'/home/ali/GitHub_Code/YOLO/YOLOV5-old/runs/detect/f_384_2min/crops_1cls',
        batch_size=64,
        lr=2e-4,
        beta1=0.5,
        GANOMALY=False, #use GANoamly defeat detection
        LOSS=4,
        ganomaly_save_img=False,
        ganomaly_log=False,
        time_log=False,
        ganomaly_conf=0.80
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    #print('Start load model')
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    #print('load model successfull')
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    #===========================
    # Alister 2022-10-13 add 
    # Create GANomaly instance
    #==========================
    frame_count = 0
    line_count = 0
    noline_count = 0
    normal_correct_detect_count = 0
    abnormal_correct_detect_count = 0
    abnormal_false_trigger_count = 0
    abnormal_miss_noline_count = 0
    #opt_tf = FLAGS
    if GANOMALY:
        frame_count = 0
        line_count = 0
        noline_count = 0
        normal_correct_detect_count = 0
        abnormal_correct_detect_count = 0
        abnormal_false_trigger_count = 0
        abnormal_miss_noline_count = 0
        cc=1
        '''
        ganomaly = GANomaly(opt,
                            train_dataset=opt.dataset,
                            valid_dataset=opt.dataset,
                            test_dataset=None)
        w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-20221109_edgetpu.tflite'
        interpreter = ganomaly.load_model_tflite(w, tflite=True, edgetpu=False)
        '''
        print('isize : {}'.format(isize))
        ganomaly = GANomaly_Detect(model_dir=r'/home/ali/Desktop/GANomaly-tf2/export_model/2023-01-03/multiple_models',
                                    model_file=r'ckpt-32-nz100-ndf64-ngf64-20230103-prelu-upsample-G-int8_edgetpu.tflite',
                                    save_image=ganomaly_save_img,
                                    show_log=ganomaly_log,
                                    show_time_log=time_log,
                                    tflite=False,
                                    edgetpu=True,
                                    isize=isize)
        #interpreter = ganomaly.get_interpreter()
        print('[detect.py] Start ganomaly.return_interpreter')
        interpreter = ganomaly.return_interpreter
        print('[detect.py] Finish ganomaly.return_interpreter')
        
    # Dataloader
    start_load_stream = time.time()
    bs = 1  # batch_size
    if webcam:
        #alister 2023-01-12 noted
        #view_img = check_imshow()
        
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        #print('Start LoadImages')
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        #print('LoadImages successful')
    vid_path, vid_writer = [None] * bs, [None] * bs
    during_load_stream = time.time() - start_load_stream
    print("[detect.py]during_load_stream : {}".format(float(during_load_stream*1000)))
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    #instablish semaphore 
    semaphore = threading.Semaphore(1)
    #====================================handle each frame=====================================================================================================================
    for path, im, im0s, vid_cap, s in dataset:
        print("[detect.py]===========Start detect one frame.===============")
        start_detect_one_frame_time = time.time()
        start_yolo = time.time()
        
        semaphore.acquire()
        #===================================Yolo inference============================================================================================
        with dt[0]:
            start_yolo_preprocess = time.time()
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            #im = cv2.cvtColor(im.numpy(), cv2.COLOR_BGR2RGB)
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            during_yolo_preprocess = time.time() - start_yolo_preprocess
            #print("[detect.py]during_yolo_preprocess: {}".format(float(during_yolo_preprocess)*1000))
        # Inference
        with dt[1]:
            #print('Start inference model')
            start_yolo_infer = time.time()
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            during_yolo_infer = time.time() - start_yolo_infer
            #print("[detect.py]during_yolo_infer : {}".format(float(during_yolo_infer)*1000))
            #print('inference model successful')
        # NMS
        with dt[2]:
            #print('Start NMS')
            start_nms = time.time()
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            during_nms = time.time() - start_nms
            #print("[detect.py]during_nms : {}".format(float(during_nms)*1000))
            #print('NMS done')
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        #==================================================================================================================================================
        if time_log:
            during_yolo = time.time() - start_yolo #start_yolo is defined at line 195
            print("[detect.py]during_yolo : {}".format(float(during_yolo)*1000))
        #acquire release 2023-01-16
        semaphore.release()
        #acquire semaphore 2023-01-16
        semaphore.acquire()
        # Process predictions
        for i, det in enumerate(pred):  # per image
            start_process_predictions = time.time()
            #=======================================process predictions=========================================================================================
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            
            
            frame_count+=1
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #===========================================================================================================================================================
            if time_log:
                during_start_process_predictions = time.time() - start_process_predictions
                print("[detect.py]during_start_process_predictions: {}".format(float(during_start_process_predictions)*1000))
            start_copy_img = time.time()
            imc = im0.copy() if save_crop else im0  # for save_crop
            #imc_ganomaly = im0.copy()
            if time_log:
                during_copy_img = time.time() - start_copy_img
                print("[detect.py]during_copy_img : {}".format(float(during_copy_img)*1000))
            
            start_initial_annotator = time.time()
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if time_log:
                during_initial_annotator = time.time() - start_initial_annotator
                print("[detect.py]during_initial_annotator: {}".format(float(during_initial_annotator*1000)))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
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
                if time_log:
                    during_filter_line = time.time() -start_filter_line
                    print("[detect.py]during_filter_line : {}".format(float(during_filter_line)*1000))
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    
                    if save_crop:
                        #Alister add 2022-11-08
                        #print('frame : {}'.format(int(frame)))
                        #if int(c)==0:
                        c = int(cls)  # integer class
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    #t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
                    #print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
                    #==================================
                    # GANomaly Alister 2022-10-13 add
                    #==================================
                    start_time_ganomaly = time.time()
                    if GANOMALY:
                        #start_time_ganomaly = datetime.datetime.now()
                        
                        abnormal = 0
                        #imc2 = im0.copy()
                        c = int(cls)
                        
                        if c==0 and float(conf) < float(ganomaly_conf) and filter_line_label==False:
                            start_crop_image = time.time()
                            crop = get_crop_image(xyxy,im0, BGR=True)
                            if time_log:
                                during_crop_image = time.time() - start_crop_image
                                print('[detect.py]during_crop_image : {} ms'.format(float(during_crop_image*1000)))
                            #crop = save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True, save=False)
                            #crop = cv2.resize(crop,(isize,isize),interpolation=cv2.INTER_LINEAR)
                            #crop = crop / 255.0
                            #crop = np.expand_dims(crop, axis=0)
                            print(crop.shape)
                            #crop=tf.convert_to_tensor(crop,dtype=tf.float32)
                            #crop=tf.cast(crop/255.0,dtype=tf.float32)
                            #===========Remove tensorflow function==================
                            #crop = crop/255.0
                            #crop = crop.astype(np.float32)
                            #===========Remove tensorflow function==================
                            #cuda = True if torch.cuda.is_available() else False
                            #Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
                            #crop = Variable(crop.type(Tensor))
                            #loss = ganomaly.infer_cropimage(crop)
                            #w=r'/home/ali/Desktop/GANomaly-tf2/export_model/2022-12-05/32-nz100-ndf64-ngf64-prelu-upsample-int8/ckpt-32-nz100-ndf64-ngf64-20221205-prelu-upsample-G-int8_edgetpu.tflite'
                            w=r'ertertert'
                            #loss, gen_img = ganomaly.infer_cropimage_tflite(crop, w, interpreter, tflite=True, edgetpu=False)
                            #time.sleep(0.001)
                            start_GANomaly_detect_image = time.time()
                            loss, gen_img = ganomaly.detect_image(w, crop, cnt=cc)
                            if True:
                                during_GANomaly_detect_image = time.time() - start_GANomaly_detect_image
                                print('[detect.py]during_GANomaly_detect_image : {} ms'.format(float(during_GANomaly_detect_image*1000)))
                            cc+=1
                            loss = int(loss*100)
                            loss = float(loss/100.0)
                            
                            start_GANomaly_annotator_box = time.time()
                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                #Just handle label line c=0
                                if c==0:
                                    if c==0:
                                        line_count+=1
                                        annotator.box_label(xyxy, "line_normal "+str(loss) + " " + str(conf.numpy()), color=(255,0,0))
                                        normal_correct_detect_count+=1
                                    if c==0 and conf<ganomaly_conf:
                                        if loss>LOSS:
                                            abnormal_false_trigger_count+=1
                                            annotator.box_label(xyxy, "line_abnormal "+str(loss) + " " + str(conf.numpy()), color=(255,255,0))
                                        else:
                                            annotator.box_label(xyxy, "line_normal "+str(loss) + " " + str(conf.numpy()), color=(255,0,0))
                                            normal_correct_detect_count+=1
                                    elif c==1:
                                        noline_count+=1
                                        if loss>LOSS:
                                            annotator.box_label(xyxy, "noline_abnormal "+str(loss) + " " + str(conf.numpy()), color=(0,0,255))
                                            abnormal_correct_detect_count+=1
                                        else:
                                            abnormal_miss_noline_count+=1
                                            annotator.box_label(xyxy, "noline_normal "+str(loss) + " " + str(conf.numpy()), color=(255,0,0))
                                    elif c==3:
                                        annotator.box_label(xyxy, "frontline " + " " + str(conf.numpy()), color=(255,0,255))
                                        
                            if time_log:
                                during_GANomaly_annotator_box  = time.time() - start_GANomaly_annotator_box
                                print('[detect.py]during_GANomaly_annotator_box : {} ms'.format(float(during_GANomaly_annotator_box*1000)))
                            
                            
                            
                                
                            if loss>LOSS and conf < ganomaly_conf:
                                print('ab-normal line--ab-normal line--ab-normal line--ab-normal line {}--{}, time: {} ms'.format(loss,loss,during_GANomaly_detect_image*1000))
                            else:
                                print('normal-normal-normal-normal-normal-normal-normal {}-- {}, time: {} ms'.format(loss,loss,during_GANomaly_detect_image*1000))
                            '''
                            print('frame_count : {}'.format(frame_count))
                            print('line_count : {}'.format(line_count))
                            print('noline_count : {}'.format(noline_count))
                            print('normal_correct_detect_count : {}'.format(normal_correct_detect_count))
                            print('abnormal_correct_detect_count : {}'.format(abnormal_correct_detect_count))
                            print('abnormal_false_trigger_count : {}'.format(abnormal_false_trigger_count))
                            print('abnormal_miss_noline_count : {}'.format(abnormal_miss_noline_count))
                            '''
                            if not line_count == 0:
                                abnormal_false_trigger_rate = abnormal_false_trigger_count/line_count
                                print('abnormal_false_trigger_rate : {}'.format(abnormal_false_trigger_rate))
                            if not noline_count == 0:
                                abnormal_miss_noline_rate = abnormal_miss_noline_count/noline_count
                                print('abnormal_miss_noline_rate : {}'.format(abnormal_miss_noline_rate))
                            
                            
                            
                            
                            t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
                            print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
                            #input()
                        #show label noline others frontline  BB
                        
                        else: #Still in GANomaly
                            print("not c==0 and conf <TH")
                            c = int(cls)
                            if c==0:
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                        if time_log:
                            during_ganomaly_time = time.time() - start_time_ganomaly
                            print("[detect.py]during_ganomaly_time : {}".format(float(during_ganomaly_time)*1000))
                    #No GANomaly     
                    else: 
                        '''   
                        c = int(cls)
                        if c==0:
                            line_count+=1
                            if conf<0.80:
                                abnormal_false_trigger_count+=1
                                annotator.box_label(xyxy, "abnormal "+ str(conf.numpy()), color=(0,0,255))
                            else: 
                                annotator.box_label(xyxy, "normal "+ str(conf.numpy()), color=(255,0,0))
                                normal_correct_detect_count+=1
                        
                        print('frame_count : {}'.format(frame_count))
                        print('line_count : {}'.format(line_count))
                        print('noline_count : {}'.format(noline_count))
                        print('normal_correct_detect_count : {}'.format(normal_correct_detect_count))
                        print('abnormal_correct_detect_count : {}'.format(abnormal_correct_detect_count))
                        print('abnormal_false_trigger_count : {}'.format(abnormal_false_trigger_count))
                        print('abnormal_miss_noline_count : {}'.format(abnormal_miss_noline_count))
                        
                        if not line_count == 0:
                            abnormal_false_trigger_rate = abnormal_false_trigger_count/line_count
                            print('abnormal_false_trigger_rate : {}'.format(abnormal_false_trigger_rate))
                        if not noline_count == 0:
                            abnormal_miss_noline_rate = abnormal_miss_noline_count/noline_count
                            print('abnormal_miss_noline_rate : {}'.format(abnormal_miss_noline_rate))
                        '''
                        start_yolo_annotator = time.time()
                        #=============================yolo annotator========================================================================================
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            
                            if c==0 and filter_line_label==False:
                                if conf<ganomaly_conf:
                                    annotator.box_label(xyxy, label+" anomaly" , color=(0,0,255))
                                else:
                                    annotator.box_label(xyxy, label+" normal" , color=(255,0,0))
                            #else:
                                #annotator.box_label(xyxy, label, color=colors(c, True))
                        #======================================================================================================================
                        if time_log:
                            during_yolo_annotator = time.time() -start_yolo_annotator #start_yolo_annotator line 453
                            print("[detect.py]during_yolo_annotator : {}".format(float(during_yolo_annotator)*1000))
                        
            # Stream results
            im0 = annotator.result()
            start_view_img = time.time()
            #==================================view image======================================================================
            if view_img:
                
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            #==========================================================================================================
            if time_log:
                during_view_img = time.time() - start_view_img
                print("[detect.py]during_view_img : {}".format(float(during_view_img)*1000))
            #release semaphore 2023-01-16
            semaphore.release()
            #acquire semaphore 2023-01-16
            semaphore.acquire()
            start_save_image = time.time()
            #====================================save images======================================================================
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
                            vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                            vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 25, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.avi'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('H', '2', '6', '4'), fps, (w, h),True)
                        #vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
            
            #release semaphore 2023-01-16
            semaphore.release()
            #========================================================================================================================
            if time_log:
                during_save_img  = time.time() - start_save_image
                print('[detect.py]during_save_img : {} ms'.format(float(during_save_img*1000)))
            #if ganomaly_time_log:
        
        if time_log:
            sum_all_during = during_yolo + during_save_img + during_view_img + during_copy_img + during_filter_line + during_start_process_predictions
            print("[detect.py]sum_all_during: {}".format(float(sum_all_during*1000)))
        druring_detect_one_frame = time.time() - start_detect_one_frame_time # start_detect_one_frame_time is initialize at line 194
        print('[detect.py]============druring_detect_one_frame : {} ms=============='.format(float(druring_detect_one_frame*1000)))
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
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
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
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    #========================================================================================
    parser.add_argument('--isize', type=int, default=32, help='GANomaly input size')
    parser.add_argument('--nz', type=int, default=100, help='GANomaly latent dim')
    parser.add_argument('--nc', type=int, default=3, help='GANomaly num of channels')
    parser.add_argument('--ndf', type=int, default=64, help='GANomaly number of discriminator filters')
    parser.add_argument('--ngf', type=int, default=64, help='GANomaly number of generator filters')
    parser.add_argument('--w_adv', type=int, default=1, help='GANomaly Adversarial loss weight')
    parser.add_argument('--w_con', type=int, default=50, help='GANomaly Reconstruction loss weight')
    parser.add_argument('--w_enc', type=int, default=1, help='GANomaly Encoder loss weight')
    parser.add_argument('--extralayers', type=int, default=0, help='GANomaly extralayers for both G and D')
    parser.add_argument('--encdims', type=str, default=None, help='GANomaly Layer dimensions of the encoder and in reverse of the decoder.')
    parser.add_argument('--dataset', type=str, default=r'/home/ali/GitHub_Code/YOLO/YOLOV5-old/runs/detect/f_384_2min/crops_1cls', help='dataset dir')
    parser.add_argument('--batch_size', type=int, default=64, help='GANomaly batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='GANomaly learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='GANomaly beta1 for Adam optimizer')
    parser.add_argument('--GANOMALY', action='store_true', help='enable GANomaly')
    parser.add_argument('--ganomaly-save-img', action='store_true', help='save GANomaly images')
    parser.add_argument('--LOSS', type=float, default=4.0, help='GANomaly generator loss threshold')
    parser.add_argument('--ganomaly-conf', type=float, default=0.80, help='GANomaly conf threshold')
    parser.add_argument('--ganomaly-log', action='store_true', help='save GANomaly logs')
    parser.add_argument('--time-log', action='store_true', help='save GANomaly time logs')
    #========================================================================================
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# YOLOv5 ğŸš€ by Ultralytics, GPL-3.0 license
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

import argparse
import os
import platform
import sys
from pathlib import Path
from datetime import datetime
import torch
import time
import numpy as np
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh, xywh2xyxy)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
frame_label_list_global = []
enable_add_label_list=True
Total_one_frame_time=0
frame_count_global=1
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
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
        save_airesult=False, # save ai result videos/images
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
):
    if os.path.exists(r'C:\GitHub_Code\cuteboyqq\YOLO\YOLOV5-rasp\log.txt'):
        os.remove(r'C:\GitHub_Code\cuteboyqq\YOLO\YOLOV5-rasp\log.txt')
    global frame_label_list_global
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
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride, save_dir=save_dir)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    frame_count = 0
    for path, im, im0s, vid_cap, s in dataset:
        print("[detect.py]===========Start detect one frame.===============")
        start_detect_one_frame_time = time.time()
        frame_count+=1
        save_anomaly_img = False
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
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
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    
                    #=======================Alister add 2023-03-16===================
                    if enable_add_label_list:
                        save_log_path = os.path.join(save_dir,'log.txt')
                        save_conf_log = True
                        #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1,4)))  # normalized xywh
                        xyxy_list = (xywh2xyxy(torch.tensor(xywh).view(1, 4))).view(-1).tolist()
                        #xyxy_list = (xywh2xyxy(torch.tensor(xywh).view(1, 4)) / gn).view(-1).tolist()
                        #line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        line = (cls, *xyxy_list, conf) if save_conf_log else (cls, *xyxy_list)  # label format
                        print(line)
                        #print(('%g ' * len(line)).rstrip() % line)
                        la = ('%g ' * len(line)).rstrip() % line
                        print(la)
                        #frame_label_list_global.append([f'{txt_path}.txt',('%g ' * len(line)).rstrip() % line ])
                        #frame_str = str(frame)
                        frame_label_list_global.append(f'{txt_path}.txt {la}')
                        
                        nn = len(frame_label_list_global) if len(frame_label_list_global)<=10 else 10
                        nn_real = len(frame_label_list_global)
                        #print("n = {}".format(nn))
                        #print("n_real = {}".format(nn_real))
                        #for ii in range(nn):
                            #print(frame_label_list_global[ii])
                        #=======================Alister add 2023-03-16===================
                        #https://stackoverflow.com/questions/899103/writing-a-list-to-a-file-with-python-with-newlines
                        #print("frame_count = {}".format(frame_count))
                        if frame_count % 100 == 0:
                            with open(save_log_path, 'a') as f:
                                print("open succussful !")
                                for line in frame_label_list_global:
                                    f.write("%s\n" % line)
                            frame_label_list_global.clear()
                                #f.writelines(frame_label_list_global) #l am not sure if this code is right~~~
                    #=======================Alister add 2023-03-16===================
                    
                    
                    #Alister add 2023-02-21    save anomaly images    
                    c = int(cls)  # integer class
                    #if c==0 and conf<0.70 and filter_line_label==False: #test c==1
                    if c==0 and conf<0.70: #test c==1 
                        save_anomaly_img = True
                        now = datetime.now()
                        s_time = datetime.strftime(now,'%y-%m-%d-%H-%M-%S')
                        #anomaly_img_count+=1
                    
                    '''
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    '''
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        #annotator.box_label(xyxy, label, color=colors(c, True))
                        #if c==0 and filter_line_label==False: #noline (test)
                        if c==0: #noline (test)
                            if conf<0.70:
                                annotator.box_label(xyxy, label+" anomaly" , color=(255,0,128))
                            else:
                                annotator.box_label(xyxy, label+" normal" , color=(255,0,0))
                        elif not c==0:
                            annotator.box_label(xyxy, label, color=colors(c, True))
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
            if save_airesult:
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
                            #Alister add 2023-03-19
                            #modified_path = analysis_path(save_path)
                            file = save_path.split("\\")[-1]
                            file_name = file.split(".")[0]
                            print(file_name)
                            if file_name=="0":
                                save_path = generate_specific_path(save_dir)
                            else:
                                save_path = save_path
                            #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
                            #save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            #vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
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
        
        druring_detect_one_frame = time.time() - start_detect_one_frame_time # start_detect_one_frame_time is initialize at line 194
        print('[detect.py]============druring_detect_one_frame : {} ms=============='.format(float(druring_detect_one_frame*1000)))
        global Total_one_frame_time
        global frame_count_global
        Total_one_frame_time+=druring_detect_one_frame
        Avg_one_frame_time = Total_one_frame_time/frame_count_global
        FPS = int(1000.0/float(Avg_one_frame_time*1000))
        frame_count_global+=1
        print("=======f:{}=Total time:{} ms==========[Sequential]Avg_one_frame_time: {} ms====FPS:{}==================".format(frame_count_global,
                                                                                                                         Total_one_frame_time*1000,
                                                                                                                         Avg_one_frame_time*1000,
                                                                                                                               FPS))
        
        
        # Print time (inference-only)
        LOGGER.info(f"{frame_count} {s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
#Alister add 2023-03-19
def analysis_path(path):
    path_dir = os.path.dirname(path)
    #file = path.split("/")[-1]
    now = datetime.now()
    s_time = datetime.strftime(now,'%y-%m-%d_%H-%M-%S')
    s_time = str(s_time)
    new_file = s_time + ".avi"
    modified_path = os.path.join(path_dir,new_file)
    return modified_path

def generate_specific_path(save_dir="runs/detect"):
    #custom_dir = r"C:\GitHub_Code\cuteboyqq\YOLO\YOLOV5-rasp\runs\detect"
    #now=datetime.now()
    #s_time=datetime.strftime(now,'%y-%m-%d_%H-%M-%S')
    #s_time=str(s_time)
    new_file = "0_result.avi"
    custom_path = os.path.join(save_dir,new_file)
    return custom_path
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--save-airesult', action='store_true', help='save images/videos of ai result')
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

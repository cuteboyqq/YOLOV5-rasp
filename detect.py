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

USE_TIME=False
set_time = 0.1
set_time_3 = 0.1
im_global=None
path_global=None
im0s_global=None
s_global=None
vid_cap_global=None
pred_global=None
model_global=None

sem1 = threading.Semaphore(0)
sem2 = threading.Semaphore(0)
sem3 = threading.Semaphore(0)
sem4 = threading.Semaphore(1)
sem5 = threading.Semaphore(1)
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
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
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
    parser.add_argument('--weights', nargs='+', type=str, default= r'C:\GitHub_Code\cuteboyqq\YOLO\YOLOV5-rasp\runs\train\f192_2022-12-29-4cls\weights\best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=r'C:\factory_data\Produce_1221_720P_30FPS_SHORT.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=r'C:\GitHub_Code\cuteboyqq\YOLO\YOLOV5-rasp\data\factory_new2.yaml', help='(optional) dataset.yaml path')
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


def Get_Frame(dataset):
    
    global im_global
    global path_global
    global im0s_global
    global s_global
    global vid_cap_global
    
    #path, im, im0s, vid_cap, s = dataset
    #print(dataset)
    dataset.__iter__()
    while True:
        
        #for path, im, im0s, vid_cap, s in dataset:
            
        #sem4.acquire() #sem4=0
            #for path, im, im0s, vid_cap, s in dataset:
            #path, im, im0s, vid_cap, s = dataset
        
        data = dataset.__next__()
        path, im, im0s, vid_cap, s = data
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            #print("im is None")
            im = im[None]  # expand for batch dim
        #print("im.shape: {}".format(im.shape))
        im_global = im
        #print("im_global shape : {}".format(im_global.shape))
        path_global = path
        im0s_global = im0s
        s_global = s
        vid_cap_global = vid_cap
        
        print("1")
        #print("[Get_Frame]get im done")
        sem1.release() #sem1=1
        #print("[Get_Frame]sem1 after release: {}".format(sem1))
        #print(im_global.shape)
        #return im, path, s, im0s, vid_cap
        #return im_global
        if USE_TIME:
            time.sleep(set_time)



def model_inference(visualize,save_dir,path,augment):
    global pred_global
    global model_global
    global im_global
    while True:
        #print("[model_inference] sem1 befroe acquire: {}".format(sem1))
        sem1.acquire() #sem1=0
        #print("[model_inference] sem1 after acquire: {}".format(sem1))
        
        # Directories
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model_global(im_global, augment=augment, visualize=visualize)
        #sem5.acquire() #sem5=0
        pred_global = pred
        #sem5.release() #sem5=1
        #return pre2
        
        print("2")
        #print("[model_inference] sem2 start release: {}".format(sem2))
        sem2.release() #sem2=1
        #print("[model_inference] sem2 release done: {}".format(sem2))
        #sem4.release() #sem4=1
        if USE_TIME:
            time.sleep(set_time)

def nms(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det):
    pred_nms = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    return pred_nms

# will use this func in Run_inference function
def Process_Prediction(pred=None,
                       source = 0,
                       path='',
                       im0s='',
                       dataset = None,
                       s='',
                       save_dir='',
                       im = im_global,
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
    
    seen, windows = 0, []
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
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
    #raise NotImplemented
    # Print time (inference-only)
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
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
                
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


def PostProcess(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det,
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
    global im0s_global
    global path_global
    global im_global
    global pred_global
    while True:
        #print("[PostProcess] sem2 before acquire")
        sem2.acquire() #sem2=0
        #print("[PostProcess] sem2 after acquire")
        
        pred = nms(pred_global, conf_thres, iou_thres, classes, agnostic_nms, max_det)
        
        #sem5.acquire() #sem5=0
        
        #sem5.release() #sem5=1
        save_path, im0 = Process_Prediction(pred=pred,
                            source = source,
                            path=path_global,
                            im0s=im0s_global,
                            dataset = dataset,
                            s=s,
                            save_dir=save_dir,
                            im =im_global,
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
        print("3")
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
    view_img = True #opt.view_img
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
    
    
    model_global = model
    
    dataset, bs, source = load_dataloader(source=source,
                                    nosave=False,
                                    imgsz=imgsz,
                                    stride=stride,
                                    pt=pt
                                    )
    
    
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
    model_global.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    
    
    
    print("Thread count: {}".format(threading.active_count()))
    print("threading.enumerate() :{}".format(threading.enumerate() ))
    #for path, im, im0s, vid_cap, s in dataset:
    #path_global = None
    print("before t1")
    t1 = threading.Thread(target = Get_Frame ,args=(dataset,))
    with dt[0]:
        t1.start()    
    print("after t1.start()")
    
    #global path_global
    print("before t2")
    t2 = threading.Thread(target = model_inference,args=(visualize,save_dir,path_global,augment,))
    with dt[1]:
        t2.start()
    print("after t2.start()")
    #
    
    
    
    print("before t3")
    t3 = threading.Thread(target = PostProcess, args=(pred_global, conf_thres, iou_thres, classes, agnostic_nms, max_det,
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
                    vid_writer,) )
    
    print("Thread count: {}".format(threading.active_count()))
    with dt[2]:
        t3.start()
    print("after t3.start()")
    
    
    #t1.join()
    #t2.join()
    #t3.join()
    
    
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
        '''
        
    '''
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
    LOGGER.info(f"{s_global}{(dt[0].dt + dt[1].dt + dt[2].dt) * 1E3:.1f}ms")
        
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
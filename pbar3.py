#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:06:03 2023

@author: ali
"""

# importing libraries
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import time
#from utils.plots import Annotator, colors, save_one_box
import cv2
import os
import shutil
SET_FPS = 10
SET_W = 1280
SET_H = 720
from datetime import datetime
import numpy as np
SAVE_RAW_STREAM=True
#============Alister add 2023-03-30==========================================================================
#=======copy yolov5 utils.plots save_one_box=============================================================================================================
'''
def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.00, pad=0, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix('.jpg'))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f, quality=100, subsampling=0)  # save RGB
    return crop
'''
#=======copy yolov5 utils.plots Colors=======================================================================================================
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'
#=========================================================================================================
#copy yolov5 utils.plots Annotator
#=====2023-03-30 add Alister====================================================================================

def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)

class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        self.pil = pil or non_ascii
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_pil_font(font='Arial.Unicode.ttf' if non_ascii else font,
                                       size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
        
    def box_label_filterline(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255),enable_filter_left_line=False,enable_filter_right_line=False):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                     box[1] + 1 if outside else box[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 2, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 6, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            self.lw / 6,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)
                
    def time_label(self,frame_count=9999, txt_color=(0,0,255),w=1280.0,h=720.0,enable_frame=True):
        tf = max(self.lw - 2, 1)  # font thickness
        now = datetime.now()
        s_time = datetime.strftime(now,'%y-%m-%d %H:%M:%S')
        if enable_frame:
            s_time = str(s_time) + ' fr:' + str(frame_count)
        else:
            s_time = str(s_time)
        if self.pil:
            self.draw.text((50,320), s_time, fill=txt_color, font=self.font)
        else:
            cv2.putText(self.im,
                        s_time, (460,36),0,
                        self.lw / 5,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)
        return s_time
    
    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                     box[1] + 1 if outside else box[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 2, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 6, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            self.lw / 6,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)

    def masks(self, masks, colors, im_gpu=None, alpha=0.5):
        """Plot masks at once.
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
        """
        if self.pil:
            # convert to numpy first
            self.im = np.asarray(self.im).copy()
        if im_gpu is None:
            # Add multiple masks of shape(h,w,n) with colors list([r,g,b], [r,g,b], ...)
            if len(masks) == 0:
                return
            if isinstance(masks, torch.Tensor):
                masks = torch.as_tensor(masks, dtype=torch.uint8)
                masks = masks.permute(1, 2, 0).contiguous()
                masks = masks.cpu().numpy()
            # masks = np.ascontiguousarray(masks.transpose(1, 2, 0))
            masks = scale_image(masks.shape[:2], masks, self.im.shape)
            masks = np.asarray(masks, dtype=np.float32)
            colors = np.asarray(colors, dtype=np.float32)  # shape(n,3)
            s = masks.sum(2, keepdims=True).clip(0, 1)  # add all masks together
            masks = (masks @ colors).clip(0, 255)  # (h,w,n) @ (n,3) = (h,w,3)
            self.im[:] = masks * alpha + self.im * (1 - s * alpha)
        else:
            if len(masks) == 0:
                self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
            colors = torch.tensor(colors, device=im_gpu.device, dtype=torch.float32) / 255.0
            colors = colors[:, None, None]  # shape(n,1,1,3)
            masks = masks.unsqueeze(3)  # shape(n,h,w,1)
            masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

            inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
            mcs = (masks_color * inv_alph_masks).sum(0) * 2  # mask color summand shape(n,h,w,3)

            im_gpu = im_gpu.flip(dims=[0])  # flip channel
            im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
            im_gpu = im_gpu * inv_alph_masks[-1] + mcs
            im_mask = (im_gpu * 255).byte().cpu().numpy()
            self.im[:] = scale_image(im_gpu.shape, im_mask, self.im.shape)
        if self.pil:
            # convert im back to PIL and update draw
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top'):
        # Add text to image (PIL-only)
        if anchor == 'bottom':  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        self.draw.text(xy, text, fill=txt_color, font=self.font)

    def fromarray(self, im):
        # Update self.im from a numpy array
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)
#=========================================================================================
#============================================================================================================
class ProgressBar(QWidget):
    def parse_log_txt(self,log_dir="runs/detect/"):
        #parsing log file
        #log format is : [save_txt_path] [class] x y w h
        #generate each frame label.txt and save at file_dir
        #log_path = log_dir + "log.txt"
        log_path = os.path.join(log_dir,"log.txt")
        label_dir = os.path.join(log_dir,"labels")
        #remove old folder
        if os.path.exists(label_dir):
            shutil.rmtree(label_dir)
            
        #with open(r"C:\GitHub_Code\cuteboyqq\YOLO\YOLOV5-rasp\runs\detect\exp22\log.txt") as f:
        with open(log_path) as f:
            for line in f.readlines():
                #format  : [label file path] cls x y w h conf 
                save_txt_path = line.split(" ")[0]
                label = line.split(" ")[1:] # cls x y x y conf 23-03-27 14:10:08 fr:10
                #label = line.split(" ")[1:7] # cls x y x y conf
                label_str = ' '.join(x for x in label)
                
                #timestamp = line.split(" ")[7:] # examaple : 23-03-27 14:10:08 fr:10
                #timestamp_str = ' '.join(x for x in timestamp)
                
                
                
                print(line.split(" ")[0])
                
                
                file,file_name,file_dir = self.Analysis_path(save_txt_path)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                # generate label.txt of each frame (format is cls x y x y conf)
                with open(save_txt_path,'a') as l_f:
                    l_f.write(label_str)
        
        return file_dir
    def __init__(self,video_path, 
                skip_frame, 
                yolo_txt_dir, 
                class_path, 
                img_size,
                yolo_infer_txt,
                log_dir,
                save_airesult):
        super().__init__()
        self.yolo_txt_dir = self.parse_log_txt(log_dir)
        self.video_path=video_path
        self.skip_frame=skip_frame 
        #self.yolo_txt_dir=yolo_txt_dir 
        self.class_path=class_path 
        self.img_size=img_size
        self.yolo_infer_txt = yolo_infer_txt
        self.log_dir=str(log_dir),
        self.save_airesult=save_airesult
        # calling initUI method
        self.initUI()
	# method for creating widgets
    def initUI(self):
        
		
        #self.video_extract_frame()
        
        # create label widget with message
        #self.message = "let start parsing !"
        #self.label = QLabel(self.message, self)
        
        # create central widget
        #central_widget = QWidget(self)
        
        # create vertical layout for central widget
        #layout = QVBoxLayout(central_widget)
        
        # create text edit widget for output
        self.output = QTextEdit(self)
        self.output.setGeometry(15, 10, 300, 200)
        #layout.addWidget(self.output)
        
        # creating progress bar
        self.pbar = QProgressBar(self)
        
		# setting its geometry
        self.pbar.setGeometry(15, 250, 200, 25)

		# creating push button
        self.btn = QPushButton('Start', self)

		# changing its position
        self.btn.move(15, 220)

		# adding action to push button
        self.btn.clicked.connect(self.video_extract_frame)
        
        
		# setting window geometry
        self.setGeometry(300, 300, 350, 300)

		# setting window action
        self.setWindowTitle("Parsing log.txt")

		# showing all the widgets
        self.show()

	# when button is pressed this method is being called
    '''
	def doAction(self):

		# setting for loop to set value of progress bar
		for i in range(101):

			# slowing down the loop
			time.sleep(0.05)

			# setting value to progress bar
			self.pbar.setValue(i)
    '''
    def Analysis_path(self,path):
        file = path.split(os.sep)[-1]
        file_name = file.split(".")[0]
        file_dir = os.path.dirname(path)
        print("file = ",file)
        print("file_name = ",file_name)
        print("file_dir = ",file_dir)
        return file,file_name,file_dir
    
    
    
    
    def video_extract_frame(self):
        
        #Alister add 2023-02-28
        if SAVE_RAW_STREAM:
            now = datetime.now()
            s_time = datetime.strftime(now,'%y-%m-%d_%H-%M-%S')
            s_time = str(s_time)
            #save_path = log_dir + "0_result_offline"+ ".avi"
            print("self.log_dir : {}".format(self.log_dir))
            save_path = os.path.join(self.log_dir[0],"0_result_offline.avi")
            w,h=SET_W,SET_H
            fps=SET_FPS
            vw = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
            #save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
        
        vidcap = cv2.VideoCapture(self.video_path)
        print("video_path:{}".format(self.video_path))
        #====================================================
        #self.output.setText("Start calculate number of frames...")
        #self.output.show()
        frame_count = 0
        success,image = vidcap.read()
        while True:
            if success:
                frame_count += 1
                
            else:
                
                break
            success,image = vidcap.read()
            if success:
                image_od_result = image.copy()
            #print('Read a new frame: ', success)
        vidcap.release()
        #self.output.setText("Start calculate number of complete...")
        #====================================================
        vidcap = cv2.VideoCapture(self.video_path)
        #self.frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        #self.frame_count = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
        print("frame_count:{}".format(frame_count))
        #input()
        #self.pbar.setRange(0,frame_count+1)    # 進度條範圍
        self.pbar.setMaximum(frame_count)
        success,image = vidcap.read()
        image_od_result = image.copy()
        count = 0
        anomaly_count = 0
        anomaly_frame = False
        file,filename,file_dir = self.Analysis_path(self.video_path)
        print(file," ",filename," ",file_dir)
        save_folder_name =  filename + "_imgs"
        save_dir = os.path.join(file_dir,save_folder_name)
        '''
        save_ori_folder_name =  filename + "_ori_imgs"
        save_ori_dir = ps.path.join(file_dir,save_ori_folder_name)
        '''
        #remove old 0_img folder
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        #create 0_img folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        '''
        #remove old 0_ori_img folder
        if os.path.exists(save_ori_dir):
            shutil.rmtree(save_ori_dir)
        #create 0_ori_img folder
        if not os.path.exists(save_ori_dir):
            os.makedirs(save_ori_dir)
        ''' 
        
        #Copy class.txt to save_dir , no need now
        #shutil.copy(class_path,save_dir)
        
        while True:
            if success:
                anomaly_frame = False
                namess = "2023-03-17"
                #drawing on image_od_result
                annotator_for_odresult = Annotator(image_od_result, line_width=2, example=str(namess))
                annotator = Annotator(image, line_width=2, example=str(namess))
                if count%self.skip_frame==0:
                    
                    #====extract video frame====
                    filename_ = filename + "_" + str(count) + ".jpg"
                    img_path = os.path.join(save_dir,filename_)
                    #image = cv2.resize(image,(img_size, int(img_size*9/16) ))
                    #image = cv2.resize(image,(img_size,img_size))
                    
                    #=====Start parsing label.txt, try to get the xyxy and cls informations======================
                    filename_txt_ = filename + "_" + str(count) + ".txt"
                    txt_path = os.path.join(self.yolo_txt_dir,filename_txt_)
                    if os.path.exists(txt_path):
                        with open(txt_path,'r') as f:
                            for line in f.readlines():
                                print(line)
                                #Line format is [class] x y w h conf
                                #Try to get xywh
                                xyxy = line.split(" ")[1:5]
                                label = line.split(" ")[0]
                                conf = line.split(" ")[5]
                                timestamp = line.split(" ")[6:]
                                conf_f = float(conf)
                                xyxy_str = ' '.join(x for x in xyxy)
                                label_str = ' '.join(x for x in label)
                                conf_str = ' '.join(x for x in conf)
                                timestamp_str = ' '.join(x for x in timestamp)
                                print("label_str = {}".format(label_str))
                                print("xyxy_str = {}".format(xyxy_str))
                                print("conf_str = {}".format(conf_str))
                                self.output.setText("Anomaly Frame Log : \n file_txt = {}\n label_str = {} \n xyxy_str = {}\n conf_str = {}\n timestamp_str = {}\n anomaly_count:{}\n save frame:{}".format(txt_path,
                                                                                                                                                                                                            label_str,
                                                                                                                                                                                                            xyxy_str,
                                                                                                                                                                                                            conf_str,
                                                                                                                                                                                                      timestamp_str,
                                                                                                                                                                                                            anomaly_count,
                                                                                                                                                                                                            count))
                                #self.output.setText("file_txt = {}\n label_str = {} \n xyxy_str = {}\n conf_str = {}\n timestamp_str = {}\n save frame {}".format(txt_path,label_str,xyxy_str,conf_str,timestamp_str,count))
                                #Not implemented
                                #save_airesult=True
                                hide_labels = False
                                hide_conf = False
                                names = ["line","noline","others","frontline"]
                                #c = int(label_str)  # integer class
                                #if c==0 and conf_f<0.70:
                                    #anomaly_count+=1
                                
                                if self.save_airesult:
                                    #=====End parsing label.txt, try to get the xyxy and cls informations======================
                                    c = int(label_str)  # integer class
                                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf_f:.2f}')
                                    #annotator.box_label(xyxy, label, color=colors(c, True))
                                    #if c==0 and filter_line_label==False: #noline (test)
                                    if c==0: #noline (test)
                                        if conf_f<0.70:
                                            annotator.box_label(xyxy, label+" anomaly" , color=(255,0,128))
                                            
                                            self.output.show()
                                        else:
                                            annotator.box_label(xyxy, label+" normal" , color=(255,0,0))
                                    elif not c==0:
                                        annotator.box_label(xyxy, label, color=colors(c, True))
                                        
                                #For OD result frame, get BB info and draw to frame 2023-03-24
                                #=====End parsing label.txt, try to get the xyxy and cls informations======================
                                #=================Saving the complete OD result frame "image_od_result" into Stream=====================================
                                c = int(label_str)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf_f:.2f}')
                                #annotator.box_label(xyxy, label, color=colors(c, True))
                                #if c==0 and filter_line_label==False: #noline (test)
                                if c==0: #noline (test)
                                    if conf_f<0.70:
                                        anomaly_frame=True
                                        annotator_for_odresult.box_label(xyxy, label+" anomaly" , color=(255,0,128))
                                    else:
                                        annotator_for_odresult.box_label(xyxy, label+" normal" , color=(255,0,0))
                                elif not c==0:
                                    annotator_for_odresult.box_label(xyxy, label, color=colors(c, True))
                        #===========================================================
                    
                #==============================================
                if SAVE_RAW_STREAM:
                    #names="test_2023_03_03"
                    #image = np.ascontiguousarray(image)  # contiguous
                    #annotator = Annotator(im0[0], line_width=3, example=str(names))
                    #if not SET_H is 720:
                        #image = image[..., ::-1]
                    #annotator.time_label(frame_count=self.count,txt_color=(0,255,128))
                    vw.write(image_od_result)
                #==============================================
                    #save the raw images without Drawing OD Bounding Box
                    cv2.imwrite(img_path,image)
                    print("save image complete",img_path)
                    
                    #cv2.imwrite(img_path,image)
                    if self.yolo_infer_txt:
                        #=====Copy .txt file=======
                        filename_txt_ = filename + "_" + str(count) + ".txt"
                        txt_path = os.path.join(self.yolo_txt_dir,filename_txt_)
                        if os.path.exists(txt_path):
                            shutil.copy(txt_path,save_dir)
                        
                    #cv2.imwrite("/home/ali/datasets-old/TL4/frame%d.jpg" % count, image)     # save frame as JPEG file    
                    print('save frame ',count)
                    # setting value to progress bar
                    
                    self.pbar.setValue(count+1)
                    #print("progress: {} %".format(count))
            else:
                print('Video capture failed, break')
                break
            success,image = vidcap.read()
            if success:
                image_od_result = image.copy()
            #print('Read a new frame: ', success)
            if anomaly_frame:
                anomaly_count+=1
            count += 1
            #self.message = str(count)
            
def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('-videopath','--video-path',help="input video path",default="/home/ali/Argos_Project/Factory_Project/Argos_Raw_Stream/2023-03-04/Big_Screen/Light_ON_OFF/20230304_BIGSCREEN_LIGHT_OFF_AND_ON.mp4")
    #parser.add_argument('-videopath','--video-path',help="input video path",default="/home/ali/Argos_Project/Factory_Project/Screen_record_Argos_Stream/20230314_203715.mp4")
    #parser.add_argument('-videopath','--video-path',help="input video path",default="/home/ali/factory_video/Argos_Record/2023-03-15/SmallScreen_Record/filterline_version2/YoloV4-f320-4cls/20230315-215557-H264-00.mp4")
    #parser.add_argument('-videopath','--video-path',help="input video path",default="/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/2023-03-17/0.avi")
    parser.add_argument('-videopath','--video-path',help="input video path",default=r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp6/0.avi")
    parser.add_argument('-logdir','--log-dir',help="input log directory",default=r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp6")
    parser.add_argument('-skipf','--skip-f',type=int,help="number of skp frame",default=1)
    parser.add_argument('-imgsize','--img-size',type=int,help="size of images",default=640)
    parser.add_argument('-yoloinfer','--yolo-infer',action='store_true',help="have yolo infer txt")
    #parser.add_argument('-yolotxt','--yolo-txt',help="yolo infer label txt dir",default="/home/ali/factory_video/2023-03-04/labels")
    parser.add_argument('-yolotxt','--yolo-txt',help="yolo infer label txt dir",default="/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/2023-03-1712/labels")
    parser.add_argument('-classtxt','--class-txt',help="class.txt path",default="/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/classes.txt")
    parser.add_argument('-saveairesult','--save-airesult',action='store_true',help="Save object detection result")
    
    return parser.parse_args()
# main method
if __name__ == '__main__':
	
    
    args=get_args()
    video_path = args.video_path
    skip_frame = args.skip_f
    yolo_txt_dir = args.yolo_txt
    class_path = args.class_txt
    yolo_infer = args.yolo_infer
    img_size = args.img_size
    log_dir = args.log_dir
    save_siresult = args.save_airesult
    print("video_path =",video_path)
    print("skip_frame = ",skip_frame)
    print("yolo_txt_dir = ",yolo_txt_dir)
    print("class_path = ",class_path)
    print("yolo_infer = ",yolo_infer)
    print("img_size = ",img_size)
    print("log_dir = ",log_dir)
    #log_dir=r"C:\GitHub_Code\cuteboyqq\YOLO\YOLOV5-rasp\runs\detect\exp22"
    #yolo_txt_dir = parse_log_txt(log_dir)
	
	# create pyqt5 app
    App = QApplication(sys.argv)

	# create the instance of our Window
    window = ProgressBar(video_path, 
                        skip_frame, 
                        yolo_txt_dir, 
                        class_path, 
                        img_size,
                        True,
                        log_dir,
                        save_siresult)

	# start the app
    sys.exit(App.exec())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:50:32 2023

@author: ali
"""
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer
import os
import subprocess
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QWidget, QVBoxLayout, QCheckBox, QMessageBox, QLabel, QPushButton, QFileDialog
from PyQt5.QtMultimediaWidgets import QVideoWidget

shift_right = 10
FONT_SIZE = 12
#FONT="Arial" # Times New Roman
FONT= "Times New Roman"
SHOW_LABELS=False
class ImageComboBox(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anomaly Image & Clips")
        self.setGeometry(20, 20, 500, 250)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        #self.layout = QVBoxLayout()
        
        
        #==================directory_label========================================================
        if SHOW_LABELS:
            self.directory_label = QLabel()
            self.directory_label.setText("Result Directories:")
            self.directory_label.setFont(QFont(FONT, FONT_SIZE))  # set the font size
            #self.directory_label.setGeometry(10+shift_right, 0, 200, 60)
            self.layout.addWidget(self.directory_label)
        
        #==================directory_combo_box========================================================
        self.directory_combo_box = QComboBox()
        self.directory_combo_box.setFont(QFont(FONT, FONT_SIZE))  # set the font size
        #self.directory_combo_box.setGeometry(150+shift_right, 0, 250, 60)
        self.directory_combo_box.addItems(self.get_subfolders(r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/"))  # Replace with the root directory path

        self.directory_combo_box.currentIndexChanged.connect(self.selected_directory_changed)
        self.directory_combo_box.currentIndexChanged.connect(self.selected_directory_changed_part2)
        
        self.layout.addWidget(self.directory_combo_box)
       
        #==================image_label========================================================
        if SHOW_LABELS:
            self.image_label = QLabel()
            self.image_label.setText("Anoamly Image Paths:")
            self.image_label.setFont(QFont(FONT, FONT_SIZE))  # set the font size
            #self.directory_label.setGeometry(10+shift_right, 0, 200, 60)
            self.layout.addWidget(self.image_label)
        
        #=====================image_combo_box=========================================================================
        self.image_combo_box = QComboBox()
        self.image_combo_box.setFont(QFont(FONT, FONT_SIZE))  # set the font size
        self.image_combo_box.currentIndexChanged.connect(self.selected_image_changed)

        self.layout.addWidget(self.image_combo_box)
        
        #==========================QPushButton===============================================================
        self.btn_select_dir = QPushButton("Select Directory Of Saving Short Anomaly Clips")
        self.btn_select_dir.setFont(QFont(FONT, FONT_SIZE))  # set the font size
        #self.btn_select_dir.setGeometry(50, 50, 200, 30)
        self.btn_select_dir.clicked.connect(self.select_dir)
        self.layout.addWidget(self.btn_select_dir)
        self.btn_select_dir.selected_dir = ""
        #==================smallclip_label========================================================
        if SHOW_LABELS:
            self.smallclip_label = QLabel()
            self.smallclip_label.setText("Anomaly Small Clips:")
            self.smallclip_label.setFont(QFont(FONT, FONT_SIZE))  # set the font size
            #self.directory_label.setGeometry(10+shift_right, 0, 200, 60)
            self.layout.addWidget(self.smallclip_label)
        
        #====================smallclip_combo_box=========================================
        self.smallclip_combo_box = QComboBox()
        self.smallclip_combo_box.setFont(QFont(FONT, FONT_SIZE))  # set the font size
        self.smallclip_combo_box.currentIndexChanged.connect(self.play_selected_video)
        self.layout.addWidget(self.smallclip_combo_box)
        
        # create the QMediaPlayer and QVideoWidget
        self.mediaPlayer = QMediaPlayer(self)
        self.videoWidget = QVideoWidget(self)
        self.player_process = None
        #=============================================================
        
        

        self.check_box = QCheckBox('Generate Short Clips', self)
        self.check_box.setFont(QFont(FONT, FONT_SIZE))  # set the font size
        self.check_box.stateChanged.connect(self.generate_short_clips_changed)

        self.layout.addWidget(self.check_box)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        #======================================================================
        # Set up layout
        #layout = QVBoxLayout(self)
      
    
        #self.setLayout(layout)
    def select_dir(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Directory", options=options)
        self.btn_select_dir.selected_dir = selected_dir
    
    def get_subfolders(self, directory):
        subfolders = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
        return subfolders

    def get_image_files(self, directory):
        image_files = []
        for file in os.listdir(directory):
            if file.endswith(".jpg") or file.endswith(".png"):
                #image_files.append(os.path.join(directory, file))
                frame_count = file.split(".")[0]
                image_files.append(int(frame_count))
        
        image_files.sort()
        for i in range(len(image_files)):
            frame_count = image_files[i]
            frame_count = str(frame_count)
            file = frame_count + ".jpg"
            image_files[i] = os.path.join(directory, file)
        return image_files
    def selected_directory_changed_part2(self, index):
        self.selected_directory = self.directory_combo_box.currentText()
        if not os.path.exists(os.path.join(self.selected_directory,"anomaly_img_offline")):
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Question)
            message_box.setText("Do you want to collect anomaly images?")
            message_box.setWindowTitle("Get Anomaly Images")
            message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            result = message_box.exec_()
            if result == QMessageBox.Yes:
                command = ['python', 'get_anomaly_image_offline_ver2.py', '--root-datadir', self.selected_directory]
                subprocess.run(command)
    
    def selected_directory_changed(self, index):
        self.selected_directory = self.directory_combo_box.currentText()
        self.image_combo_box.clear()
        self.image_combo_box.addItems(self.get_image_files(os.path.join(self.selected_directory, 'anomaly_img_offline')))
        #2023-03-22 modified use  custom directory
        #self.anomaly_clips_offline = os.path.join(self.selected_directory, 'anomaly_clips')
        #self.anomaly_clips_offline = self.selected_directory
        self.anomaly_clips_offline = self.btn_select_dir.selected_dir
        # populate the QComboBox with a list of video files found in the directory
        self.smallclip_combo_box.clear()
        for filename in os.listdir(self.anomaly_clips_offline):
            if filename.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                self.smallclip_combo_box.addItem(filename)
    #=====================================================================
    def selected_smallclip_changed(self, index):
        selected_smallclip = self.smallclip_combo_box.currentText()
    
    
    '''
    def on_video_selected(self, index):
        # get the selected video filename
        filename = self.smallclip_combo_box.currentText()

        # create a file path to the selected video
        filepath = os.path.join(self.selected_directory, "anomaly_clips", filename)

        # set the media content of the QMediaPlayer to the selected video
        media = QMediaContent(QUrl.fromLocalFile(filepath))
        self.mediaPlayer.setMedia(media)

        # set the QVideoWidget as the video output of the QMediaPlayer
        self.mediaPlayer.setVideoOutput(self.videoWidget)

        # start playing the video
        self.mediaPlayer.play()
    '''
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.player_process:
            self.player_process.kill()
        
    def play_selected_video(self,index):
        # get the selected video filename
        #filename = self.smallclip_combo_box.currentText()
        if self.player_process:
            self.player_process.kill()
        #2023-03-22 modified, use custom directory
        video_path = os.path.join(self.btn_select_dir.selected_dir,self.smallclip_combo_box.currentText())
        #video_path = os.path.join(self.selected_directory,"anomaly_clips",self.smallclip_combo_box.currentText())
        #args = ["vlc", "--fullscreen", video_path]
        #self.player_process = subprocess.Popen(args)
        
        args = ["/usr/bin/vlc", video_path]
        self.player_process = subprocess.Popen(args)
    #=====================================================================
    def selected_image_changed(self, index):
        selected_image = self.image_combo_box.currentText()
        pixmap = QPixmap(selected_image)
        self.image_label.setPixmap(pixmap)

        if self.check_box.isChecked():
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Question)
            message_box.setText("Do you want to generate short clips?")
            message_box.setWindowTitle("Generate Short Clips")
            message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            result = message_box.exec_()
            if result == QMessageBox.Yes:
                if not self.btn_select_dir.selected_dir=="":
                    command = ['python', 'gen_shortclips_ver2.py', '--anomaly-img', selected_image, '--root-datadir', self.selected_directory, '--save-anoclipdir', self.btn_select_dir.selected_dir]
                else:
                    command = ['python', 'gen_shortclips_ver2.py', '--anomaly-img', selected_image, '--root-datadir', self.selected_directory, '--save-anoclipdir', self.anomaly_clips_offline]
                subprocess.run(command)
                # populate the QComboBox with a list of video files found in the directory
                self.smallclip_combo_box.clear()
                for filename in os.listdir(self.btn_select_dir.selected_dir): #self.anomaly_clips_offline
                    if filename.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                        self.smallclip_combo_box.addItem(filename)
                    
              
    def generate_short_clips_changed(self, state):
        pass

if __name__ == '__main__':
    app = QApplication([])
    window = ImageComboBox()
    window.show()
    app.exec_()


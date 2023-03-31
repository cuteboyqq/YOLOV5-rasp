#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:50:32 2023

@author: ali
"""
import sys
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer
import os
import subprocess
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QProcess, QIODevice, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QWidget, QVBoxLayout, QCheckBox, QMessageBox, QLabel, QPushButton, QFileDialog, QProgressBar, QHBoxLayout
from PyQt5.QtMultimediaWidgets import QVideoWidget
import shutil
shift_right = 10
FONT_SIZE = 14
#FONT="Arial" # Times New Roman
FONT= "Times New Roman"
SHOW_LABELS=True
DATA_DIR=r"/home/ali/Desktop/YOLOV5-rasp/runs/detect"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
FOLDER_PATH=DATA_DIR
class ImageComboBox(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GUI of View Anomaly Image & Clips")
        self.setGeometry(20, 20, 500, 250)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        #self.layout = QVBoxLayout()
        #===========QLabel=====================================
        self.source_label = QLabel(self)
        self.source_label.setText("Step 1: AI Result:")
        self.source_label.setFont(QFont("Times New Roman", FONT_SIZE))  # set the font size
        self.source_label.setGeometry(25, 250, 350, 30)
        self.layout.addWidget(self.source_label)
        
        #=================2023-03-28======QCheckBox==================================================================
        self.disable_checkbox = QCheckBox('Disable', self)
        self.disable_checkbox.setFont(QFont("Times New Roman", FONT_SIZE))  # set the font size
        self.enable_checkbox = QCheckBox('Enable', self)
        self.enable_checkbox.setFont(QFont("Times New Roman", FONT_SIZE))  # set the font size
        self.enable_checkbox.setChecked(True)
        self.enable_checkbox.setGeometry(50, 50, 30, 30)
        
        #layout = QHBoxLayout()
        #layout.addWidget(self.enable_checkbox)
        #layout.addWidget(self.disable_checkbox)
        #self.setLayout(layout)
        self.layout.addWidget(self.enable_checkbox)
        self.layout.addWidget(self.disable_checkbox)
        # Connect the stateChanged signals of the two checkboxes to the appropriate slots
        self.enable_checkbox.stateChanged.connect(self.on_enable_checkbox_changed)
        self.disable_checkbox.stateChanged.connect(self.on_disable_checkbox_changed)
        
        self.enable_checkbox.stateChanged.connect(self.generate_short_clips_changed)
        
        #===========QLabel=====================================
        self.source_label = QLabel(self)
        self.source_label.setText("Step 2: Select Directory Of Saving Clips")
        self.source_label.setFont(QFont("Times New Roman", FONT_SIZE))  # set the font size
        self.source_label.setGeometry(25, 250, 350, 30)
        self.layout.addWidget(self.source_label)
        #==========================QPushButton===============================================================
        self.btn_select_dir = QPushButton("Select Directory Button")
        self.btn_select_dir.setFont(QFont(FONT, FONT_SIZE))  # set the font size
        #self.btn_select_dir.setGeometry(50, 50, 200, 30)
        self.btn_select_dir.clicked.connect(self.select_dir)
        self.layout.addWidget(self.btn_select_dir)
        self.btn_select_dir.selected_dir = ""
        #==================QCheckBox====================================================
        '''
        self.check_box2 = QCheckBox('Step 1: Enable/Disable AI Result',self)
        self.check_box2.setFont(QFont(FONT, FONT_SIZE))  # set the font size
        self.check_box2.stateChanged.connect(self.generate_short_clips_changed)
        self.layout.addWidget(self.check_box2)
        #self.image_label = QLabel()
        #self.layout.addWidget(self.image_label)
        '''
        #==============QProgressBar========================================================
        '''
        self.progress_bar = QProgressBar(self)
        #self.progress_bar.setGeometry(30, 40, 200, 25)

        #self.setCentralWidget(self.progress_bar)
 
        self.process = QProcess(self)
 
        # Connect signals
        self.process.started.connect(self.on_started)
        self.process.readyReadStandardError.connect(self.on_ready_read_standard_error)
        self.process.finished.connect(self.on_finished)
        self.layout.addWidget(self.progress_bar)
        '''
        #===========QLabel=====================================
        self.source_label = QLabel(self)
        self.source_label.setText("Step 3: Select which Log.txt to parsing (Get Anomaly Events):")
        self.source_label.setFont(QFont("Times New Roman", FONT_SIZE))  # set the font size
        self.source_label.setGeometry(25, 250, 350, 30)
        self.layout.addWidget(self.source_label)
        
        # Create a QComboBox widget
        self.comboBox = QComboBox(self)
        self.comboBox.setFont(QFont("Times New Roman", FONT_SIZE))  # set the font size
        self.comboBox.setGeometry(350, 250, 300, 30)
        self.layout.addWidget(self.comboBox)
        
        
        # Populate the QComboBox with folder names
        folder_path = FOLDER_PATH
        folder_names = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        self.comboBox.addItems(folder_names)

        # Connect the activated signal of the QComboBox to a slot
        self.comboBox.activated[str].connect(self.onComboBoxActivated)
        #================================================
        
        
        '''
        #==================directory_label========================================================
        if SHOW_LABELS:
            self.directory_label = QLabel()
            self.directory_label.setText("Step 3: Select OD Result Directories:")
            self.directory_label.setFont(QFont(FONT, FONT_SIZE))  # set the font size
            #self.directory_label.setGeometry(10+shift_right, 0, 200, 60)
            self.layout.addWidget(self.directory_label)
        
        #==================directory_combo_box========================================================
        self.directory_combo_box = QComboBox()
        self.directory_combo_box.setFont(QFont(FONT, FONT_SIZE))  # set the font size
        #self.directory_combo_box.setGeometry(150+shift_right, 0, 250, 60)
        self.directory_combo_box.addItems(self.get_subfolders(DATA_DIR))  # Replace with the root directory path
        self.directory_combo_box.currentIndexChanged.connect(self.selected_directory_changed_part2)
        self.directory_combo_box.currentIndexChanged.connect(self.selected_directory_changed)
        
        
        self.layout.addWidget(self.directory_combo_box)
       '''
        #==================image_label========================================================
        if SHOW_LABELS:
            self.image_label = QLabel()
            self.image_label.setText("Step 4: Select/View Anoamly Event:")
            self.image_label.setFont(QFont(FONT, FONT_SIZE))  # set the font size
            #self.directory_label.setGeometry(10+shift_right, 0, 200, 60)
            self.layout.addWidget(self.image_label)
            
            
        #=====================image_combo_box=========================================================================
        self.image_combo_box = QComboBox()
        self.image_combo_box.setFont(QFont(FONT, FONT_SIZE))  # set the font size
        self.image_combo_box.currentIndexChanged.connect(self.selected_image_changed)
        
        self.selected_directory = self.comboBox.currentText()
        self.image_combo_box.clear()
        #if not os.path.exists(os.path.join(self.selected_directory, 'anomaly_img_offline')):
            #os.makedirs(os.path.join(self.selected_directory, 'anomaly_img_offline'))
            
        #self.image_combo_box.addItems(self.get_image_files(os.path.join(FOLDER_PATH,self.selected_directory, 'anomaly_img_offline')))
        #self.image_combo_box.addItems(self.get_image_files(os.path.join(FOLDER_PATH,self.selected_directory, 'anomaly_img_offline')))
        
        
        self.layout.addWidget(self.image_combo_box)
        
        # Populate the QComboBox with folder names
        #folder_path = FOLDER_PATH
        #folder_names = [f for f in os.listdir(os.path.join(folder_path,)) if os.path.isfile(os.path.join(folder_path, f))]
        #self.image_combo_box.addItems(folder_names)
        #===========QLabel=====================================
        self.source_label = QLabel(self)
        self.source_label.setText("Step 5: Short Anomaly Clips")
        self.source_label.setFont(QFont("Times New Roman", FONT_SIZE))  # set the font size
        self.source_label.setGeometry(25, 250, 350, 30)
        self.layout.addWidget(self.source_label)
        
        #=================QCheckBox============================================
    
        self.check_box = QCheckBox('Enable Generate', self)
        self.check_box.setFont(QFont(FONT, FONT_SIZE))  # set the font size
        self.check_box.stateChanged.connect(self.generate_short_clips_changed)

        self.layout.addWidget(self.check_box)

        
        #==================smallclip_label========================================================
        if SHOW_LABELS:
            self.smallclip_label = QLabel()
            self.smallclip_label.setText("Step 6: Select Anomaly Small Clips and Show It:")
            self.smallclip_label.setFont(QFont(FONT, FONT_SIZE))  # set the font size
            #self.directory_label.setGeometry(10+shift_right, 0, 200, 60)
            self.layout.addWidget(self.smallclip_label)
        
        #====================smallclip_combo_box=========================================
        self.smallclip_combo_box = QComboBox()
        self.smallclip_combo_box.setFont(QFont(FONT, FONT_SIZE))  # set the font size
        self.smallclip_combo_box.currentIndexChanged.connect(self.play_selected_video)
        self.layout.addWidget(self.smallclip_combo_box)
        
        
        #==========================QPushButton for clean all detect folder datasets===============================================================
        # create label
        self.label = QLabel("Click the button to delete the folder", self)
        #self.label.move(50, 50)
        self.layout.addWidget(self.label)
        
        self.btn_delete = QPushButton("Delete All Result Folder")
        self.btn_delete.setFont(QFont(FONT, FONT_SIZE))  # set the font size
        #self.btn_select_dir.setGeometry(50, 50, 200, 30)
        self.btn_delete.clicked.connect(self.show_dialog)
        self.layout.addWidget(self.btn_delete)
        #self.btn_delete.selected_dir = ""
        
        # create the QMediaPlayer and QVideoWidget
        self.mediaPlayer = QMediaPlayer(self)
        self.videoWidget = QVideoWidget(self)
        self.player_process = None
        
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        # Set up layout
        #layout = QVBoxLayout(self)
        self.load_file_list = []
        
    
        #self.setLayout(layout)
    #================Alister 2023-03-28=====================================================
    def on_enable_checkbox_changed(self, state):
        if state == 2: # Checked
            self.disable_checkbox.setChecked(False)
            self.enable_checkbox.setChecked(True)
        elif state == 0: # Checked
            self.disable_checkbox.setChecked(True)
            self.enable_checkbox.setChecked(False)
    def on_disable_checkbox_changed(self, state):
        if state == 2: # Checked
            self.enable_checkbox.setChecked(False)
            self.disable_checkbox.setChecked(True)
        elif state == 0: # Checked
            self.enable_checkbox.setChecked(True)
            self.disable_checkbox.setChecked(False)
    #==============Alister 2023-03-27=======================================================
    def run_command(self):
        self.process.start("your_command_here")

    @pyqtSlot()
    def on_started(self):
        self.progress_bar.setValue(0)

    @pyqtSlot()
    def on_finished(self):
        self.progress_bar.setValue(100)

    @pyqtSlot()
    def on_ready_read_standard_error(self):
        data = self.process.readAllStandardError()
        # Parse data to get progress percentage and update progress bar
        percentage = self.parse_progress_percentage(data)
        self.progress_bar.setValue(percentage)
    def parse_progress_percentage(self, data):
        # Replace this method with your own logic to parse progress percentage
        # from the data received from the subprocess
        # In this example, we assume that the percentage is in the format "progress: XX%"
        data_str = str(data, "utf-8").strip()
        if data_str.startswith("progress: "):
            percentage_str = data_str.split("progress: ")[-1].split("%")[0]
            percentage = int(percentage_str)
            return percentage
        else:
            return None
    #===================Alister add 2023-03-23==============================================
    def onComboBoxActivated(self, folder_name):
        # Get the full path of the selected folder
        folder_path = os.path.join(FOLDER_PATH, folder_name)
        video_path = os.path.join(FOLDER_PATH,folder_name,"0.avi")
        # Set the folder path as a command parameter
        sys.argv = ['log_parser_ver2.py', '--log-dir', folder_path, '--video-path', video_path]
        
        

        # Execute the command using subprocess
        #command = ["python", "log_parser_ver2.py", "--log-dir", folder_path, '--video-path', video_path]
        
        if not self.btn_select_dir.selected_dir:
           QMessageBox.warning(self, "Warning", "Please Select a save clip directory !!.")
           return
        
        if not os.path.exists(os.path.join(folder_path,"0.avi")) or not os.path.exists(os.path.join(folder_path,"log.txt")):
            QMessageBox.warning(self, "Error", "Need log.txt and 0.avi")
        else:
            if os.path.exists(os.path.join(folder_path,"0_result_offline.avi")):
                reply = QMessageBox.information(self, 'Integration', 'Already Have Parsed log.txt , do you want to re-generate it?',
                QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
            else:
                reply = QMessageBox.information(self, 'Integration', 'Start to generate ai result video',QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
            if reply == QMessageBox.Ok:
                if self.enable_checkbox.isChecked():
                    # Execute the command using subprocess
                    #command = ["python", "log_parser_ver2.py", "--log-dir", folder_path, "--video-path",video_path, "--save-airesult"]
                    command = ["python", "pbar3.py", "--log-dir", folder_path, "--video-path",video_path, "--save-airesult"]
                else:
                    # Execute the command using subprocess
                    command = ["python", "pbar3.py", "--log-dir", folder_path, "--video-path",video_path]
                    #command = ["python", "log_parser_ver2.py", "--log-dir", folder_path, "--video-path",video_path]
                    #subprocess.run(command)
                try:
                    process = subprocess.run(command, check=True)
                    #self.process.start(command)
                    #process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    #timer = QTimer(self)
                    #timer.timeout.connect(lambda: self.checkProcessStatus(process))
                    #timer.start(1000)  # Check the status every 1000 milliseconds
                    # If the command finishes successfully, show a message box
                    QMessageBox.information(self, "Success", "The process is done!")
                except subprocess.CalledProcessError as e:
                    QMessageBox.warning(self, "Error", f"Error executing command: {e}")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Unknown error: {e}")
            else:
                print('Close clicked.')
    #=============================================================================================    
        #========Alister add 2023-03-27======================
        self.selected_directory = self.comboBox.currentText()
        '''
        if not os.path.exists(os.path.join(FOLDER_PATH,self.selected_directory,"anomaly_img_offline")):
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Question)
            message_box.setText("Do you want to collect anomaly images?")
            message_box.setWindowTitle("Get Anomaly Images")
            message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            result = message_box.exec_()
            if result == QMessageBox.Yes:
                command = ['python', 'get_anomaly_image_offline_ver2.py', '--root-datadir', os.path.join(FOLDER_PATH,self.selected_directory)]
                subprocess.run(command)
        else:
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Question)
            message_box.setText("Already have Anomaly Images, do you want to re-generate it?")
            message_box.setWindowTitle("Get Anomaly Images")
            message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            result = message_box.exec_()
            if result == QMessageBox.Yes:
                command = ['python', 'get_anomaly_image_offline_ver2.py', '--root-datadir', os.path.join(FOLDER_PATH,self.selected_directory)]
                subprocess.run(command)
        '''
        command = ['python', 'get_anomaly_image_offline_ver2.py', '--root-datadir', os.path.join(FOLDER_PATH,self.selected_directory)]
        subprocess.run(command)
    #==================================================================================================
        #======Alister add 2023-03-27========================
        #self.selected_directory = self.directory_combo_box.currentText()
        self.image_combo_box.clear()
        #if not os.path.exists(os.path.join(self.selected_directory, 'anomaly_img_offline')):
            #os.makedirs(os.path.join(self.selected_directory, 'anomaly_img_offline'))
            
        #self.image_combo_box.addItems(self.get_image_files(os.path.join(FOLDER_PATH,self.selected_directory, 'anomaly_img_offline')))
        self.image_combo_box.addItems(self.get_image_files(os.path.join(FOLDER_PATH,self.selected_directory, 'anomaly_img_offline')))
        #2023-03-22 modified use  custom directory
        self.anomaly_clips_offline = os.path.join(FOLDER_PATH,self.selected_directory, 'anomaly_clips')
        
        if not os.path.exists(self.anomaly_clips_offline):
            os.makedirs(self.anomaly_clips_offline)
        #self.anomaly_clips_offline = self.selected_directory
        #self.anomaly_clips_offline = self.btn_select_dir.selected_dir
        # populate the QComboBox with a list of video files found in the directory
        self.smallclip_combo_box.clear()
        if not self.btn_select_dir.selected_dir=='':
            for filename in os.listdir(self.btn_select_dir.selected_dir): #self.anomaly_clips_offline self.btn_select_dir.selected_dir
                if filename.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    self.smallclip_combo_box.addItem(filename)
        else:
            for filename in os.listdir(self.anomaly_clips_offline): #self.anomaly_clips_offline self.btn_select_dir.selected_dir
                if filename.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    self.smallclip_combo_box.addItem(filename)
    
    def show_dialog(self):
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Delete Folder")
        dialog.setIcon(QMessageBox.Warning)
        dialog.setText("Are you sure you want to delete the folder?")
        dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        response = dialog.exec_()
        if response == QMessageBox.Yes:
            self.delete_dir()
        else:
            self.label.setText("Folder deletion cancelled")
    
    
    def delete_dir(self):
        folder_path = DATA_DIR  # replace with the path to the folder you want to delete
        try:
            shutil.rmtree(folder_path)
            #os.rmdir(folder_path)  # remove folder
            self.label.setText("Folder deleted successfully")
        except OSError as e:
            self.label.setText("Error: {} - {}".format(e.filename, e.strerror))
            

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
                #frame_count = file.split(".")[0]
                file_name = file.split(".")[0]
                frame_count = file_name.split(" ")[0]
                image_files.append(int(frame_count))
        
        image_files.sort()
        for i in range(len(image_files)):
            frame_count = image_files[i]
            frame_count_str = str(frame_count)
            #file = frame_count + ".jpg"
            #image_files[i] = os.path.join(directory, file)
            for file in os.listdir(directory):
                if file.endswith(".jpg") or file.endswith(".png"):
                    #image_files.append(os.path.join(directory, file))
                    #frame_count = file.split(".")[0]
                    file_name = file.split(".")[0]
                    frame_count_ori = file_name.split(" ")[0]
                    
                    if frame_count_ori==frame_count_str:
                        match_file = file
                        #image_files[i] = os.path.join(directory, match_file)
                        image_files[i] = match_file
                        
        self.load_file_list.clear()           
        for i, image_file in enumerate(image_files):
            self.load_file_list.append(image_file)
            image_files[i] = image_files[i].split()[1].split('.jpg')[0]

        return image_files
    '''
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
        else:
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Question)
            message_box.setText("Already collected, do you want to re-generate it?")
            message_box.setWindowTitle("Get Anomaly Images")
            message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            result = message_box.exec_()
            if result == QMessageBox.Yes:
                command = ['python', 'get_anomaly_image_offline_ver2.py', '--root-datadir', self.selected_directory]
                subprocess.run(command)
    
    def selected_directory_changed(self, index):
        self.selected_directory = self.directory_combo_box.currentText()
        self.image_combo_box.clear()
        #if not os.path.exists(os.path.join(self.selected_directory, 'anomaly_img_offline')):
            #os.makedirs(os.path.join(self.selected_directory, 'anomaly_img_offline'))
            
        self.image_combo_box.addItems(self.get_image_files(os.path.join(self.selected_directory, 'anomaly_img_offline')))
        #2023-03-22 modified use  custom directory
        self.anomaly_clips_offline = os.path.join(self.selected_directory, 'anomaly_clips')
        
        if not os.path.exists(self.anomaly_clips_offline):
            os.makedirs(self.anomaly_clips_offline)
        #self.anomaly_clips_offline = self.selected_directory
        #self.anomaly_clips_offline = self.btn_select_dir.selected_dir
        # populate the QComboBox with a list of video files found in the directory
        self.smallclip_combo_box.clear()
        if not self.btn_select_dir.selected_dir=='':
            for filename in os.listdir(self.btn_select_dir.selected_dir): #self.anomaly_clips_offline self.btn_select_dir.selected_dir
                if filename.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    self.smallclip_combo_box.addItem(filename)
        else:
            for filename in os.listdir(self.anomaly_clips_offline): #self.anomaly_clips_offline self.btn_select_dir.selected_dir
                if filename.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    self.smallclip_combo_box.addItem(filename)
    ''' 
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
        #video_path = os.path.join(self.btn_select_dir.selected_dir,self.smallclip_combo_box.currentText())
        if not self.btn_select_dir.selected_dir=='':
            video_path = os.path.join(self.btn_select_dir.selected_dir,self.smallclip_combo_box.currentText())
        else:
            video_path = os.path.join(FOLDER_PATH,self.selected_directory,"anomaly_clips",self.smallclip_combo_box.currentText())
        #args = ["vlc", "--fullscreen", video_path]
        #self.player_process = subprocess.Popen(args)
        
        args = ["/usr/bin/vlc", video_path]
        self.player_process = subprocess.Popen(args)
    #=====================================================================
    def selected_image_changed(self, index):
        #selected_image = self.image_combo_box.currentText()
        selected_image = self.load_file_list[self.image_combo_box.currentIndex()]; #TODO:
        folder_name = self.comboBox.currentText()
        #Alister 2023-03-28
        selected_image = os.path.join(DATA_DIR,folder_name,"anomaly_img_offline",selected_image)
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
                if not self.btn_select_dir.selected_dir=='':
                    command = ['python', 'gen_shortclips_ver2.py', '--anomaly-img', os.path.join("runs","detect",folder_name,"anomaly_img_offline",selected_image), '--root-datadir', os.path.join(FOLDER_PATH,self.selected_directory), '--save-anoclipdir', self.btn_select_dir.selected_dir]
                else:
                    command = ['python', 'gen_shortclips_ver2.py', '--anomaly-img', os.path.join("runs","detect",folder_name,"anomaly_img_offline",selected_image), '--root-datadir', os.path.join(FOLDER_PATH,self.selected_directory), '--save-anoclipdir', self.anomaly_clips_offline]
                subprocess.run(command)
                # populate the QComboBox with a list of video files found in the directory
                self.smallclip_combo_box.clear()
                if not self.btn_select_dir.selected_dir=='':
                    for filename in os.listdir(self.btn_select_dir.selected_dir): #self.anomaly_clips_offline self.btn_select_dir.selected_dir
                        if filename.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                            self.smallclip_combo_box.addItem(filename)
                else:
                    for filename in os.listdir(self.anomaly_clips_offline): #self.anomaly_clips_offline self.btn_select_dir.selected_dir
                        if filename.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                            self.smallclip_combo_box.addItem(filename)
                    
              
    def generate_short_clips_changed(self, state):
        pass

if __name__ == '__main__':
    app = QApplication([])
    window = ImageComboBox()
    window.show()
    app.exec_()


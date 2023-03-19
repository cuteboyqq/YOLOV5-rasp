# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 23:57:05 2023

@author: User
"""

from PyQt5.QtCore import Qt
import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QPushButton, QCheckBox, QFileDialog, QMessageBox, QVBoxLayout
import os
import logging
from datetime import datetime

logging.basicConfig(filename='example.log', level=logging.DEBUG)
class MainWindow(QMainWindow):
    def __init__(self,video_dir):
        super().__init__()
        
        
        #===============================================
        self.source_label = QLabel(self)
        self.source_label.setText("Play Recoding Raw Stream:")
        self.source_label.setGeometry(50, 300, 180, 30)
        self.video_dir = video_dir
        print(self.video_dir)
        self.player_process = None
        self.video_combobox = QComboBox(self)
        self.video_combobox.setGeometry(230, 300, 300, 30)
        #self.video_combobox = QComboBox()
        #self.video_combobox.setStyleSheet("background-color: #f2f2f2;")
        #self.video_combobox.currentIndexChanged.connect(self.play_selected_video)
        
        
        self.load_video_list()
        self.video_combobox.currentIndexChanged.connect(self.play_selected_video)
        
        
        # Create a frame to hold the combo box
        #self.frame = QFrame()
        #self.frame.setStyleSheet("background-color: #ffffff; border: 2px solid #000000;")  # Set background and border properties
        #self.frame.setMinimumSize(400, 400)  # Set minimum size
        
        #layout = QVBoxLayout(self.frame)
        #layout.setAlignment(Qt.AlignTop)  # Align the layout to the top
        #layout.addWidget(self.video_combobox)
        #===============================================
        
        self.setWindowTitle("Object Detection")
        self.setGeometry(100, 100, 600, 500)

        self.source_label = QLabel(self)
        self.source_label.setText("Select Source:")
        self.source_label.setGeometry(50, 50, 100, 30)

        self.source_combo = QComboBox(self)
        self.source_combo.addItem("Camera Stream")
        self.source_combo.addItem("Video Stream")
        self.source_combo.addItem("Images")
        self.source_combo.setGeometry(150, 50, 150, 30)

        self.viewimg_checkbox = QCheckBox(self)
        self.viewimg_checkbox.setText("View Output Images")
        self.viewimg_checkbox.setGeometry(50, 100, 200, 30)

        self.saveairesult_checkbox = QCheckBox(self)
        self.saveairesult_checkbox.setText("Save AI Results")
        self.saveairesult_checkbox.setGeometry(250, 100, 150, 30)

        self.browse_video_button = QPushButton(self)
        self.browse_video_button.setText("Browse Video")
        self.browse_video_button.setGeometry(50, 150, 120, 30)
        self.browse_video_button.clicked.connect(self.browse_video)

        self.browse_image_button = QPushButton(self)
        self.browse_image_button.setText("Browse Image")
        self.browse_image_button.setGeometry(200, 150, 120, 30)
        self.browse_image_button.clicked.connect(self.browse_image)

        self.run_button = QPushButton(self)
        self.run_button.setText("Run")
        self.run_button.setGeometry(150, 200, 100, 30)
        self.run_button.clicked.connect(self.run_command)

        self.video_path = ""
        self.images_path = ""
        
        
        # Set up layout
        layout = QVBoxLayout(self)
        # Add combo box for selecting video file
        #self.combo_videos = QComboBox(self)
        #self.combo_videos.setFixedWidth(200)
        #self.combo_videos.addItem("Select a video...")
        #self.combo_videos.currentTextChanged.connect(self.video_selected)
        #layout.addWidget(self.combo_videos)
    
        self.setLayout(layout)
    
    def load_video_list(self):
        for file_name in os.listdir(self.video_dir):
            if file_name.endswith(".mp4") or file_name.endswith(".avi"):
                self.video_combobox.addItem(file_name)

    def play_selected_video(self):
        if self.player_process:
            self.player_process.kill()
        video_path = os.path.join(self.video_dir, self.video_combobox.currentText())
        #args = ["vlc", "--fullscreen", video_path]
        #self.player_process = subprocess.Popen(args)
        
        args = ["C:/Program Files/VideoLAN/VLC/vlc.exe", video_path]
        self.player_process = subprocess.Popen(args)


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.player_process:
            self.player_process.kill()
    
    
    def browse_video(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video files (*.mp4 *.avi *.mov)")
        if file_dialog.exec_():
            self.video_path = file_dialog.selectedFiles()[0]

            
    def browse_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Image files (*.png *.jpg *.jpeg)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            self.images_path = file_dialog.selectedFiles()[0]
            
    def options_changed(self):
        # This method will be called when the options combobox changes
        self.run_button.setEnabled(True)
    '''
    def setup_ui(self):
        # Set up window
        self.setGeometry(100, 100, 500, 200)
        self.setWindowTitle("Object Detection")
    
        # Set up layout
        layout = QVBoxLayout(self)
    
        # Add combo box for selecting video file
        #self.combo_videos = QComboBox(self)
        #self.combo_videos.setFixedWidth(200)
        #self.combo_videos.addItem("Select a video...")
        #self.combo_videos.currentTextChanged.connect(self.video_selected)
        #layout.addWidget(self.combo_videos)
    
        # Add radio buttons for selecting detection source
        self.cb_options = QComboBox(self)
        self.cb_options.addItems(["Camera Stream", "Video Stream", "Images"])
        self.cb_options.currentIndexChanged.connect(self.options_changed)
        layout.addWidget(self.cb_options)
    
        # Add image selection button
        btn_image = QPushButton("Select Image", self)
        btn_image.clicked.connect(self.browse_image)
        layout.addWidget(btn_image)
    
        # Add check boxes for additional options
        self.cb_view_img = QCheckBox("View Image", self)
        self.cb_view_img.setChecked(True)
        layout.addWidget(self.cb_view_img)
    
        self.cb_save_airesult = QCheckBox("Save AI Results", self)
        self.cb_save_airesult.setChecked(True)
        layout.addWidget(self.cb_save_airesult)
    
        # Add run button
        btn_run = QPushButton("Run", self)
        btn_run.clicked.connect(self.run_command)
        layout.addWidget(btn_run)
    
        
        
        # Connect the directory button to the open_directory slot
        #self.directory_button = QPushButton("Open directory", self)
        #self.directory_button.clicked.connect(self.open_directory)
        #layout.addWidget(self.directory_button)
        
        self.setLayout(layout)
        # Load video file names into combo box
        #video_dir = r"D:/settings"  # Change this to the directory containing your video files
        #video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        #print(os.listdir(video_dir))
        #self.combo_videos.addItems(video_files)
        '''
    '''
    def open_directory(self):
        # Open a dialog to select the directory
        directory = QFileDialog.getExistingDirectory(self, r"C:/factory_data")
    
        # Update the directory label
        self.directory_label.setText(directory)
    
        # Update the video combobox
        self.video_combobox.clear()
        video_files = [f for f in os.listdir(directory) if f.endswith('.mp4') or f.endswith('.avi') or f.endswith('.mov')]
        print("list the videos")
        print(video_files)
        logging.debug(video_files)
        self.video_combobox.addItems(video_files)
        '''
    '''    
    def video_selected(self, text):
        if text == "Select a video...":
            self.video_path = None
        else:
            video_dir = r"C:/factory_data"  # Change this to the directory containing your video files
            self.video_path = os.path.join(video_dir, text)
'''
    
    def run_command(self):
        now = datetime.now()
        s_time = datetime.strftime(now,'%y-%m-%d_%H-%M-%S')
        str_s_time = "20"+str(s_time)
        source_type = self.source_combo.currentText()
        view_img = "--view-img" if self.viewimg_checkbox.isChecked() else ""
        save_airesult = "--save-airesult" if self.saveairesult_checkbox.isChecked() else ""
        weights = "--weights runs/train/f192_3cls_Argos_2023-03-11/weights/best.pt"
        img_size = "--img-size 192"
        data = "--data data/factory_new2.yaml"
        save_folder_name = "--name "+str_s_time
        #video_path = "/home/ali/factory_video/ori_video_ver2.mp4"
        detect_file = "detect-ori-log.py"
        if source_type == "Camera Stream":
            command = f"python {detect_file} --source 0 {weights} {img_size} {data} {view_img} {save_folder_name}"
        elif source_type == "Video Stream":
            if not self.video_path:
               QMessageBox.warning(self, "Warning", "Please select a video path.")
               return
            command = f"python {detect_file} {weights} {img_size} {data} --source {self.video_path} {view_img} {save_folder_name}"
        else:
            if not self.images_path:
                QMessageBox.warning(self, "Warning", "Please select an image path.")
                return
            command = f"python {detect_file} {weights} {img_size} {data} --source {self.images_path} {view_img} {save_folder_name}"

        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        print(output.decode("utf-8"))
        if error:
            print(error.decode("utf-8"))
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_dir = r"C:\GitHub_Code\cuteboyqq\YOLO"  # Replace with the path to your video folder
    window = MainWindow(video_dir)
    #window.setup_ui()

    window.show()
    app.exec_()
    #sys.exit(app.exec_())
'''
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
'''
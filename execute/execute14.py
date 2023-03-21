# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 23:57:05 2023

@author: User
"""

from PyQt5.QtCore import Qt
import sys
import subprocess
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QPushButton, QCheckBox, QFileDialog, QMessageBox, QVBoxLayout, QListWidget, QListWidgetItem 
import os
import logging
from datetime import datetime
shift_right = 50
logging.basicConfig(filename='example.log', level=logging.DEBUG)
#Setting the folder path for code  to use
FOLDER_PATH = r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect"
class MainWindow(QMainWindow):
    def __init__(self,video_dir):
        super().__init__()
        #================================================
        self.source_label = QLabel(self)
        self.source_label.setText("Select Folder To Generate Result-Video:")
        self.source_label.setFont(QFont("Times New Roman", 12))  # set the font size
        self.source_label.setGeometry(25, 250, 350, 30)
        
        
        # Create a QComboBox widget
        self.comboBox = QComboBox(self)
        self.comboBox.setFont(QFont("Times New Roman", 12))  # set the font size
        self.comboBox.setGeometry(350, 250, 300, 30)

        # Populate the QComboBox with folder names
        folder_path = FOLDER_PATH
        folder_names = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        self.comboBox.addItems(folder_names)

        # Connect the activated signal of the QComboBox to a slot
        self.comboBox.activated[str].connect(self.onComboBoxActivated)
        #================================================
        
        
        self.source_label = QLabel(self)
        self.source_label.setText("Saved OD Result:")
        self.source_label.setFont(QFont("Times New Roman", 12))  # set the font size
        self.source_label.setGeometry(25, 350, 300, 30)
        # create a combo box to select the folder
        self.folder_combobox = QComboBox(self)
        self.folder_combobox.setFont(QFont("Times New Roman", 12))  # set the font size
        self.folder_combobox.move(200, 355)
        self.folder_combobox.resize(300, 20)  # set the size of the list widget
        self.folder_combobox.activated.connect(self.on_folder_changed)

        # create a list widget to display the files
        self.file_listwidget = QListWidget(self)
        self.file_listwidget.move(25, 390)
        self.file_listwidget.resize(550, 200)  # set the size of the list widget
        self.file_listwidget.setFont(QFont("Times New Roman", 12))  # set the font size
        self.file_listwidget.itemDoubleClicked.connect(self.on_file_double_clicked)

        # initialize the folder combo box
        self.initialize_folder_combobox()
        
        #===============================================
        self.source_label = QLabel(self)
        self.source_label.setText("Saved raw stream:")
        self.source_label.setFont(QFont("Times New Roman", 12))  # set the font size
        self.source_label.setGeometry(25, 300, 300, 30)
        self.video_dir = video_dir
        print(self.video_dir)
        self.player_process = None
        self.video_combobox = QComboBox(self)
        self.video_combobox.setFont(QFont("Times New Roman", 12))  # set the font size
        self.video_combobox.setGeometry(200, 300, 300, 30)
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
        self.setGeometry(100, 100, 600, 600)

        self.source_label = QLabel(self)
        self.source_label.setText("Select Source:")
        self.source_label.setFont(QFont("Times New Roman", 12))  # set the font size
        self.source_label.setGeometry(50+shift_right, 50, 150, 30)

        self.source_combo = QComboBox(self)
        self.source_combo.addItem("Camera")
        self.source_combo.addItem("Video")
        self.source_combo.addItem("Images")
        self.source_combo.setFont(QFont("Times New Roman", 12))  # set the font size
        self.source_combo.setGeometry(180+shift_right, 50, 150, 30)

        self.viewimg_checkbox = QCheckBox(self)
        self.viewimg_checkbox.setText("View Live OD Stream")
        self.viewimg_checkbox.setFont(QFont("Times New Roman", 12))  # set the font size
        self.viewimg_checkbox.setGeometry(50+shift_right, 100, 200, 30)

        self.saveairesult_checkbox = QCheckBox(self)
        self.saveairesult_checkbox.setText("Save OD Stream")
        self.saveairesult_checkbox.setFont(QFont("Times New Roman", 12))  # set the font size
        self.saveairesult_checkbox.setGeometry(280+shift_right, 100, 180, 30)

        self.browse_video_button = QPushButton(self)
        self.browse_video_button.setText("Browse Video")
        self.browse_video_button.setFont(QFont("Times New Roman", 12))  # set the font size
        self.browse_video_button.setGeometry(60+shift_right, 150, 150, 40)
        self.browse_video_button.clicked.connect(self.browse_video)

        self.browse_image_button = QPushButton(self)
        self.browse_image_button.setText("Browse Image")
        self.browse_image_button.setFont(QFont("Times New Roman", 12))  # set the font size
        self.browse_image_button.setGeometry(240+shift_right, 150, 150, 40)
        self.browse_image_button.clicked.connect(self.browse_image)

        self.run_button = QPushButton(self)
        self.run_button.setText("Run")
        self.run_button.setFont(QFont("Times New Roman", 12))  # set the font size
        self.run_button.setGeometry(170+shift_right, 200, 100, 30)
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
        
    def checkProcessStatus(self, process):
        # Check the status of the subprocess
        if process.poll() is not None:
            # If the subprocess has finished, stop the QTimer and show a message box
            QTimer().stop()
            QMessageBox.information(self, "Success", "The process is done!")
        else:
            # If the subprocess has not finished yet, do nothing
            pass
        
    def onComboBoxActivated(self, folder_name):
        # Get the full path of the selected folder
        folder_path = os.path.join(FOLDER_PATH, folder_name)
        video_path = os.path.join(FOLDER_PATH,folder_name,"0.avi")
        # Set the folder path as a command parameter
        sys.argv = ['log_parser_ver2.py', '--log-dir', folder_path, '--video-path', video_path]
        
        

        # Execute the command using subprocess
        #command = ["python", "log_parser_ver2.py", "--log-dir", folder_path, '--video-path', video_path]
        
        
        
        if not os.path.exists(os.path.join(folder_path,"0.avi")) or not os.path.exists(os.path.join(folder_path,"log.txt")):
            QMessageBox.warning(self, "Error", "Need log.txt and 0.avi")
        else:
            if os.path.exists(os.path.join(folder_path,"0_result_offline.avi")):
                reply = QMessageBox.information(self, 'Integration', 'Already have 0_result_offline.avi, do you want to re-generate it?',
                QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
            else:
                reply = QMessageBox.information(self, 'Integration', 'Start to generate ai result video',QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
            if reply == QMessageBox.Ok:
                # Execute the command using subprocess
                command = ["python", "log_parser_ver2.py", "--log-dir", folder_path, "--video-path",video_path]
                #subprocess.run(command)
                try:
                    process = subprocess.run(command, check=True)
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
                
            
            
            #QMessageBox.information(None, 'Integration', 'Start to generate ai result video')
            # Execute the command using subprocess
            #command = ["python", "log_parser_ver2.py", "--log-dir", folder_path, "--video-path",video_path]
            #subprocess.run(command)
    
    
    
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
        
    #=========================================================================
    def initialize_folder_combobox(self):
        # get the specific directory
        specific_dir = FOLDER_PATH

        # get the list of folders in the directory
        folders = [f.name for f in os.scandir(specific_dir) if f.is_dir()]

        # add the folders to the combo box
        self.folder_combobox.addItems(folders)

    def on_folder_changed(self, index):
        # get the selected folder
        selected_folder = self.folder_combobox.currentText()

        # get the specific directory
        specific_dir = FOLDER_PATH

        # get the list of files in the selected folder
        file_list = os.listdir(os.path.join(specific_dir, selected_folder))

        # clear the list widget
        self.file_listwidget.clear()

        # add the files to the list widget
        for file_name in file_list:
            item = QListWidgetItem(file_name, self.file_listwidget)
            item.setToolTip(os.path.join(specific_dir, selected_folder, file_name))
            
            
            
    #=====================================================================================
    import subprocess

    def on_file_double_clicked(self, item):
        # get the selected folder
        selected_folder = self.folder_combobox.currentText()
    
        # get the specific directory
        specific_dir = FOLDER_PATH
    
        # get the full path to the selected file
        file_path = os.path.join(specific_dir, selected_folder, item.text())
    
        # check if the file exists
        if not os.path.exists(file_path):
            print(f"The file '{file_path}' does not exist.")
            return
    
        # check if the file is readable
        if not os.access(file_path, os.R_OK):
            print(f"You do not have permission to read the file '{file_path}'.")
            return
    
        # open the file with the default application
        try:
            if sys.platform == 'darwin':  # for MacOS
                subprocess.call(('open', file_path))
            elif sys.platform == 'linux':  # for Linux
                subprocess.call(('xdg-open', file_path))
            else:  # for Windows
                os.startfile(file_path)
        except Exception as e:
            print(f"Failed to open the file '{file_path}': {e}")

    '''
    def on_file_double_clicked(self, item):
        # get the selected folder
        selected_folder = self.folder_combobox.currentText()

        # get the specific directory
        specific_dir = r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect"

        # get the full path to the selected file
        file_path = os.path.join(specific_dir, selected_folder, item.text())

        # check if the file exists
        if not os.path.exists(file_path):
            print(f"The file '{file_path}' does not exist.")
            return

        # check if the file is readable
        if not os.access(file_path, os.R_OK):
            print(f"You do not have permission to read the file '{file_path}'.")
            return

        # open the file with the default application
        try:
            os.startfile(file_path)
        except Exception as e:
            print(f"Failed to open the file '{file_path}': {e}")
            '''
    #=========================================================================
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
        #save_folder_name = "--name "+str_s_time
        save_folder_name = ""
        #video_path = "/home/ali/factory_video/ori_video_ver2.mp4"
        detect_file = "detect-ori-log.py"
        if source_type == "Camera":
            command = f"python {detect_file} --source 0 {weights} {img_size} {data} {view_img} {save_folder_name} {save_airesult}"
        elif source_type == "Video":
            if not self.video_path:
               QMessageBox.warning(self, "Warning", "Please select a video path.")
               return
            command = f"python {detect_file} {weights} {img_size} {data} --source {self.video_path} {view_img} {save_folder_name} {save_airesult}"
        else:
            if not self.images_path:
                QMessageBox.warning(self, "Warning", "Please select an image path.")
                return
            command = f"python {detect_file} {weights} {img_size} {data} --source {self.images_path} {view_img} {save_folder_name} {save_airesult}"

        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        print(output.decode("utf-8"))
        if error:
            print(error.decode("utf-8"))
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_dir = FOLDER_PATH  # Replace with the path to your video folder
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
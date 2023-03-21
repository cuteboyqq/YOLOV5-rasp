# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 21:11:40 2023

@author: User
"""

import os
import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QComboBox, QWidget

class ComboBoxDemo(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the GUI
        self.setGeometry(100, 100, 300, 100)
        self.setWindowTitle('QComboBox Demo')

        # Create a QComboBox widget
        self.comboBox = QComboBox(self)
        self.comboBox.setGeometry(10, 10, 280, 30)

        # Populate the QComboBox with folder names
        folder_path = r"C:\GitHub_Code\cuteboyqq\YOLO\YOLOV5-rasp\runs\detect"
        folder_names = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        self.comboBox.addItems(folder_names)

        # Connect the activated signal of the QComboBox to a slot
        self.comboBox.activated[str].connect(self.onComboBoxActivated)

    def onComboBoxActivated(self, folder_name):
        # Get the full path of the selected folder
        folder_path = os.path.join(r"C:\GitHub_Code\cuteboyqq\YOLO\YOLOV5-rasp\runs\detect", folder_name)
        video_path = os.path.join(r"C:\GitHub_Code\cuteboyqq\YOLO\YOLOV5-rasp\runs\detect",folder_name,"0.avi")
        # Set the folder path as a command parameter
        sys.argv = ['log_parser_ver2.py', '--log-dir', folder_path, '--video-path', video_path]

        # Execute the command using subprocess
        command = ["python", " log_parser_ver2.py", "--log-dir ", folder_path, "--video-path ",video_path]
        subprocess.run(command)

if __name__ == '__main__':
    app = QApplication([])
    window = ComboBoxDemo()
    window.show()
    app.exec_()

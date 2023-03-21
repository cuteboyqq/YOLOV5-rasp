#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 19:58:44 2023

@author: ali
"""

import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QComboBox

class VideoListComboBox(QWidget):
    def __init__(self, directory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.directory = directory
        self.init_ui()

    def init_ui(self):
        # create the QComboBox widget
        self.comboBox = QComboBox()

        # populate the QComboBox with a list of video files found in the directory
        for filename in os.listdir(self.directory):
            if filename.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                self.comboBox.addItem(filename)

        # create the layout and add the QComboBox to it
        layout = QVBoxLayout()
        layout.addWidget(self.comboBox)
        self.setLayout(layout)

if __name__ == '__main__':
    # create the PyQt5 application
    app = QApplication(sys.argv)

    # create the main window
    window = QMainWindow()

    # create the VideoListComboBox widget and add it to the main window
    video_combo = VideoListComboBox(directory=r'/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp2/anomaly_clips')
    window.setCentralWidget(video_combo)

    # show the main window
    window.show()

    # run the PyQt5 event loop
    sys.exit(app.exec_())


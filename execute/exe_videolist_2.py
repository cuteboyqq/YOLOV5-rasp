#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 19:58:44 2023

@author: ali
"""

import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QComboBox, QHBoxLayout
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimediaWidgets import QVideoWidget


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

        # create the QMediaPlayer and QVideoWidget
        self.mediaPlayer = QMediaPlayer(self)
        self.videoWidget = QVideoWidget(self)

        # create a QHBoxLayout to hold the QComboBox and QVideoWidget
        hbox = QHBoxLayout()
        hbox.addWidget(self.comboBox)
        hbox.addWidget(self.videoWidget)

        # create the layout and add the QHBoxLayout to it
        layout = QVBoxLayout()
        layout.addLayout(hbox)
        self.setLayout(layout)

        # connect the currentIndexChanged signal of the QComboBox to the on_video_selected method
        self.comboBox.currentIndexChanged.connect(self.on_video_selected)

    def on_video_selected(self, index):
        # get the selected video filename
        filename = self.comboBox.currentText()

        # create a file path to the selected video
        filepath = os.path.join(self.directory, filename)

        # set the media content of the QMediaPlayer to the selected video
        media = QMediaContent(QUrl.fromLocalFile(filepath))
        self.mediaPlayer.setMedia(media)

        # set the QVideoWidget as the video output of the QMediaPlayer
        self.mediaPlayer.setVideoOutput(self.videoWidget)

        # start playing the video
        self.mediaPlayer.play()


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

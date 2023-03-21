#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:50:32 2023

@author: ali
"""

import os
import subprocess
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QWidget, QVBoxLayout, QCheckBox, QMessageBox, QLabel

class ImageComboBox(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image ComboBox")
        self.setGeometry(100, 100, 500, 250)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.directory_combo_box = QComboBox()
        self.directory_combo_box.addItems(self.get_subfolders(r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/"))  # Replace with the root directory path

        self.directory_combo_box.currentIndexChanged.connect(self.selected_directory_changed)

        self.layout.addWidget(self.directory_combo_box)

        self.image_combo_box = QComboBox()
        self.image_combo_box.currentIndexChanged.connect(self.selected_image_changed)

        self.layout.addWidget(self.image_combo_box)

        self.check_box = QCheckBox('Generate Short Clips', self)
        self.check_box.stateChanged.connect(self.generate_short_clips_changed)

        self.layout.addWidget(self.check_box)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

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

    def selected_directory_changed(self, index):
        selected_directory = self.directory_combo_box.currentText()
        self.image_combo_box.clear()
        self.image_combo_box.addItems(self.get_image_files(os.path.join(selected_directory, 'anomaly_img_offline')))

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
                command = ['python', 'gen_shortclips.py', '--anomaly-img', selected_image]
                subprocess.run(command)
    
    def generate_short_clips_changed(self, state):
        pass

if __name__ == '__main__':
    app = QApplication([])
    window = ImageComboBox()
    window.show()
    app.exec_()


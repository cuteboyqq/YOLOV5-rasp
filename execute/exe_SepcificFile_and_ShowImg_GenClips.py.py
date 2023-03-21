#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:20:23 2023

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

        self.comboBox = QComboBox()
        directory = r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp2/anomaly_img_offline"  # Replace with the directory path
        self.comboBox.addItems(self.get_image_files(directory))

        self.comboBox.currentIndexChanged.connect(self.selected_image_changed)

        self.layout.addWidget(self.comboBox)

        self.check_box = QCheckBox('Generate Short Clips', self)
        self.check_box.stateChanged.connect(self.generate_short_clips_changed)

        self.layout.addWidget(self.check_box)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

    def get_image_files(self, directory):
        image_files = []
        for file in os.listdir(directory):
            if file.endswith(".jpg") or file.endswith(".png"):
                #image_files.append(os.path.join(directory, file))
                file_name = file.split(".")[0]
                image_files.append(int(file_name))
                
        #sort the image_files
        image_files.sort()
        for i in range(len(image_files)):
            image_files[i] = str(image_files[i]) + ".jpg"
        return image_files

    def selected_image_changed(self, index):
        selected_image = self.comboBox.currentText()
        #selected_image = str(selected_image)
        #selected_image = selected_image + ".jpg"
        selected_image_path = os.path.join(r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp2/anomaly_img_offline",selected_image)
        pixmap = QPixmap(selected_image_path)
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

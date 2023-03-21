#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:13:18 2023

@author: ali
"""

import os
import subprocess
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QWidget, QVBoxLayout, QCheckBox, QMessageBox, QLabel, QFileDialog

class ImageComboBox(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image ComboBox")
        self.setGeometry(100, 100, 500, 250)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.comboBox = QComboBox()
        self.comboBox.addItems(self.get_image_files())

        self.comboBox.currentIndexChanged.connect(self.selected_image_changed)

        self.layout.addWidget(self.comboBox)

        self.check_box = QCheckBox('Generate Short Clips', self)
        self.check_box.stateChanged.connect(self.generate_short_clips_changed)

        self.layout.addWidget(self.check_box)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

    def get_image_files(self):
        directory = QFileDialog.getExistingDirectory(None, r"/home/ali/GitHub_Code/cuteboyqq/YOLO/YOLOV5-rasp/runs/detect/exp2/0_imgs", '')
        image_files = []
        for file in os.listdir(directory):
            if file.endswith(".jpg") or file.endswith(".png"):
                image_files.append(os.path.join(directory, file))
        return image_files

    def selected_image_changed(self, index):
        selected_image = self.comboBox.currentText()
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

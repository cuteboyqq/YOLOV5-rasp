#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:28:19 2023

@author: ali
"""

import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QPushButton, QCheckBox, QFileDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection")
        self.setGeometry(100, 100, 400, 300)

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
        self.viewimg_checkbox.setGeometry(50, 100, 150, 30)

        self.saveairesult_checkbox = QCheckBox(self)
        self.saveairesult_checkbox.setText("Save AI Results")
        self.saveairesult_checkbox.setGeometry(200, 100, 150, 30)

        self.browse_button = QPushButton(self)
        self.browse_button.setText("Browse")
        self.browse_button.setGeometry(150, 150, 100, 30)
        self.browse_button.clicked.connect(self.browse_video)

        self.run_button = QPushButton(self)
        self.run_button.setText("Run")
        self.run_button.setGeometry(150, 200, 100, 30)
        self.run_button.clicked.connect(self.run_command)

        self.video_path = ""

    def browse_video(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video files (*.mp4 *.avi *.mov)")
        if file_dialog.exec_():
            self.video_path = file_dialog.selectedFiles()[0]

    def run_command(self):
        source_type = self.source_combo.currentText()
        view_img = "--view-img" if self.viewimg_checkbox.isChecked() else ""
        save_airesult = "--save-airesult" if self.saveairesult_checkbox.isChecked() else ""
        weights = "--weights runs/train/f192_3cls_Argos_2023-03-11/weights/best.pt"
        img_size = "--img-size 192"
        data = "--data data/factory_new2.yaml"
        video_path = "/home/ali/factory_video/ori_video_ver2.mp4"
        if source_type == "Camera Stream":
            command = f"python detect-simple.py {weights} {img_size} {data} --source 0 {view_img} {save_airesult}"
        elif source_type == "Video Stream":
            command = f"python detect-simple.py {weights} {img_size} {data} --source {self.video_path} {view_img} {save_airesult}"
        else:
            command = f"python detect.py {weights} {img_size} {data} --source {images_path} {view_img} {save_airesult}"

        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        print(output.decode("utf-8"))
        if error:
            print(error.decode("utf-8"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
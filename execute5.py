# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 17:03:30 2023

@author: User
"""
import os
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QListView, QPushButton, QVBoxLayout #, QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Save Options")

        # Create a label for the options
        label = QLabel("Select options:")

        # Create a list view to select the options
        self.listView = QListView()
        self.listView.setSelectionMode(QListView.MultiSelection)
        self.listView_model = QStandardItemModel()
        for option_text in ["Save raw video", "Save AI result video", "Save anomaly images"]:
            item = QStandardItem(option_text)
            item.setCheckable(True)
            self.listView_model.appendRow(item)
        self.listView.setModel(self.listView_model)

        # Create a button to execute the command
        button = QPushButton("Execute Command")
        button.clicked.connect(self.execute_command)

        # Add the label, list view, and button to the layout
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.listView)
        layout.addWidget(button)

        self.setLayout(layout)

    def execute_command(self):
        # Get the selected options from the list view
        selected_options = [self.listView.model().item(i).text() for i in range(self.listView_model.rowCount()) if self.listView.model().item(i).checkState() == Qt.Checked]

        # Generate the command based on the selected options
        command_options = ""
        if "Save raw video" in selected_options:
            print("Save raw video")
            command_options += " --weights runs/train/f192_2022-12-29-4cls/weights/best.pt --data data/factory_new2.yaml --img-size 192 --source 0 --name 2023-03-11 --view-img"
            print(command_options)
        if "Save AI result video" in selected_options:
            command_options += " --weights runs/train/f192_2022-12-29-4cls/weights/best.pt --data data/factory_new2.yaml --img-size 192 --source C:/factory_data/ori_video_ver2.mp4 --name 2023-03-11 --view-img"
        if "Save anomaly images" in selected_options:
            command_options += " --weights runs/train/f192_2022-12-29-4cls/weights/best.pt --data data/factory_new2.yaml --img-size 192 --source C:/factory_data/ori_video_ver2.mp4 --name 2023-03-11 --view-img"

        # Execute the command
        print("Execute the command")
        command = "python detect.py" + command_options
        print(command)
        os.system(command)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
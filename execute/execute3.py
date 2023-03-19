# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 16:46:38 2023

@author: User
"""
import os
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Save Options")

        # Create a label for the options
        label = QLabel("Select an option:")

        # Create a combo box to select the options
        self.comboBox = QComboBox()
        self.comboBox.addItem("Save raw video")
        self.comboBox.addItem("Save AI result video")
        self.comboBox.addItem("Save anomaly images")

        # Create a button to execute the command
        button = QPushButton("Execute Command")
        button.clicked.connect(self.execute_command)

        # Add the label, combo box, and button to the layout
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.comboBox)
        layout.addWidget(button)

        self.setLayout(layout)

    def execute_command(self):
        # Get the selected option from the combo box
        selected_option = self.comboBox.currentText()

        # Execute the appropriate command based on the selected option
        if selected_option == "Save raw video":
            command = "python detect.py --save_raw_video"
        elif selected_option == "Save AI result video":
            command = "python detect.py --save_ai_video"
        else:
            command = "python detect.py --save_anomaly_images"

        # Execute the command
        os.system(command)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
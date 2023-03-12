# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 15:48:17 2023

@author: User
"""

import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton


class MyApp(QWidget):

    def __init__(self):
        super().__init__()

        # Create the button and set its properties
        self.button = QPushButton('Run detect.py', self)
        self.button.setToolTip('Click this button to run detect.py')
        self.button.resize(self.button.sizeHint())
        self.button.clicked.connect(self.run_detect)

        # Set the window properties
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Button Example')
        self.show()

    def run_detect(self):
        # Run the detect.py script using subprocess
        print("start run subprocess")
        subprocess.run(["python","detect.py --weights runs\train\f192_2022-12-29-4cls\weights\best.pt --data data\factory_new2.yaml --img-size 192 --source C:\factory_data\ori_video_ver2.mp4 --name 2023-03-11 --view-img"])
        print("finished run subprocess")

if __name__ == '__main__':
    # Create the PyQt5 application
    app = QApplication(sys.argv)
    # Create the main window
    window = MyApp()
    # Run the application
    sys.exit(app.exec_())

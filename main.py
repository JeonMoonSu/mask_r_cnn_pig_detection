from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5 import QtCore,QtGui,QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cv2
import os
import sys
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import skimage
from mrcnn import visualize
from mrcnn.visualize import display_instances

# epoch number
EPOCH_NUM = 10
SP_EPOCH = 100

ROOT_DIR = os.path.abspath("../../")
DATASET_DIR = ""
WEIGHTS_PATH = ""
IMAGE_PATH = ""

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR,"logs")

class TestWindow(QDialog):
    def _open_file_dialog1(self):
        directory = QFileDialog.getOpenFileName(self,'Open file','/home/default/logs')
        self.lineEdit1.setText(directory[0])

    def _open_file_dialog2(self):
        directory = QFileDialog.getOpenFileName(self,'Open file','/home/default')
        self.lineEdit2.setText(directory[0])
    
    def runClicked(self):
        global WEIGHTS_PATH
        global IMAGE_PATH
        WEIGHTS_PATH = self.lineEdit1.text()
        IMAGE_PATH = self.lineEdit2.text()
        
        if WEIGHTS_PATH == "" or IMAGE_PATH == "":
            QMessageBox.about(self,"error","You missed something")
        else:
            self.model.load_weights(WEIGHTS_PATH,by_name=True)
            masked_image=detect_and_color_splash(self.model, image_path=IMAGE_PATH,video_path="./")
            masked_image.show()
            print("끝")

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        class InferenceConfig(PigConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        self.config = InferenceConfig()
        self.config.display()
        self.model = modellib.MaskRCNN(mode="inference",config = self.config,model_dir = DEFAULT_LOGS_DIR)
        label1 = QLabel('Weight:')
        label2 = QLabel('Image:')

        self.lineEdit1 = QLineEdit()
        self.lineEdit1.setEnabled(False) 
        self.lineEdit1.setGeometry(QtCore.QRect(10, 10, 191, 20)) 

        self.lineEdit2 = QLineEdit()
        self.lineEdit2.setEnabled(False) 
        self.lineEdit2.setGeometry(QtCore.QRect(10, 10, 191, 20)) 

        self.toolButtonOpenDialog1 = QToolButton(self)
        self.toolButtonOpenDialog1.setGeometry(QtCore.QRect(210,10,25,19))
        self.toolButtonOpenDialog1.setObjectName("toolButtonOpenDialog")
        self.toolButtonOpenDialog1.clicked.connect(self._open_file_dialog1)

        self.toolButtonOpenDialog2 = QToolButton(self)
        self.toolButtonOpenDialog2.setGeometry(QtCore.QRect(210,10,25,19))
        self.toolButtonOpenDialog2.setObjectName("toolButtonOpenDialog")
        self.toolButtonOpenDialog2.clicked.connect(self._open_file_dialog2)

        self.pushButton = QPushButton("run test")
        self.pushButton.clicked.connect(self.runClicked)
        
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        self.retranslateUi(self) 
        QtCore.QMetaObject.connectSlotsByName(self) 
        #Layout

        myLayout1 = QHBoxLayout()
        myLayout2 = QHBoxLayout()

        myLayout1.addWidget(label1)
        myLayout1.addWidget(self.toolButtonOpenDialog1)

        myLayout2.addWidget(label2)
        myLayout2.addWidget(self.toolButtonOpenDialog2)
        
        self.leftLayout = QVBoxLayout()
        self.leftLayout.addWidget(self.canvas)

        rightLayout = QVBoxLayout()
        rightLayout.addLayout(myLayout1)
        rightLayout.addWidget(self.lineEdit1)
        rightLayout.addLayout(myLayout2)
        rightLayout.addWidget(self.lineEdit2)
        rightLayout.addWidget(self.pushButton)
        rightLayout.addStretch(1)

        layout = QHBoxLayout()
        layout.addLayout(self.leftLayout)
        layout.addLayout(rightLayout)
        layout.setStretchFactor(self.leftLayout,4)
        layout.setStretchFactor(rightLayout,1)
        
        self.setLayout(layout)
        self.setGeometry(100,100,900,600)
        
    def retranslateUi(self,TestQFileDialog):
        _translate = QtCore.QCoreApplication.translate
        TestQFileDialog.setWindowTitle(_translate("TestQFileDialog","Dialog"))
        self.toolButtonOpenDialog1.setIcon(QtGui.QIcon('./icon.png'))
        self.toolButtonOpenDialog2.setIcon(QtGui.QIcon('./icon.png'))


#train         
class TrainWindow(QDialog):
    def _open_file_dialog1(self):
        directory = str(QFileDialog.getExistingDirectory())
        self.lineEdit1.setText('{}'.format(directory))
        
    def _open_file_dialog2(self):
        directory = QFileDialog.getOpenFileName(self,'Open file','/home/default')
        self.lineEdit2.setText(directory[0])

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        #train model
        self.config = PigConfig()
        self.config.display()
        self.model = modellib.MaskRCNN(mode = "training", config = self.config, model_dir = DEFAULT_LOGS_DIR)

        #pyqt
        layout = QGridLayout()
        boxlayout = QBoxLayout(QBoxLayout.TopToBottom,self)

        self.toolButtonOpenDialog1 = QToolButton(self)
        self.toolButtonOpenDialog1.setGeometry(QtCore.QRect(210,10,25,19))
        self.toolButtonOpenDialog1.setObjectName("toolButtonOpenDialog")
        self.toolButtonOpenDialog1.clicked.connect(self._open_file_dialog1)

        self.toolButtonOpenDialog2 = QToolButton(self)
        self.toolButtonOpenDialog2.setGeometry(QtCore.QRect(210,10,25,19))
        self.toolButtonOpenDialog2.setObjectName("toolButtonOpenDialog2")
        self.toolButtonOpenDialog2.clicked.connect(self._open_file_dialog2)
            
        self.lineEdit1 = QLineEdit(self) 
        self.lineEdit1.setEnabled(False) 
        self.lineEdit1.setGeometry(QtCore.QRect(10, 10, 191, 20)) 
        
        self.lineEdit2 = QLineEdit(self) 
        self.lineEdit2.setEnabled(False) 
        self.lineEdit2.setGeometry(QtCore.QRect(10, 10, 191, 20)) 
         
        self.retranslateUi(self) 
        QtCore.QMetaObject.connectSlotsByName(self) 

        self.setGeometry(1100,200,300,100)
        self.setWindowTitle("train")
        
        btn1 = QPushButton("Start Training")
        btn2 = QPushButton("Close")

        btn1.clicked.connect(self.trainClicked)
        btn2.clicked.connect(self.closeClicked)

        self.groupbox = QGroupBox("",self)
        self.groupbox.setLayout(boxlayout)

        self.chk1 = QRadioButton("COCO",self)
        self.chk2 = QRadioButton("LAST",self)
        self.chk3 = QRadioButton("Other",self)

        boxlayout.addWidget(self.chk1)
        boxlayout.addWidget(self.chk2)
        boxlayout.addWidget(self.chk3)

        label1 = QLabel("Epoch : ")
        label2 = QLabel("Numbers per epoch : ")
        label3 = QLabel("Location of dataset : ")
        label4 = QLabel("Weight : ")

        self.spinBox1 = QSpinBox(self)
        self.spinBox2 = QSpinBox(self)

        self.spinBox1.setMaximum(1000)
        self.spinBox2.setMaximum(1000)
        
        self.spinBox1.setValue(10)
        self.spinBox2.setValue(100)

        layout.addWidget(label1,0,0)
        layout.addWidget(label2,1,0)
        layout.addWidget(label3,2,0)
        layout.addWidget(label4,4,0)

        layout.addWidget(self.spinBox1,0,1)
        layout.addWidget(self.spinBox2,1,1)
        layout.addWidget(self.toolButtonOpenDialog1,2,1)
        layout.addWidget(self.lineEdit1,3,0,1,2)
        layout.addWidget(self.groupbox,4,1)
        layout.addWidget(self.toolButtonOpenDialog2,5,1)
        layout.addWidget(self.lineEdit2,6,0,1,2)

        layout.addWidget(btn1,7,0)
        layout.addWidget(btn2,7,1)
        self.setLayout(layout)
        self.setGeometry(300,300,300,200)

    def trainClicked(self):
        global EPOCH_NUM
        global SP_EPOCH
        
        EPOCH_NUM = self.spinBox1.value()
        SP_EPOCH = self.spinBox2.value()
        self.config.STEPS_PER_EPOCH = SP_EPOCH

        #dataset path
        global DATASET_DIR 
        DATASET_DIR = self.lineEdit1.text()

        if DATASET_DIR == "":
            QMessageBox.about(self,"error","Set dataset directory!")
     
        #weights path
        global WEIGHTS_PATH
        WEIGHTS_PATH = self.lineEdit2.text()

        if self.chk1.isChecked():
                WEIGHTS_PATH = COCO_WEIGHTS_PATH
        elif self.chk2.isChecked():
                WEIGHTS_PATH = self.model.find_last()
        elif self.chk3.isChecked():
            if WEIGHTS_PATH == "":
                QMessageBox.about(self,"error","Set weight directory!")
        else:
            QMessageBox.about(self,"error","Check weight!")

        #load weights
        if DATASET_DIR == "" or WEIGHTS_PATH == "":
            print("하기전에 좀 체크좀해라 ")
        else:
            print("epoch_num is : ",EPOCH_NUM)
            print("steps per epoch is : ",SP_EPOCH)
            print("dataset directory is : ",DATASET_DIR)
            print("weights directory is : ",WEIGHTS_PATH)

            if WEIGHTS_PATH == COCO_WEIGHTS_PATH:
                self.model.load_weights(WEIGHTS_PATH,by_name = True, exclude=["mrcnn_class_logits","mrcnn_bbox_fc","mrcnn_bbox","mrcnn_mask"])
            else:
                self.model.load_weights(WEIGHTS_PATH,by_name=True)
            train(self.model)
            self.close()

    def closeClicked(self):
        self.close()
              
    def retranslateUi(self,TestQFileDialog):
        _translate = QtCore.QCoreApplication.translate
        TestQFileDialog.setWindowTitle(_translate("TestQFileDialog","Dialog"))
        self.toolButtonOpenDialog1.setText(_translate("TestQfileDialog","..."))
        self.toolButtonOpenDialog2.setText(_translate("TestQfileDialog2","..."))   
    
class ExWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):

        label = QLabel('Mask R-CNN PIG',self)
        label.setAlignment(Qt.AlignCenter)

        font = label.font()
        font.setBold(True)
        
        btn1 = QPushButton('Train',self)
        btn2 = QPushButton('Test',self)
        btn3 = QPushButton('Close',self)
        
        btn1.clicked.connect(self.trainEvent)
        btn2.clicked.connect(self.testEvent)
        btn3.clicked.connect(self.close)

        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addWidget(btn1)
        vbox.addWidget(btn2)
        vbox.addWidget(btn3)

        self.setLayout(vbox)

        self.setGeometry(800,200,300,300)
        self.show()

    def testEvent(self):
        dia = TestWindow()
        dia.exec_()

    def trainEvent(self):
        dia = TrainWindow()
        dia.exec_()

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?',QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

class PigConfig(Config):

    NAME = "pig"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    NUM_CLASSES = 1 + 2  # Background + objects

    STEPS_PER_EPOCH = SP_EPOCH

    DETECTION_MIN_CONFIDENCE = 0.9

class PigDataset(utils.Dataset):
    def load_VIA(self, dataset_dir, subset, hc=False):

        self.add_class("pig", 1, "standing_pig")
        self.add_class("pig", 2, "lying_pig")

        assert subset in ["train","val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations1 = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations1.values())  
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
# Get the x, y coordinaets of points of the polygons that make up
# the outline of each object instance. There are stores in the
# shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            names = [r['region_attributes'] for r in a['regions'].values()]
# load_mask() needs the image size to convert polygons to masks.
# Unfortunately, VIA doesn't include it in JSON, so we must read
# the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "pig",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                names=names)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "pig":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        class_ids = np.zeros([len(info["polygons"])])

        for i, p in enumerate(class_names):
            if p['name'] == 'standing_pig':
                class_ids[i] = 1
            elif p['name'] == 'lying_pig':
                class_ids[i] = 2
#assert code here to extend to other labels
        class_ids = class_ids.astype(int)
# Return mask, and array of class IDs of each instance. Since we have
# one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "pig":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def detect_and_color_splash(model, image_path=None, video_path=None, out_dir=''):
    assert image_path or video_path

    class_names = ['BG', 'standing_pig', 'lying_pig']

# Image or video?
    if image_path:
        print("Running on {}".format(IMAGE_PATH))
# Read image
        image = skimage.io.imread(IMAGE_PATH)
# Detect objects
        r = model.detect([image], verbose=1)[0]
# Color splash and save
        masked_image = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
            class_names, r['scores'],"image")
        return masked_image

    elif video_path:
# Video capture
        vcapture = cv2.VideoCapture(video_path)
# width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = 1600
        height = 1600
        fps = vcapture.get(cv2.CAP_PROP_FPS)
# Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.wmv".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
            cv2.VideoWriter_fourcc(*'MJPG'),
            fps, (width, height))

        count = 0
        success = True
#For video, we wish classes keep the same mask in frames, generate colors for masks
        colors = visualize.random_colors(len(class_names))
        while success:
            print("frame: ", count)
            plt.clf()
            plt.close()
            success, image = vcapture.read()
            if success:
# OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
# Detect objects
                r = model.detect([image], verbose=0)[0]
# Color splash
# splash = color_splash(image, r['masks'])

                splash = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
            class_names, r['scores'], colors=colors, making_video=True)
# Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()

def train(model):
    dataset_train = PigDataset()
    dataset_train.load_VIA(DATASET_DIR, "train")
    dataset_train.prepare()

    dataset_val = PigDataset()
    dataset_val.load_VIA(DATASET_DIR, "val")
    dataset_val.prepare()
    print("Training network heads")
    model.train(dataset_train, dataset_val,
            learning_rate=PigConfig().LEARNING_RATE,
            epochs=model.epoch+EPOCH_NUM,
            layers='heads')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ExWindow()
    sys.exit(app.exec_())


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'single-window.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot
import multiprocessing
import sys
import requests
# from detect import Detector
from utils.utils import *
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import datetime
from requests_toolbelt.multipart.encoder import MultipartEncoder

class Ui_MainWindow(object):
    def __init__(self, MainWindow):
        self.setupUi(MainWindow)
        self.slot_init(MainWindow)
        self.urlAddress = 'www.msyzjut.cn:8080'
        self.apiName = '/warningLog'
        self.withoutHelmet = "0"
        self.Wno="001"
        # self.p1 = multiprocessing.Process(target=self.alarm())
        self.cfg = 'cfg/yolov3-spp.cfg'
        self.names = 'data/rbc.names'
        self.weights = 'weights/loss4.75.pt'
        self.source = '0'
        # self.source='rtsp://admin:admin@172.20.10.4:554/stream1'
        self.output = 'output'
        self.img_size = 416
        self.conf_thres = 0.20
        self.iou_thres = 0.5
        self.fourcc = 'mp4v'
        self.half = False
        self.device = ''
        self.view_img = False
        self.save_txt = False
        self.agnostic_nms = False
        self.classes = None
        MainWindow.show()


    def change_source(self,text):
        self.source=text

    def slot_init(self, MainWindow):
        self.pushButton.clicked.connect(self.pushButton_click)
        # self.pushButton.clicked.connect(self.send)
    def pushButton_click(self):
        self.pushButton.setText("结束检测")
        text1 = self.lineEdit_2.text()
        if text1 != '':
            self.change_source(text1)
        self.detect()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(780, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        self.formLayout = QtWidgets.QFormLayout(self.centralwidget)
        self.formLayout.setObjectName("formLayout")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setSpacing(0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(-1, -1, 9, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(40)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setMouseTracking(False)
        self.pushButton.setAutoDefault(False)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_2.addWidget(self.pushButton)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(8, -1, -1, -1)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, 8, -1, 8)
        self.horizontalLayout.setSpacing(9)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(40)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMouseTracking(False)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(40)
        sizePolicy.setHeightForWidth(self.lineEdit_2.sizePolicy().hasHeightForWidth())
        self.lineEdit_2.setSizePolicy(sizePolicy)
        self.lineEdit_2.setMouseTracking(False)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout.addWidget(self.lineEdit_2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(-1, 8, -1, -1)
        self.horizontalLayout_2.setSpacing(54)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setMouseTracking(False)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit.sizePolicy().hasHeightForWidth())
        self.lineEdit.setSizePolicy(sizePolicy)
        self.lineEdit.setMouseTracking(False)
        self.lineEdit.setObjectName("lineEdit")
        # self.lineEdit.setText("3333")
        self.horizontalLayout_2.addWidget(self.lineEdit)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 80)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.gridLayout_3.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 0, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout_3.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.gridLayout_3.setRowStretch(0, 10)
        self.gridLayout_3.setRowStretch(1, 2)
        self.formLayout.setLayout(0, QtWidgets.QFormLayout.SpanningRole, self.gridLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.label.setBuddy(self.lineEdit)
        self.label_2.setBuddy(self.lineEdit)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "安全帽佩戴检测系统"))
        self.pushButton.setText(_translate("MainWindow", "开始检测"))
        self.label.setText(_translate("MainWindow", "待检测视频路径："))
        self.label_2.setText(_translate("MainWindow", "报警信息："))

    def detect(self):
        save_img = False
        img_size = (320, 192) if ONNX_EXPORT else self.img_size

        out, source, weights, half, view_img, save_txt = self.output, self.source, self.weights, self.half, self.view_img, self.save_txt
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        save_path="./"
        device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else self.device)
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out)
        model = Darknet(self.cfg, img_size)
        attempt_download(weights)
        if weights.endswith('.pt'):
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:
            load_darknet_weights(model, weights)

        model.to(device).eval()
        # vid_write=None
        if webcam:
            view_img = True
            torch.backends.cudnn.benchmark = True
            dataset = LoadStreams(source, img_size=img_size, half=half)
        else:

            view_img = True
            dataset = LoadImages(source, img_size=img_size, half=half)

        names = load_classes(self.names)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        count = 0
        vid_writer = cv2.VideoWriter("./1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480), 1)
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)

            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img)[0]

            if self.half:
                pred = pred.float()

            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                       agnostic=self.agnostic_nms)



            for i, det in enumerate(pred):

                person = False
                sum = 0
                count += 1
                if webcam:
                    p, s, im0 = path[i], '%g: ' % i, im0s[i]
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                # print(save_path[:18])
                s += '%gx%g ' % img.shape[2:]
                if det is not None and len(det):

                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                        if names[int(c)] == "person":
                            sum = '%g ' % n
                            person = True

                    for *xyxy, conf, cls in det:
                        if save_txt:
                            with open(save_path[:18] + '.txt', 'a') as file:
                                file.write(('%s '+'%g ' * 5 + '\n') % (names[int(cls)], conf,*xyxy))

                        if save_img or view_img:
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                if person and count == 20:
                    count = 0
                    self.withoutHelmet = str(sum)
                    curr_time = datetime.datetime.now()
                    # print('Last found: There is ' + sum + 'person(s) without helmet.' + str(curr_time))
                    self.lineEdit.setText('Last found: There is ' + sum + 'person(s) without helmet.' + str(
                        curr_time))  # put your api here
                    cv2.imwrite('./1.jpg', im0)
                    # self.p1.start()
                    # os.system('paplay Alarm.wav')
                    # self.send
                    self.withoutHelmet = "0"

                if count == 20:
                    count = 0
                # cv2.waitKey(50)
                if view_img:
                    if source.startswith('rtsp'):
                        im0 = cv2.resize(im0, (1280, 720))
                    cv2.imshow(p, im0)
                    vid_writer.write(im0)
                    show = cv2.resize(im0, (720, 500))
                    show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                    showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                    self.label_5.setPixmap(QtGui.QPixmap.fromImage(showImage))

                    if cv2.waitKey(1) == ord('q'):  # q to quit

                        vid_writer.release()
                        # break
                        return


                # if isinstance(vid_writer, cv2.VideoWriter):
                #     vid_writer.release()  # release previous video writer




        # 向服务器发送信号
        # 这个函数不用管，我用来做测试的，你可以参考一下这个流程

    def alarm(self):
        os.system('paplay Alarm.wav')


    def send(self):
        req=self.start1()
        # QMessageBox.question(self, "Message", req.text, QMessageBox.Ok, QMessageBox.Ok)  # 弹窗

    def start1(self):
        request_data=MultipartEncoder(
            fields={'Wno':'001','file':('6.png',open('./6.png','rb'))})
        header={'Content-Type':request_data.content_type}
        return requests.post('http://www.msyzjut.cn:8080/WarningMessage',data=request_data,headers=header)




    @pyqtSlot()
    def sendMessage(self):


        req = self.doRequest(url, self.withoutHelmet,self.Wno)  # 调用函数发送请求
        QMessageBox.question(self, "Message", req.text, QMessageBox.Ok, QMessageBox.Ok)  # 弹窗

    def doRequest(self, url, withoutHelmet,Wno):
        payload = {'withoutHelmet': withoutHelmet,'Wno': Wno}  # 用字典形式封装
        return requests.post('http://www.msyzjut.cn:8080/warningLog', data=payload)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = QMainWindow(None)
    ex = Ui_MainWindow(widget)
    sys.exit(app.exec_())
    pass

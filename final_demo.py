from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot

import sys
import requests
import os
import json
import numpy as np
import cv2
import base64
# from detect import Detector
from utils.utils import *
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from requests_toolbelt.multipart.encoder import MultipartEncoder
import datetime



class MyWindow(QWidget):
    def __init__(self):
        self.urlAddress = 'www.msyzjut.cn:8080'
        self.apiName = '/warningLog '
        self.withoutHelmet = 0  # 被检测出没有戴安全帽的人数
        super(MyWindow, self).__init__()
        self.setWindowTitle("安全帽检测系统")
        self.resize(900, 600)

        self.cfg = 'cfg/yolov3-spp.cfg'
        self.names = 'data/rbc.names'
        self.weights = 'weights/last.pt'


        self.source = '0'
        # self.source='rtsp://admin:admin@192.168.1.107:554/stream1'


        self.output = 'output'
        self.img_size = 416
        self.conf_thres = 0.25
        self.iou_thres = 0.5
        self.fourcc = 'mp4v'
        self.half = False
        self.device = ''
        self.view_img = False
        self.save_txt = False
        self.agnostic_nms = False
        self.classes = None


        self.initUI()
        self.show()


    def change_source(self,text):
        self.source=text



    def initUI(self):
        Layout=QGridLayout(self)

        # 菜单栏
        menuBar = QMenuBar()
        # menuBar.addMenu('文件')
        # menuBar.addMenu('视频')
        Layout.addWidget(menuBar,0,0)
        # menuBar.setNativeMenuBar(False)  # MacOS平台下为了显示菜单，Windows下可以删去

        # 左上方，按钮布局
        buttonLayout=QVBoxLayout()
        # btn2=QPushButton("发送信号")
        btn3=QPushButton("开始检测")
        # buttonLayout.addWidget(btn2)
        buttonLayout.addWidget(btn3)
        Layout.addLayout(buttonLayout,2, 0)

        # 右上方，视频布局
        vedioLayout=QGridLayout()

        # lblImage = QLabel()
        # lblImage.setText("")
        # lblImage.setObjectName("lblImage")
        # vedioLayout.addWidget(lblImage)
        # Layout.addLayout(vedioLayout,1,1)

        # lblImage.setGeometry(QtCore.QRect(0, 0, 300, 200))  # 640 480
        # lblImage.setMouseTracking(False)




        # 划分布局比例
        Layout.setColumnStretch(0, 1)
        Layout.setColumnStretch(1, 7)

        # 下方，报警信息布局
        messageLayout=QGridLayout()
        label1=QLabel('待检测视频路径：')
        label2=QLabel('报警信息：')
        self.line1=QLineEdit()
        # self.line1.setText(self.urlAddress+self.apiName)
        self.line2=QLineEdit()
        messageLayout.addWidget(label1,0,0)
        messageLayout.addWidget(self.line1,0,1)
        messageLayout.addWidget(label2,1,0)
        messageLayout.addWidget(self.line2,1,1)
        Layout.addLayout(messageLayout,2,1)

        #按钮连接槽函数
        # btn2.clicked.connect(self.sendMessage)
        btn3.clicked.connect(self.start)

    def detect(self):
        save_img = False
        img_size = (320, 192) if ONNX_EXPORT else self.img_size

        out, source, weights, half, view_img, save_txt = self.output, self.source, self.weights, self.half, self.view_img, self.save_txt
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

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

        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            torch.backends.cudnn.benchmark = True
            dataset = LoadStreams(source, img_size=img_size, half=half)
        else:
            save_img = True
            view_img = True
            dataset = LoadImages(source, img_size=img_size, half=half)

        names = load_classes(self.names)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        count=0
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)

            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img)[0]

            if self.half:
                pred = pred.float()

            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

            for i, det in enumerate(pred):
                person=False
                sum=0
                count += 1
                if webcam:
                    p, s, im0 = path[i], '%g: ' % i, im0s[i]
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]
                if det is not None and len(det):

                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string



                        if names[int(c)]=="person" :
                            sum = '%g ' % n
                            person=True



                    for *xyxy, conf, cls in det:
                        if save_txt:
                            with open(save_path + '.txt', 'a') as file:
                                file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                        if save_img or view_img:
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                if person and count==20:
                    count=0
                    self.withoutHelmet=sum
                    curr_time = datetime.datetime.now()
                    # print('Last found: There is ' +sum+ 'person(s) without helmet.'+str(curr_time))
                    self.line2.setText('Last found: There is ' +sum+ 'person(s) without helmet.'+str(curr_time))   #put your api here
                    cv2.imwrite('./1.jpg', im0)
                    # self.sendMessage
                    self.withoutHelmet=0




                if count==20:
                    count=0
                if view_img:
                    if source.startswith('rtsp'):
                        im0 = cv2.resize(im0, (1280, 720))
                    cv2.imshow(p, im0)
                    # vid_writer.write(im0)
                    show = cv2.resize(im0, (720, 500))
                    show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                    showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                    self.label_5.setPixmap(QtGui.QPixmap.fromImage(showImage))

                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        # break
                        return


                # if save_img:
                #     if dataset.mode == 'images':
                #         cv2.imwrite(save_path, im0)
                #     else:
                #         if vid_path != save_path:  # new video
                #             vid_path = save_path
                #             if isinstance(vid_writer, cv2.VideoWriter):
                #                 vid_writer.release()  # release previous video writer
                #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*self.fourcc), fps, (w, h))
                #         vid_writer.write(im0)

        if save_txt or save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + out)




    def send(self):
        req=self.start1()
        QMessageBox.question(self, "Message", req.text, QMessageBox.Ok, QMessageBox.Ok)  # 弹窗

    def start1(self):
        request_data=MultipartEncoder(
            fields={'Wno':'001','file':('1.jpg',open('./1.jpg','rb'))})

        header={'Content-Type':request_data.content_type}
        return requests.post('https://www.msyzjut.cn:8443/warningLog',data=request_data,headers=header)


    # 向服务器发送信号
    # 这个函数不用管，我用来做测试的，你可以参考一下这个流程
    @pyqtSlot()
    def sendMessage(self):
        url = self.urlAddress + self.apiName  # 发起请求的url
        # self.withoutHelmet = 3  # 设置有几个人没有戴安全帽
        req = self.doRequest(url,self.withoutHelmet)  # 调用函数发送请求
        # QMessageBox.question(self, "Message", req.text, QMessageBox.Ok, QMessageBox.Ok)  # 弹窗

    def doRequest(self,url,withoutHelmet):
        payload = {'withoutHelmet': withoutHelmet}  # 用字典形式封装
        return requests.post(url,data=payload)

    # 在这个函数里调用你自己的detect类，开始检测
    # 如果你已经封装好了detect类，检测的过程应该是一个函数，把检测到的人数作为返回值，输入到self.withoutHelmet里就可以了
    @pyqtSlot()
    def start(self):

        # QMessageBox.question(self, "Message", "开始检测", QMessageBox.Ok, QMessageBox.Ok)  # 弹窗
        text1=self.line1.text()
        if text1 !='':
            self.change_source(text1)
        # with torch.no_grad():
        self.detect()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())

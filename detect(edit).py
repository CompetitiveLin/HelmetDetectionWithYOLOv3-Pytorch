import argparse
import os
from argparse import ArgumentParser
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


class Detector():
    def __init__(self):
        self.cfg='cfg/yolov3-spp.cfg'

    def detect(self):
        save_img = False
        img_size = (320, 192) if ONNX_EXPORT else opt.img_size

        out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out)
        model = Darknet(opt.cfg, img_size)
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

        names = load_classes(opt.names)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        count=0
        for path, img, im0s, vid_cap in dataset:



            img = torch.from_numpy(img).to(device)

            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img)[0]

            if opt.half:
                pred = pred.float()

            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            print(opt.classes)


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
                    print('There is ' +sum+ 'person(s) without helmet.')
                if count==20:
                    count=0
                if view_img:
                    if source.startswith('rtsp'):
                        im0 = cv2.resize(im0, (1280, 720))
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration


                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + out)



if __name__ == '__main__':
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/rbc.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/loss4.75.pt', help='path to weights file')
    # parser.add_argument('--source', type=str, default='rtsp://admin:admin@192.168.1.107:554/stream1', help='source')  # input file/folder, 0 for webcam

    parser.add_argument('--source', type=str, default='0', help='source')  # input file/folder, 0 for webcam

    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    x=Detector()
    with torch.no_grad():
        x.detect()

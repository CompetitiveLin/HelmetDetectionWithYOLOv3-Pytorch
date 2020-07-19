from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


class Detector():
    def __init__(self):
        self.cfg='cfg/yolov3-spp.cfg'
        self.names='data/rbc.names'
        self.weights='weights/loss4.75.pt'
        self.source='0'
        self.output='output'
        self.img_size=416
        self.conf_thres=0.25
        self.iou_thres=0.5
        self.fourcc='mp4v'
        self.half=False
        self.device=''
        self.view_img=False
        self.save_txt=False
        self.agnostic_nms=False
        self.classes=None
        self.sum0=0


    def change_source(self,text):
        self.source=text

    def get_sum(self):
        return self.sum0

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
                    self.sum0=sum
                    print('There is ' +sum+ 'person(s) without helmet.')




                if count==20:
                    count=0
                if view_img:
                    if source.startswith('rtsp'):
                        im0 = cv2.resize(im0, (1280, 720))
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        # break
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
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*self.fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + out)

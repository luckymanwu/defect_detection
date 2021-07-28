
class Opt(object):
    def __init__(self):
        super(Opt, self).__init__()
        self.agnostic_nms=False
        self.augment=False
        self.cfg='cfg/yolov3-tiny.cfg'
        self.classes=None
        self.conf_thres=0.1
        self.device=''
        self.fourcc='mp4v'
        self.half=False
        self.img_size=512
        self.iou_thres=0.6
        self.names='../data/rbc.names'
        self.output='../output'
        self.save_txt=True
        self.source='../data/samples'
        self.view_img=False
        self.weights='../weights/best.pt'
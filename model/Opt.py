class Opt(object):
    def __init__(self):
        super(Opt, self).__init__()
        self.fps = 10
        self.width = 400
        self.height = 360
        self.agnostic_nms=False
        self.augment=False
        self.cfg='E:\defect_detection-main\cfg\yolov3-tiny.cfg'
        self.classes=None
        self.conf_thres=0.1
        self.device=''
        self.fourcc='mp4v'
        self.half=False
        self.img_size=512
        self.iou_thres=0.6
        self.names = 'C:/Users/Administrator/Desktop/cixinDataSet/rbc.names'
        self.output='../output'
        self.save_txt=True
        self.view_img=False
        self.weights='../weights/best.pt'
        self.epochs = 800
        self.batchSize = 16
        self.confidence = 80


# -*- coding: utf-8 -*-

from Service.models import *  # set ONNX_EXPORT in models.py
from project.datasets import *
from project.utils import *

class hkDetect:
    def __init__(self,opt=None):
        self.opt = opt
        self.init_model(opt)

    def init_model(self,opt):
        imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        weights, half = opt.weights, opt.half
        defect_class = ''
        # Initialize
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)  # using cpu
        # Initialize model，调用darknet模型
        self.model = Darknet(opt.cfg, imgsz)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
            # model.load_state_dict(torch.load(weights, map_location='cpu'))
        else:  # darknet format
            load_darknet_weights(self.model, weights)

        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
            modelc.to(self.device).eval()

        # Eval mode
        self.model.to(self.device).eval()

        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()

        # Export mode
        if ONNX_EXPORT:
            self.model.fuse()
            img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
            f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
            torch.onnx.export(self.model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

            # Validate exported model
            import onnx
            model = onnx.load(f)  # Load the ONNX model
            onnx.checker.check_model(model)  # Check that the IR is well formed
            print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
            return

        # Half precision
        half = half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            self.model.half()

        # Get names and colors
        self.names = load_classes(opt.names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

        # Run inference
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        _ = self.model(img.half() if half else img.float()) if self.device.type != 'cpu' else None  # run once

    def detect(self,image0,opt=None):
        image = letterbox(image0, new_shape=416)[0]
        defect_class=''
        confidence=0
        # Convert
        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(self.device)
        image = image.half() if opt.half else image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        pred = self.model(image, augment=opt.augment)[0]  #三个尺度下的输出 cancat 到一起
            # to float
        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms,detect_th = opt.confidence)

            # Process detections

        # for i, det in enumerate(pred):  # detections for image i
        for i, det in enumerate(pred):  # detections for image i
            s, im0 =  '', image0
            s += '%gx%g ' % image.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], image0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string
                    defect_class = self.names[int(c)]


            # Write results画边界框和标签
            # list = []

                for *xyxy, conf, cls in det:
                        # if(self.names[int(cls)] is "good" and  conf<self.opt.confidence):


                        label = '%s %.2f' % (self.names[int(cls)], conf)    # 索引值对应的类别，置信度
                        xt,yt = plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)])
                        confidence = conf.item()
                        xy = xyxy[0].item()
        return im0,defect_class






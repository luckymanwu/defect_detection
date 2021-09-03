# # -*- coding: utf-8 -*-
# import argparse
# from collections import OrderedDict
# from models import *  # set ONNX_EXPORT in models.py
# from project.datasets import *
# from project.utils import *
# from Opt import Opt
# import cv2
# import numpy as np
# from PIL import Image
#
# area = 0
# def ostu(img):
#     area = 0
#     image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转灰度
#     blur = cv2.GaussianBlur(image,(5,5),0) # 阈值一定要设为 0 ！高斯模糊
#     # ret3,th3 = cv2.threshold(blur,30,255,cv2.THRESH_BINARY) # 二值化 0 = black ; 1 = white 255白色 0黑色
#     ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
#     height, width = th3.shape
#     for i in range(height):
#         for j in range(width):
#             if th3[i, j] == 255:
#                 area += 1
#     return area
#
# def detect(save_img=False,opt=None):
#     defect_dict={}
#     imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
#     out, weights, half, view_img, save_txt = opt.output, opt.weights, opt.half, opt.view_img, opt.save_txt
#     # webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
#     # img_path = source.split("/")
#     # img_name = img_path[len(img_path) - 1]
#     # Initialize
#     device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device) #using cpu
#     if bool(1-os.path.exists(out)):
#         os.makedirs(out)  # make new output folder
#     #     shutil.rmtree(out)  # delete output folder
#     # os.makedirs(out)  # make new output folder
#
#     # Initialize model，调用darknet模型
#     model = Darknet(opt.cfg, imgsz)
#
#     # Load weights
#     attempt_download(weights)
#     if weights.endswith('.pt'):  # pytorch format
#
#         model.load_state_dict(torch.load(weights, map_location=device)['model'])
#         # model.load_state_dict(torch.load(weights, map_location='cpu'))
#     else:  # darknet format
#         load_darknet_weights(model, weights)
#
#     # Second-stage classifier
#     classify = False
#     if classify:
#         modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
#         modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
#         modelc.to(device).eval()
#
#     # Eval mode
#     model.to(device).eval()
#
#     # Fuse Conv2d + BatchNorm2d layers
#     # model.fuse()
#
#     # Export mode
#     if ONNX_EXPORT:
#         model.fuse()
#         img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
#         f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
#         torch.onnx.export(model, img, f, verbose=False, opset_version=11,
#                           input_names=['images'], output_names=['classes', 'boxes'])
#
#         # Validate exported model
#         import onnx
#         model = onnx.load(f)  # Load the ONNX model
#         onnx.checker.check_model(model)  # Check that the IR is well formed
#         print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
#         return
#
#     # Half precision
#     half = half and device.type != 'cpu'  # half precision only supported on CUDA
#     if half:
#         model.half()
#
#     # Set Dataloader
#     vid_path, vid_writer = None, None
#     # if webcam:
#     #     view_img = True
#     #     torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
#     #      = LoadStreams(source, img_size=imgsz)
#     # else:
#     #     save_img = True
#         dataset = LoadImages(source, img_size=imgsz)
#
#
#     # Get names and colors
#     names = load_classes(opt.names)
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
#
#     # Run inference
#     t0 = time.time()
#     img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
#     _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
#     for path, img, im0s, vid_cap in dataset:
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#
#         # Inference
#         t1 = torch_utils.time_synchronized()
#         pred = model(img, augment=opt.augment)[0]  #三个尺度下的输出 cancat 到一起
#
#         t2 = torch_utils.time_synchronized()
#
#         # to float
#         if half:
#             pred = pred.float()
#
#         # Apply NMS
#         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
#                                    multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
#
#         # Apply Classifier
#         if classify:
#             pred = apply_classifier(pred, modelc, img, im0s)
#
#         # Process detections
#         for i, det in enumerate(pred):  # detections for image i
#             if webcam:  # batch_size >= 1
#                 p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
#             else:
#                 p, s, im0 = path, '', im0s
#
#
#             save_path = str(Path(out) / Path(p).name)
#             s += '%gx%g ' % img.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
#             if det is not None and len(det):
#                 # Rescale boxes from imgsz to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#
#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += '%g %ss, ' % (n, names[int(c)])  # add to string
#                     defect_dict[img_name]= names[int(c)]
#
#
#
#
#                 # Write results画边界框和标签
#                 # list = []
#                 for *xyxy, conf, cls in det:
#
#                     if save_txt:  # Write to file，xyxy2xywh在project的utils中
#                         print("======>",xyxy2xywh(torch.tensor(xyxy).view(1, 4)))
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) ).view(-1).tolist()  # normalized xywh/ gn
#                         with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
#                             # file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
#                             file.write(('%s %s %g %g %g %g ' + '\n') % (img_name, defect_dict[img_name], *xywh))
#
#                             #  file.write(img_name, defect_dict[img_name], *xywh)
#
#                     if save_img or view_img:  # Add bbox to image
#                         label = '%s %.2f' % (names[int(cls)], conf)    # 索引值对应的类别，置信度
#                         xt,yt = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
#                         # list.append([xt,yt])
#                 # print("list=",list)
#                 # vector1 = np.array(list[0])
#                 # vector2 = np.array(list[1])
#                 # op1 = np.sqrt(np.sum(np.square(vector1 - vector2)))
#                 # return op1
#
#
#             # Print time (inference + NMS)
#             print('%sDone. (%.3fs)' % (s, t2 - t1))
#
#             # Stream results
#             if view_img:
#                 cv2.imshow(p, im0)
#                 if cv2.waitKey(1) == ord('q'):  # q to quit
#                     raise StopIteration
#
#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == 'images':
#                     cv2.imwrite(save_path, im0)
#                 else:
#                     if vid_path != save_path:  # new video
#                         vid_path = save_path
#                         if isinstance(vid_writer, cv2.VideoWriter):
#                           vid_writer.release()  # release previous video writer
#
#                         fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
#                     vid_writer.write(im0)
#         if not defect_dict:
#             defect_dict[img_name] = ""
#
#     if save_txt or save_img:
#         print('Results saved to %s' % os.getcwd() + os.sep + out)
#         # if platform == 'darwin':  # MacOS
#         #     # os.system('open ' + save_path)
#
#     print('Done. (%.3fs)' % (time.time() - t0))
#     return defect_dict;
#
#
# def startDetect(weights='../weights/best.pt',names='../data/rbc.names',cfg='cfg/yolov3-tiny.cfg'):
#     opt = Opt()
#     opt.cfg=cfg
#     opt.weights=weights
#     opt.names=names
#     opt.cfg = list(glob.iglob('../**/' + opt.cfg, recursive=True))[0]  # find file
#     opt.names = list(glob.iglob('../**/' + opt.names, recursive=True))[0]  # find file
#
#     with torch.no_grad():
#         defect_type=detect(False,opt)
#         return defect_type

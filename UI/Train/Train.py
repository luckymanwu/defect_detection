import os.path

import cv2
import numpy as np
import win32con
from PyQt5 import QtGui
from PyQt5.QtChart import QValueAxis, QChart, QSplineSeries
from PyQt5.QtCore import QPointF, QThread, Qt
# -*- coding: utf-8 -*-
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QBrush, QFont
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from Service import test
from Service.Serial import COM, com
from Service.hkDetect import hkDetect
from Service.models import *
from model.Opt import Opt
from project.datasets import *
from project.utils import *
from UI.Train.DataSet import DataSet
from UI.Train.TrainWin1 import TrainWin
from model.trainOpt import trainOpt
from model.Camera import Camera
from utils.CommonHelper import CommonHelper
from  MvCameraControl_class import *
import threading
from win32process import SuspendThread, ResumeThread
winfun_ctype = WINFUNCTYPE
device = None
tb_writer = None
# Hyperparameters 超参数
hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])
class Train(TrainWin):
    stop_thread = pyqtSignal()
    def __init__(self,configuration=None):
        super(Train, self).__init__()
        self.setupUi(self)
        styleFile = '../../resource/Train.qss'
        # 换肤时进行全局修改，只需要修改不同的QSS文件即可
        style = CommonHelper.readQss(styleFile)
        self.setStyleSheet(style)
        self.configuration = configuration
        self.cams = {}
        self.num = 0
        self.idx = 0
        self.progress = 0
        self.chartViewInit( self.configuration.value('TRAIN_MODEL'))
        self.cameras = {"1": self.carmer1, "2": self.carmer2,"3": self.carmer3, "4": self.carmer4}
        self.detect_threshold = 4000
        self.dataSetDialog = DataSet(dataSetPath = self.configuration.value('SAVE_DATASET_PATH'))
        self.start_sample.clicked.connect(self.startSample)
        self.stop_sample.clicked.connect(self.stopSample)
        self.generate_dataSet.clicked.connect(lambda: self.dataSetDialog.show())
        self.start_train.clicked.connect(self.startTrain)
        self.resume_train.clicked.connect(self.resume)
        self.suspend_train.clicked.connect(self.suspend)
        self.stop_train.clicked.connect(self.stop)
        self.stop_train.setEnabled(False)
        self.resume_train.setEnabled(False)
        self.suspend_train.setEnabled(False)



    def chartViewInit(self,title):
        self.chart = QChart()
        self.chart.setTitleBrush(QBrush(Qt.black))
        self.chart.setTitle("训练模式: "+str(title))
        self.series = QSplineSeries()
        self.series.setName("Loss")
        self.series.append(QPointF(0, 0))
        self.chart.addSeries(self.series)
        self.chart.axisY = QValueAxis()
        self.chart.axisY.setRange(0, 1)
        self.chart.axisY.setTickCount(11)
        self.chart.axisX = QValueAxis()
        self.chart.axisX.setRange(0, 100)
        self.chart.axisX.setTickCount(21)
        self.chart.addAxis(self.chart.axisY, Qt.AlignLeft)
        self.chart.addAxis(self.chart.axisX, Qt.AlignBottom)
        self.series.attachAxis(self.chart.axisY)
        self.series.attachAxis(self.chart.axisX)
        self.chartView.setChart(self.chart)


    def startSample(self):
        self.tabWidget.setCurrentIndex(0)
        speed = com.get_data(50)
        # timeout = int(75.0 / float(speed))
        for camNo in self.configuration.value('CAM_LIST'):
            cam = Camera(camNo, Opt(), 2)
            cam.sample_signal.connect(self.image_save)
            self.cams.update({camNo: cam})
            cam.openCam()
        self.start_sample.setEnabled(False)

    def stopSample(self):
         self.num = 0
         self.sample_num_label.clear()

         for camNo in self.configuration.value('CAM_LIST'):
              self.cams[camNo].g_bExit = True
              self.cams[camNo].closeCam()
         # self.com.close()
         self.start_sample.setEnabled(True)
    def startTrain(self):
        self.tabWidget.setCurrentIndex(1)
        self.series.clear()
        self.train_model = self.configuration.value("TRAIN_MODEL")
        if(self.train_model == "normal"):
            self.currentThread = TrainThread()
        else:
            self.currentThread = SemiTrainThread()
        self.currentThread.chartViewSignal.connect(self.update)
        self.currentThread.progessSignal.connect(self.update_progress)
        self.currentThread.trainEndSignal.connect(lambda: self.start_train.setEnabled(True))
        self.epochs = self.configuration.value('EPOCH')
        batch_size = self.configuration.value('BATCHSIZE')
        weights = self.configuration.value('WEIGHTS_PATH')
        cfg = self.configuration.value('CFG_PATH')
        dataSet_path = self.configuration.value('SAVE_DATASET_PATH')
        wdir = self.configuration.value('MODEL_SAVE_PATH')
        opt = trainOpt()
        detect_opt = Opt()
        detect_opt.cfg = cfg
        detect_opt.output = dataSet_path
        detect_opt.weights = weights
        opt.epochs = int(self.epochs)
        opt.batch_size = int(batch_size)
        opt.weights = weights
        opt.cfg = cfg
        opt.wdir = wdir
        opt.data = dataSet_path + '/rbc.data'
        opt.names = dataSet_path + '/rbc.names'
        if self.train_model == "normal":
            self.currentThread.opt = opt
            self.currentThread.start()
        else:
            self.currentThread.opt = opt
            self.currentThread.detect_opt = detect_opt
            self.currentThread.start()
        self.start_train.setEnabled(False)
        self.stop_train.setEnabled(True)
        self.resume_train.setEnabled(True)
        self.suspend_train.setEnabled(True)

    def resume(self):
        ret = ResumeThread( self.currentThread.handle)
        print('恢复线程',  self.currentThread.handle, ret)
        self.suspend_train.setEnabled(True)
        self.resume_train.setEnabled(False)

    def suspend(self):
        if   self.currentThread.handle == -1:
            return print('handle is wrong')
        ret = SuspendThread(  self.currentThread.handle)
        print('挂起线程',   self.currentThread.handle, ret)
        self.resume_train.setEnabled(True)
        self.suspend_train.setEnabled(False)

    def stop(self):
        self.currentThread.stop_flag=True
        self.start_train.setEnabled(True)
        self.stop_train.setEnabled(False)
        self.resume_train.setEnabled(False)
        self.suspend_train.setEnabled(False)

    def update(self,loss):
        self.idx+=1
        self.series.append(QPointF(self.idx,loss))
        if self.idx > 100:
            self.chart.scroll(4, 0)

    def update_progress(self,epochs):
        progressValue = epochs/float(self.epochs)*100
        print(progressValue)
        self.progressBar.setValue(progressValue)
        # self.progressBar.setProperty("value", progressValue)

    def image_save(self,image,camNo):
        img_path = self.configuration.value("SAVE_IMG_PATH")
        if(self.num == 0):
            path_file_number = glob.glob(img_path+'/*.jpg')
            self.file_num = len(path_file_number)
        print(self.file_num)
        self.num+=1
        cv2.imwrite(img_path + os.path.sep + str(self.num + self.file_num) + ".jpg", image)
        image = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                                    QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.cameras[camNo].setPixmap(QtGui.QPixmap.fromImage(image))
        self.sample_num_label.setText("采样数："+str(self.num))


class SemiTrainThread(QThread):

    trainEndSignal = pyqtSignal(object)
    chartViewSignal = pyqtSignal(object)
    progessSignal = pyqtSignal(object)
    def __init__(self):
        super(SemiTrainThread, self).__init__()
        self.opt = None
        self.detect_opt = None
        self.stop_flag = False

    handle = -1
    def run(self):
        try:
            # 这个目前我没弄明白这里写法
            self.handle = ctypes.windll.kernel32.OpenThread(  # @UndefinedVariable
                win32con.PROCESS_ALL_ACCESS, False, int(QThread.currentThreadId()))

        except Exception as e:
            print('get thread handle failed', e)
        print('thread id', int(QThread.currentThreadId()))

        opt = self.opt
        wdir = opt.wdir
        last = opt.wdir + '/last.pt'
        best = opt.wdir + '/best.pt'
        results_file = 'results.txt'
        mixed_precision = False
        check_git_status()
        # opt.data = list(glob.iglob('./**/' + opt.data, recursive=True))[0]  # find file
        opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
        device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
        if device.type == 'cpu':
            mixed_precision = False
        # scale hyp['obj'] by img_size (evolved at 320)
        # hyp['obj'] *= opt.img_size[0] / 320.
        if not opt.evolve:  # Train normally
            print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
            tb_writer = SummaryWriter(comment=opt.name)
        else:  # Evolve hyperparameters (optional)
            opt.notest, opt.nosave = True, True  # only test/save final epoch
            if opt.bucket:
                os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

            for _ in range(1):  # generations to evolve
                if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                    # Select parent(s)
                    parent = 'single'  # parent selection method: 'single' or 'weighted'
                    x = np.loadtxt('evolve.txt', ndmin=2)
                    n = min(5, len(x))  # number of previous results to consider
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min()  # weights
                    if parent == 'single' or len(x) == 1:
                        # x = x[random.randint(0, n - 1)]  # random selection
                        x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                    elif parent == 'weighted':
                        x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                    # Mutate
                    method, mp, s = 3, 0.9, 0.2  # method, mutation probability, sigma
                    npr = np.random
                    npr.seed(int(time.time()))
                    g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                    ng = len(g)
                    if method == 1:
                        v = (npr.randn(ng) * npr.random() * g * s + 1) ** 2.0
                    elif method == 2:
                        v = (npr.randn(ng) * npr.random(ng) * g * s + 1) ** 2.0
                    elif method == 3:
                        v = np.ones(ng)
                        while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                            # v = (g * (npr.random(ng) < mp) * npr.randn(ng) * s + 1) ** 2.0
                            v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                    for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                        hyp[k] = x[i + 7] * v[i]  # mutate

                # Clip to limits
                keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
                limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9),
                          (0, 3)]
                for k, v in zip(keys, limits):
                    hyp[k] = np.clip(hyp[k], v[0], v[1])

                # Train mutation
                results = train(hyp.copy())

                # Write mutation results
                print_mutation(hyp, results, opt.bucket)

                # Plot results
                # plot_evolution_results(hyp)
        cfg = opt.cfg
        data = opt.data
        epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
        batch_size = opt.batch_size
        accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
        weights = opt.weights  # initial training weights
        imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)

        # Image Sizes
        gs = 64  # (pixels) grid size
        assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
        opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
        if opt.multi_scale:
            if imgsz_min == imgsz_max:
                imgsz_min //= 1.5
                imgsz_max //= 0.667
            grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
            imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        img_size = imgsz_max  # initialize with max size

        # Configure run
        init_seeds()
        data_dict = parse_data_cfg(data)
        name_dict = parse_rbc_name(opt.names)
        train_label_path = data_dict['label_train']
        train_Unlabel_path = data_dict['unlabel_train']
        # train_path= data_dict['train']
        test_path = data_dict['valid']
        nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
        hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

        # Remove previous results
        for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg).to(device)

        # Optimizer
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(model.named_parameters()).items():
            if '.bias' in k:
                pg2 += [v]  # biases
            elif 'Conv2d.weight' in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else

        if opt.adam:
            # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
            optimizer = optim.Adam(pg0, lr=hyp['lr0'])
            # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
        else:
            optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
        optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        start_epoch = 0
        best_fitness = 0.0
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
            chkpt = torch.load(weights, map_location=device)

            # load model
            try:
                # chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(torch.load(weights, map_location=device)['model'])
                # model.load_state_dict(chkpt['model'], strict=False)
            except KeyError as e:
                s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. See https://github.com/ultralytics/yolov3/issues/657" % (
                    opt.weights, opt.cfg, opt.weights)
                raise KeyError(s)  # from e

            # load optimizer
            if chkpt['optimizer'] is not None:
                optimizer.load_state_dict(chkpt['optimizer'])
                best_fitness = chkpt['best_fitness']

            # load results
            if chkpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(chkpt['training_results'])  # write results.txt

            start_epoch = chkpt['epoch'] + 1
            del chkpt

        elif len(weights) > 0:  # darknet format
            # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
            load_darknet_weights(model, weights)

        # Mixed precision training https://github.com/NVIDIA/apex
        if mixed_precision:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scheduler.last_epoch = start_epoch - 1  # see link below
        # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822

        # Plot lr schedule
        # y = []
        # for _ in range(epochs):
        #     scheduler.step()
        #     y.append(optimizer.param_groups[0]['lr'])
        # plt.plot(y, '.-', label='LambdaLR')
        # plt.xlabel('epoch')
        # plt.ylabel('LR')
        # plt.tight_layout()
        # plt.savefig('LR.png', dpi=300)

        # Initialize distributed training
        if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
            dist.init_process_group(backend='nccl',  # 'distributed backend'
                                    init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                    world_size=1,  # number of nodes for distributed training
                                    rank=0)  # distributed training node rank
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

        label_file = readData(train_label_path)
        unLabel_file = readData(train_Unlabel_path)

        # Dataset
        dataset = LoadImagesAndLabels(train_label_path, img_size, batch_size,
                                      augment=True,
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=opt.rect,  # rectangular training
                                      cache_images=opt.cache_images,
                                      single_cls=opt.single_cls)

        # Dataloader
        batch_size = min(batch_size, len(dataset))

        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=4,
                                                 shuffle=not opt.rect,
                                                 # Shuffle=True unless rectangular training is used
                                                 pin_memory=True,
                                                 collate_fn=dataset.collate_fn)

        # Testloader
        testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                     hyp=hyp,
                                                                     rect=True,
                                                                     cache_images=opt.cache_images,
                                                                     single_cls=opt.single_cls),
                                                 batch_size=batch_size,
                                                 num_workers=4,
                                                 pin_memory=True,
                                                 collate_fn=dataset.collate_fn)

        # Model parameters
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights

        # Model EMA
        ema = torch_utils.ModelEMA(model)

        # Start training
        nb = len(dataloader)  # number of batches
        n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
        maps = np.zeros(nc)  # mAP per class
        # torch.autograd.set_detect_anomaly(True)
        results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
        t0 = time.time()
        print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
        print('Using %g dataloader workers' % nw)
        print('Starting training for %g epochs...' % epochs)
        th = 0.8
        train_flag = True
        train_epoch = start_epoch
        while train_flag:

            for epoch in range(train_epoch,
                               train_epoch+50):  # epoch ------------------------------------------------------------------

                self.progessSignal.emit(train_epoch)
                model.train()
                train_epoch += 1

                # Update image weights (optional)
                if dataset.image_weights:
                    w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                    image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                    dataset.indices = random.choices(range(dataset.n), weights=image_weights,
                                                     k=dataset.n)  # rand weighted idx

                mloss = torch.zeros(4).to(device)  # mean losses
                print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
                pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
                for i, (
                        imgs, targets, paths,
                        _) in pbar:  # batch -------------------------------------------------------------
                    if(self.stop_flag):
                        return
                    ni = i + nb * epoch  # number integrated batches (since train start)
                    imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
                    targets = targets.to(device)

                    # Burn-in
                    if ni <= n_burn:
                        xi = [0, n_burn]  # x interp
                        model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                        accumulate = max(1, np.interp(ni, xi, [1, 64 / batch_size]).round())
                        for j, x in enumerate(optimizer.param_groups):
                            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                            x['weight_decay'] = np.interp(ni, xi, [0.0, hyp['weight_decay'] if j == 1 else 0.0])
                            if 'momentum' in x:
                                x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

                    # Multi-Scale
                    if opt.multi_scale:
                        if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                            img_size = random.randrange(grid_min, grid_max + 1) * gs
                        sf = img_size / max(imgs.shape[2:])  # scale factor
                        if sf != 1:
                            ns = [math.ceil(x * sf / gs) * gs for x in
                                  imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                            imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    # Forward
                    pred = model(imgs)

                    # Loss
                    loss, loss_items = compute_loss(pred, targets, model)
                    if not torch.isfinite(loss):
                        print('WARNING: non-finite loss, ending training ', loss_items)
                        return results

                    # Backward
                    loss *= batch_size / 64  # scale loss
                    if mixed_precision:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    # Optimize
                    if ni % accumulate == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        ema.update(model)

                    self.chartViewSignal.emit(loss.item())

                    # Print
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
                    pbar.set_description(s)

                    # Plot
                    if ni < 1:
                        f = 'train_batch%g.jpg' % i  # filename
                        res = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                        if tb_writer:
                            tb_writer.add_image(f, res, dataformats='HWC', global_step=epoch)
                            # tb_writer.add_graph(model, imgs)  # add model to tensorboard

                    # end batch ------------------------------------------------------------------------------------------------

                # Update scheduler
                scheduler.step()

                # Process epoch results
                ema.update_attr(model)
                final_epoch = epoch + 1 == epochs
                if not opt.notest or final_epoch:  # Calculate mAP
                    is_coco = any(
                        [x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
                    results, maps = test.test(cfg,
                                              data,
                                              batch_size=batch_size,
                                              imgsz=imgsz_test,
                                              model=ema.ema,
                                              save_json=final_epoch and is_coco,
                                              single_cls=opt.single_cls,
                                              dataloader=testloader,
                                              multi_label=ni > n_burn)

                # Write
                with open(results_file, 'a') as f:
                    f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
                if len(opt.name) and opt.bucket:
                    os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

                # Tensorboard
                if tb_writer:
                    tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                            'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                    for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                        tb_writer.add_scalar(tag, x, epoch)

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
                if fi > best_fitness:
                    best_fitness = fi

                # Save model
                save = (not opt.nosave) or (final_epoch and not opt.evolve)
                if save:
                    with open(results_file, 'r') as f:  # create checkpoint
                        chkpt = {'epoch': epoch,
                                 'best_fitness': best_fitness,
                                 'training_results': f.read(),
                                 'model': ema.ema.module.state_dict() if hasattr(model,
                                                                                 'module') else ema.ema.state_dict(),
                                 'optimizer': None if final_epoch else optimizer.state_dict()}

                    # Save last, best and delete
                    torch.save(chkpt, last)
                    if (best_fitness == fi) and not final_epoch:
                        torch.save(chkpt, best)
                    del chkpt
            # end epoch ----------------------------------------------------------------------------------------------------
            model.to(device).eval()
            unLabel_file_num = len(unLabel_file)
            for path in unLabel_file:
                image0 = cv2.imread(path)  # BGR
                image = letterbox(image0, new_shape=416)[0]
                image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                image = np.ascontiguousarray(image)
                image = torch.from_numpy(image).to(device)
                image = image.half() if self.detect_opt.half else image.float()  # uint8 to fp16/32
                image /= 255.0  # 0 - 255 to 0.0 - 1.0
                if image.ndimension() == 3:
                    image = image.unsqueeze(0)

                pred = model(image, augment=self.detect_opt.augment)[0]  # 三个尺度下的输出 cancat 到一起
                # to float
                if self.detect_opt.half:
                    pred = pred.float()
                # Apply NMS
                pred = non_max_suppression(pred, self.detect_opt.conf_thres, self.detect_opt.iou_thres,
                                           multi_label=False, classes=self.detect_opt.classes,
                                           agnostic=self.detect_opt.agnostic_nms)

                # Process detections
                # for i, det in enumerate(pred):  # detections for image i
                for i, det in enumerate(pred):  # detections for image i

                    if det is not None and len(det):
                        # Rescale boxes from imgsz to im0 size
                        det[:, :4] = scale_coords(image.shape[2:], det[:, :4], image.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            # s += '%g %ss, ' % (n, name_dict[int(c)])  # add to string
                            defect_class_id = int(c)
                        for *xyxy, conf, cls in det:
                            confidence = conf.item()
                            print(confidence)

                if (det is not None ):
                    dataset.imgs.append(None)
                    dataset.img_files.append(path)
                    dataset.n += 1
                    newlabel = np.array(
                        [[defect_class_id, xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]],
                        dtype=np.float32)
                    dataset.labels.append(newlabel)
                    unLabel_file.remove(path)
            if (len(unLabel_file) == unLabel_file_num or len(unLabel_file) ==0):
                train_flag = False

        # end training

        n = opt.name
        if len(n):
            n = '_' + n if not n.isnumeric() else n
            fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
            for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
                if os.path.exists(f1):
                    os.rename(f1, f2)  # rename
                    ispt = f2.endswith('.pt')  # is *.pt
                    strip_optimizer(f2) if ispt else None  # strip optimizer
                    os.system(
                        'gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

        if not opt.evolve:
            plot_results()  # save as results.png
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
        torch.cuda.empty_cache()
        self.trainEndSignal.emit()

        # return results

class TrainThread(QThread):
    handle = -1
    chartViewSignal = pyqtSignal(object)
    progessSignal = pyqtSignal(object)
    trainEndSignal = pyqtSignal()
    def __init__(self):
        super(TrainThread, self).__init__()
        self.opt = None
        self.stop_flag = False
    def run(self):

        try:
            # 这个目前我没弄明白这里写法
            self.handle = ctypes.windll.kernel32.OpenThread(  # @UndefinedVariable
                win32con.PROCESS_ALL_ACCESS, False, int(QThread.currentThreadId()))
        except Exception as e:
            print('get thread handle failed', e)

        opt = self.opt
        wdir = opt.wdir
        last = opt.wdir + '/last.pt'
        best = opt.wdir + '/best.pt'
        results_file = 'results.txt'
        mixed_precision = False
        check_git_status()
        # opt.data = list(glob.iglob('./**/' + opt.data, recursive=True))[0]  # find file
        opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
        device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
        if device.type == 'cpu':
            mixed_precision = False
        # scale hyp['obj'] by img_size (evolved at 320)
        # hyp['obj'] *= opt.img_size[0] / 320.
        if not opt.evolve:  # Train normally
            print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
            tb_writer = SummaryWriter(comment=opt.name)
        else:  # Evolve hyperparameters (optional)
            opt.notest, opt.nosave = True, True  # only test/save final epoch
            if opt.bucket:
                os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

            for _ in range(1):  # generations to evolve
                if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                    # Select parent(s)
                    parent = 'single'  # parent selection method: 'single' or 'weighted'
                    x = np.loadtxt('evolve.txt', ndmin=2)
                    n = min(5, len(x))  # number of previous results to consider
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min()  # weights
                    if parent == 'single' or len(x) == 1:
                        # x = x[random.randint(0, n - 1)]  # random selection
                        x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                    elif parent == 'weighted':
                        x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                    # Mutate
                    method, mp, s = 3, 0.9, 0.2  # method, mutation probability, sigma
                    npr = np.random
                    npr.seed(int(time.time()))
                    g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                    ng = len(g)
                    if method == 1:
                        v = (npr.randn(ng) * npr.random() * g * s + 1) ** 2.0
                    elif method == 2:
                        v = (npr.randn(ng) * npr.random(ng) * g * s + 1) ** 2.0
                    elif method == 3:
                        v = np.ones(ng)
                        while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                            # v = (g * (npr.random(ng) < mp) * npr.randn(ng) * s + 1) ** 2.0
                            v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                    for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                        hyp[k] = x[i + 7] * v[i]  # mutate

                # Clip to limits
                keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
                limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9),
                          (0, 3)]
                for k, v in zip(keys, limits):
                    hyp[k] = np.clip(hyp[k], v[0], v[1])

                # Train mutation
                results = train(hyp.copy())

                # Write mutation results
                print_mutation(hyp, results, opt.bucket)

                # Plot results
                # plot_evolution_results(hyp)
        cfg = opt.cfg
        data = opt.data
        epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
        batch_size = opt.batch_size
        accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
        weights = opt.weights  # initial training weights
        imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)

        # Image Sizes
        gs = 64  # (pixels) grid size
        assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
        opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
        if opt.multi_scale:
            if imgsz_min == imgsz_max:
                imgsz_min //= 1.5
                imgsz_max //= 0.667
            grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
            imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        img_size = imgsz_max  # initialize with max size

        # Configure run
        init_seeds()
        data_dict = parse_data_cfg(data)
        name_dict = parse_rbc_name(opt.names)
        train_label_path = data_dict['label_train']
        train_Unlabel_path = data_dict['unlabel_train']
        # train_path= data_dict['train']
        test_path = data_dict['valid']
        nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
        hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

        # Remove previous results
        for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg).to(device)

        # Optimizer
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(model.named_parameters()).items():
            if '.bias' in k:
                pg2 += [v]  # biases
            elif 'Conv2d.weight' in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else

        if opt.adam:
            # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
            optimizer = optim.Adam(pg0, lr=hyp['lr0'])
            # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
        else:
            optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
        optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        start_epoch = 0
        best_fitness = 0.0
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
            chkpt = torch.load(weights, map_location=device)

            # load model
            try:
                chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
                # model.load_state_dict(torch.load(weights, map_location=device)['model'])
                model.load_state_dict(chkpt['model'], strict=False)
            except KeyError as e:
                s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. See https://github.com/ultralytics/yolov3/issues/657" % (
                    opt.weights, opt.cfg, opt.weights)
                raise KeyError(s)  # from e

            # load optimizer
            if chkpt['optimizer'] is not None:
                optimizer.load_state_dict(chkpt['optimizer'])
                best_fitness = chkpt['best_fitness']

            # load results
            if chkpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(chkpt['training_results'])  # write results.txt

            start_epoch = chkpt['epoch'] + 1
            del chkpt

        elif len(weights) > 0:  # darknet format
            # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
            load_darknet_weights(model, weights)

        # Mixed precision training https://github.com/NVIDIA/apex
        if mixed_precision:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scheduler.last_epoch = start_epoch - 1  # see link below
        # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822

        # Plot lr schedule
        # y = []
        # for _ in range(epochs):
        #     scheduler.step()
        #     y.append(optimizer.param_groups[0]['lr'])
        # plt.plot(y, '.-', label='LambdaLR')
        # plt.xlabel('epoch')
        # plt.ylabel('LR')
        # plt.tight_layout()
        # plt.savefig('LR.png', dpi=300)

        # Initialize distributed training
        if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
            dist.init_process_group(backend='nccl',  # 'distributed backend'
                                    init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                    world_size=1,  # number of nodes for distributed training
                                    rank=0)  # distributed training node rank
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

        label_file = readData(train_label_path)
        unLabel_file = readData(train_Unlabel_path)

        # Dataset
        dataset = LoadImagesAndLabels(train_label_path, img_size, batch_size,
                                      augment=True,
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=opt.rect,  # rectangular training
                                      cache_images=opt.cache_images,
                                      single_cls=opt.single_cls)

        # Dataloader
        batch_size = min(batch_size, len(dataset))
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=4,
                                                 shuffle=not opt.rect,
                                                 # Shuffle=True unless rectangular training is used
                                                 pin_memory=True,
                                                 collate_fn=dataset.collate_fn,
                                              )

        # Testloader
        testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                     hyp=hyp,
                                                                     rect=True,
                                                                     cache_images=opt.cache_images,
                                                                     single_cls=opt.single_cls),
                                                 batch_size=batch_size,
                                                 num_workers=4,
                                                 pin_memory=True,
                                                 collate_fn=dataset.collate_fn )

        # Model parameters
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights

        # Model EMA
        ema = torch_utils.ModelEMA(model)

        # Start training
        nb = len(dataloader)  # number of batches
        n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
        maps = np.zeros(nc)  # mAP per class
        # torch.autograd.set_detect_anomaly(True)
        results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
        t0 = time.time()
        print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
        print('Using %g dataloader workers' % nw)
        print('Starting training for %g epochs...' % epochs)

        for epoch in range(start_epoch,
                           opt.epochs):  # epoch ------------------------------------------------------------------
            self.progessSignal.emit(epoch)
            model.train()


            # Update image weights (optional)
            if dataset.image_weights:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                dataset.indices = random.choices(range(dataset.n), weights=image_weights,
                                                 k=dataset.n)  # rand weighted idx

            mloss = torch.zeros(4).to(device)  # mean losses
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
            for i, (
                    imgs, targets, paths,
                    _) in pbar:  # batch -------------------------------------------------------------
                if(self.stop_flag):
                    return
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
                targets = targets.to(device)

                # Burn-in
                if ni <= n_burn:
                    xi = [0, n_burn]  # x interp
                    model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                    accumulate = max(1, np.interp(ni, xi, [1, 64 / batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        x['weight_decay'] = np.interp(ni, xi, [0.0, hyp['weight_decay'] if j == 1 else 0.0])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

                # Multi-Scale
                if opt.multi_scale:
                    if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                        img_size = random.randrange(grid_min, grid_max + 1) * gs
                    sf = img_size / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in
                              imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                # Forward
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                pred = model(imgs)

                # Loss
                loss, loss_items = compute_loss(pred, targets, model)
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results

                # Backward
                loss *= batch_size / 64  # scale loss
                if mixed_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Optimize
                if ni % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    ema.update(model)

                self.chartViewSignal.emit(loss.item())

                # Print
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
                pbar.set_description(s)

                # Plot
                if ni < 1:
                    f = 'train_batch%g.jpg' % i  # filename
                    res = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    if tb_writer:
                        tb_writer.add_image(f, res, dataformats='HWC', global_step=epoch)
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard

                # end batch ------------------------------------------------------------------------------------------------

            # Update scheduler
            scheduler.step()

            # Process epoch results
            ema.update_attr(model)
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
                results, maps = test.test(cfg,
                                          data,
                                          batch_size=batch_size,
                                          imgsz=imgsz_test,
                                          model=ema.ema,
                                          save_json=final_epoch and is_coco,
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          multi_label=ni > n_burn)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

            # Tensorboard
            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                        'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, 'r') as f:  # create checkpoint
                    chkpt = {'epoch': epoch,
                             'best_fitness': best_fitness,
                             'training_results': f.read(),
                             'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
                             'optimizer': None if final_epoch else optimizer.state_dict()}

                # Save last, best and delete
                torch.save(chkpt, last)
                if (best_fitness == fi) and not final_epoch:
                    torch.save(chkpt, best)
                del chkpt
            # end epoch ----------------------------------------------------------------------------------------------------

    # end training
        n = opt.name
        if len(n):
            n = '_' + n if not n.isnumeric() else n
            fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
            for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
                if os.path.exists(f1):
                    os.rename(f1, f2)  # rename
                    ispt = f2.endswith('.pt')  # is *.pt
                    strip_optimizer(f2) if ispt else None  # strip optimizer
                    os.system(
                        'gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

        if not opt.evolve:
            plot_results()  # save as results.png
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
        torch.cuda.empty_cache()
        self.trainEndSignal.emit()
        # return results


def readData(path):
        path = str(Path(path))  # os-agnostic
        parent = str(Path(path).parent) + os.sep
        if os.path.isfile(path):  # file
            with open(path, 'r') as f:
                f = f.read().splitlines()
                f = [x.replace('./', parent) if x.startswith('./') else x for x in f]  # local to global path
            return f

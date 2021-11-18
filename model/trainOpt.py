class trainOpt(object):
    def __init__(self):
        super(trainOpt, self).__init__()
        self.adam = False
        self.batch_size = 16
        self.bucket=''
        self.cache_images=False
        self.cfg=None
        self.device=''
        self.epochs = 1000
        self.evolve=False
        self.img_size = [320, 640]
        self.multi_scale=False
        self.name = ''
        self.names=''
        self.nosave = False
        self.notest=False
        self.rect = False
        self.resume=False
        self.single_cls=False
        self.weights=None
        self.data=None
        self.wdir=None

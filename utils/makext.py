import os
import random

def makext(imgpath,xmlfilepath,txtsavepath):
        trainval_percent = 0.2
        train_percent = 0.8
        total_xml = os.listdir(xmlfilepath)
        imgs = os.listdir(imgpath)
        num = len(total_xml)
        list = range(num)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(list, tv)
        train = random.sample(trainval, tr)
        funlabelTrain = open(txtsavepath+'/unlabelTrain.txt','w')
        ftrainval = open(txtsavepath + '/trainval.txt', 'w')
        ftest = open(txtsavepath +'/test.txt', 'w')
        ftrain = open(txtsavepath +'/labeltrain.txt', 'w')
        fval = open(txtsavepath +'/val.txt', 'w')
        for i in range(len(imgs)):
            name = imgs[i][:-4]
            xml_name = name +'.xml'
            if xml_name not in total_xml:
                funlabelTrain.write(name+'\n')
        for i in list:
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftest.write(name)
                else:
                    fval.write(name)
            else:
                ftrain.write(name)
        ftrainval.close()
        funlabelTrain.close()
        ftrain.close()
        fval.close()
        ftest.close()

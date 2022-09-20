import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
sets = ['labeltrain','unlabelTrain','test','val']
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(dataset_path,image_id,classes):
    in_file = open(dataset_path+'/Annotations/%s.xml' % (image_id))
    out_file = open(dataset_path+'/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def vocLabel(classes,dataSet):
    for image_set in sets:
        if not os.path.exists(dataSet+'/labels'):
            os.makedirs(dataSet+'/labels')
        image_ids = open(dataSet+'/ImageSets/%s.txt' % (image_set)).read().strip().split()
        list_file = open(dataSet+'/%s.txt' % (image_set), 'w')
        for image_id in image_ids:
            list_file.write(dataSet+'/images/%s.jpg\n' % (image_id))
            if image_set is not 'unlabelTrain':
                convert_annotation(dataSet,image_id,classes)
        list_file.close()

"""
parse PASCAL VOC xml annotations
"""

import os
import sys
import xml.etree.ElementTree as ET
import glob
import numpy as np
import PIL.Image

DATA_PATH = '/Users/liangchuangu/Development/machine_learning/tensorflow/models/research/object_detection/data/FullIJCNN2013/'

def _pp(l): # pretty printing
    for i in l: print('{}: {}'.format(i,l[i]))

def pascal_voc_clean_xml(ANN, pick, exclusive = False):
    print('Parsing for {} {}'.format(
            pick, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.xml')
    size = len(annotations)

    for i, file in enumerate(annotations):
        # progress bar
        sys.stdout.write('\r')
        percentage = 1. * (i+1) / size
        progress = int(percentage * 20)
        bar_arg = [progress*'=', ' '*(19-progress), percentage*100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()

        # actual parsing
        in_file = open(file)
        tree=ET.parse(in_file)
        root = tree.getroot()
        jpg = str(root.find('filename').text)
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        all = list()

        for obj in root.iter('object'):
                current = list()
                name = obj.find('name').text
                if name not in pick:
                        continue

                xmlbox = obj.find('bndbox')
                xn = int(float(xmlbox.find('xmin').text))
                xx = int(float(xmlbox.find('xmax').text))
                yn = int(float(xmlbox.find('ymin').text))
                yx = int(float(xmlbox.find('ymax').text))
                current = [name,xn,yn,xx,yx]
                all += [current]

        add = [[jpg, [w, h, all]]]
        dumps += add
        in_file.close()

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]]+=1
                else:
                    stat[current[0]] =1

    print('\nStatistics:')
    _pp(stat)
    print('Dataset size: {}'.format(len(dumps)))

    os.chdir(cur_dir)
    return dumps

def get_training_examples(ANN):
    examples_list = []
    # annotation_file = os.path.join(dir, "gt.txt")
    ground_truth = os.path.join(ANN, 'gt.txt')
    print(ground_truth)
    examples_list = examples_list + read_examples_list(ground_truth)
    #print(*examples_list, sep='\n')
    return examples_list

def read_examples_list(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        #print(*lines)
    return [line.strip().split(' ')[0] for line in lines]

def gtsdb_text(ANN, pick):
    # 1) Build a index to name mapping, because we need name in the dumps
    # 2) Get annotation lines
    #   * loop over lines and construct dumps
    #   * consider using a dict to store
    index_to_name_map = {}
    print('Parsing for {}'.format(pick))
    for i, name in enumerate(pick):
        index_to_name_map[i] = name
    print('index_to_name_map')
    #for key, val in index_to_name_map.items():
        #print(key, val)
    training_list = get_training_examples(ANN)
    train_len = len(training_list)
    print("There are " + str(train_len) + " labels\n")
    unique_set = set([])
    for idx, example in enumerate(training_list):
        unique_set.add(example.split(';')[0])
    print("There are " + str(len(unique_set)) + " images containing labels\n")

    image_to_ann_map = {}
    for idx, example in enumerate(training_list):
        #print("exmaple: " + example)
        training_data = example.split(';')
        anno_data = []
        if training_data[0] in image_to_ann_map:
            anno_data = image_to_ann_map[training_data[0]]
        else:
            image_name = training_data[0].split('.')[0] + '.jpg'
            s = np.array(PIL.Image.open(os.path.join(DATA_PATH, image_name)), dtype=np.uint8).shape
            anno_data = image_to_ann_map[training_data[0]] = [[image_name, [s[0], s[1], []]]]
        # Get class name from map
        name = index_to_name_map[int(training_data[5])]
        if name not in pick:
            continue
        # Add to total di,ps
        current = []
        current.append(name)
        current.append(int(training_data[1])) # xn
        current.append(int(training_data[2])) # yn
        current.append(int(training_data[3])) # xx
        current.append(int(training_data[4])) # yx
        anno_data[0][1][2].append(current)
        if idx % 100 == 0:
            print('On image {} of {}'.format(idx, train_len))
    dumps = []
    print('image_to_ann_map')
    for key, val in image_to_ann_map.items():
        print(key, val)
        dumps += image_to_ann_map[key]
        # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]]+=1
                else:
                    stat[current[0]] =1

    print('\nStatistics:')
    _pp(stat)
    print('Dataset size: {}'.format(len(dumps)))
    return dumps

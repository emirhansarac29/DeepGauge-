from __future__ import print_function

import os
import argparse
import random

from keras.datasets import mnist
from keras.layers import Input
from numpy import loadtxt
from keras import backend as K
from keras.models import Model
from scipy.misc import imsave
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import *
import threading
from configs import bcolors
import json

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3


def write_test_coverage(dir, NC, KM, NBC, SNA, TKNC, BKNC, TKNP, BKNP):
    f = open("adv/EH/" + dir + "/nc.txt", "a")
    for k, l in NC:
        f.write(k)
        f.write(" ")
        f.write(str(l))
        f.write(" ")
        f.write(str(NC[(k, l)]))
        f.write("\n")
    f.close()

    f = open("adv/EH/" + dir + "/km.txt", "a")
    for k, l in KM:
        f.write(k)
        f.write(" ")
        f.write(str(l))
        for v in KM[(k, l)]:
            f.write(" ")
            f.write(str(v))
        f.write("\n")
    f.close()

    f = open("adv/EH/" + dir + "/nbc.txt", "a")
    for k, l in NBC:
        f.write(k)
        f.write(" ")
        f.write(str(l))
        f.write(" ")
        f.write(str(NBC[(k, l)][0]))
        f.write(" ")
        f.write(str(NBC[(k, l)][1]))
        f.write("\n")
    f.close()
    f = open("adv/EH/" + dir + "/sna.txt", "a")
    for k, l in SNA:
        f.write(k)
        f.write(" ")
        f.write(str(l))
        f.write(" ")
        f.write(str(SNA[(k, l)]))
        f.write("\n")
    f.close()
    f = open("adv/EH/" + dir + "/tknc.txt", "a")
    for k, l in TKNC:
        f.write(k)
        f.write(" ")
        f.write(str(l))
        f.write(" ")
        f.write(str(TKNC[(k, l)]))
        f.write("\n")
    f.close()
    f = open("adv/EH/" + dir + "/bknc.txt", "a")
    for k, l in BKNC:
        f.write(k)
        f.write(" ")
        f.write(str(l))
        f.write(" ")
        f.write(str(BKNC[(k, l)]))
        f.write("\n")
    f.close()
    f = open("adv/EH/" + dir + "/tknp.txt", "a")
    for k in TKNP:
        f.write(k)
        f.write("\n")
    f.close()
    f = open("adv/EH/" + dir + "/bknp.txt", "a")
    for k in BKNP:
        f.write(k)
        f.write("\n")
    f.close()


def write_max_min(model_layer_dict1_low, model_layer_dict1_high, model_layer_dict2_low, model_layer_dict2_high,
                  model_layer_dict3_low, model_layer_dict3_high):
    f = open("HL/model1_low.txt", "a")
    for k, l in model_layer_dict1_low:
        f.write(k)
        f.write(" ")
        f.write(str(l))
        f.write(" ")
        f.write(str(model_layer_dict1_low[(k, l)]))
        f.write("\n")
    f.close()
    f = open("HL/model1_high.txt", "a")
    for k, l in model_layer_dict1_high:
        f.write(k)
        f.write(" ")
        f.write(str(l))
        f.write(" ")
        f.write(str(model_layer_dict1_high[(k, l)]))
        f.write("\n")
    f.close()
    f = open("HL/model2_low.txt", "a")
    for k, l in model_layer_dict2_low:
        f.write(k)
        f.write(" ")
        f.write(str(l))
        f.write(" ")
        f.write(str(model_layer_dict2_low[(k, l)]))
        f.write("\n")
    f.close()
    f = open("HL/model2_high.txt", "a")
    for k, l in model_layer_dict2_high:
        f.write(k)
        f.write(" ")
        f.write(str(l))
        f.write(" ")
        f.write(str(model_layer_dict2_high[(k, l)]))
        f.write("\n")
    f.close()
    f = open("HL/model3_low.txt", "a")
    for k, l in model_layer_dict3_low:
        f.write(k)
        f.write(" ")
        f.write(str(l))
        f.write(" ")
        f.write(str(model_layer_dict3_low[(k, l)]))
        f.write("\n")
    f.close()
    f = open("HL/model3_high.txt", "a")
    for k, l in model_layer_dict3_high:
        f.write(k)
        f.write(" ")
        f.write(str(l))
        f.write(" ")
        f.write(str(model_layer_dict3_high[(k, l)]))
        f.write("\n")
    f.close()


def import_max_min(model_layer_dict1_low, model_layer_dict1_high, model_layer_dict2_low, model_layer_dict2_high,
                   model_layer_dict3_low, model_layer_dict3_high):
    file1 = open('HL/model1_low.txt', 'r')
    lines = file1.readlines()
    for a in range(52):
        sp = lines[a].split(' ')
        model_layer_dict1_low[(sp[0], int(sp[1]))] = float(sp[2])
    file1.close()

    file1 = open('HL/model1_high.txt', 'r')
    lines = file1.readlines()
    for a in range(52):
        sp = lines[a].split(' ')
        model_layer_dict1_high[(sp[0], int(sp[1]))] = float(sp[2])
    file1.close()

    file1 = open('HL/model2_low.txt', 'r')
    lines = file1.readlines()
    for a in range(148):
        sp = lines[a].split(' ')
        model_layer_dict2_low[(sp[0], int(sp[1]))] = float(sp[2])
    file1.close()

    file1 = open('HL/model2_high.txt', 'r')
    lines = file1.readlines()
    for a in range(148):
        sp = lines[a].split(' ')
        model_layer_dict2_high[(sp[0], int(sp[1]))] = float(sp[2])
    file1.close()

    file1 = open('HL/model3_low.txt', 'r')
    lines = file1.readlines()
    for a in range(268):
        sp = lines[a].split(' ')
        model_layer_dict3_low[(sp[0], int(sp[1]))] = float(sp[2])
    file1.close()

    file1 = open('HL/model3_high.txt', 'r')
    lines = file1.readlines()
    for a in range(268):
        sp = lines[a].split(' ')
        model_layer_dict3_high[(sp[0], int(sp[1]))] = float(sp[2])
    file1.close()


def show_image(data):
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(data.reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()


os.environ['KERAS_BACKEND'] = 'tensorflow'
img_rows, img_cols = 28, 28
(TRAIN_FEAT, TRAIN_LABEL), (TEST_FEAT, TEST_LABEL) = mnist.load_data()

TEST_FEAT = TEST_FEAT.reshape(TEST_FEAT.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

TRAIN_FEAT = TRAIN_FEAT.reshape(TRAIN_FEAT.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

input_tensor = Input(shape=input_shape)

model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)

model1_layer_name_number = []
model2_layer_name_number = []
model3_layer_name_number = []
for layer in model1.layers:
    if 'flatten' in layer.name or 'input' in layer.name:
        continue
    model1_layer_name_number.append((layer.name, layer.output_shape[-1]))

for layer in model2.layers:
    if 'flatten' in layer.name or 'input' in layer.name:
        continue
    model2_layer_name_number.append((layer.name, layer.output_shape[-1]))

for layer in model3.layers:
    if 'flatten' in layer.name or 'input' in layer.name:
        continue
    model3_layer_name_number.append((layer.name, layer.output_shape[-1]))

NC_THRESHOLD = 0.75

model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)

model_layer_dict1_low, model_layer_dict1_high, model_layer_dict2_low, model_layer_dict2_high, model_layer_dict3_low, model_layer_dict3_high = init_kmnc_tables(
    model1, model2, model3)
import_max_min(model_layer_dict1_low, model_layer_dict1_high, model_layer_dict2_low, model_layer_dict2_high,
               model_layer_dict3_low, model_layer_dict3_high)

K_MULTISECTION = 1000
k_m_partition_1, k_m_partition_2, k_m_partition_3 = partition(model_layer_dict1_low, model_layer_dict1_high,
                                                              model_layer_dict2_low, model_layer_dict2_high,
                                                              model_layer_dict3_low, model_layer_dict3_high,
                                                              K_MULTISECTION)
NBC_1, NBC_2, NBC_3, SNA_1, SNA_2, SNA_3 = part_corner(model_layer_dict1_low, model_layer_dict1_high,
                                                       model_layer_dict2_low, model_layer_dict2_high,
                                                       model_layer_dict3_low, model_layer_dict3_high)

TK = 2
BK = 2
TKNC_1, TKNC_2, TKNC_3, BKNC_1, BKNC_2, BKNC_3 = part_tkbk(model_layer_dict1_low, model_layer_dict1_high,
                                                           model_layer_dict2_low, model_layer_dict2_high,
                                                           model_layer_dict3_low, model_layer_dict3_high)

# Currently only 1 is supported
TKNP = 1
BKNP = 1
TKNP_1, TKNP_2, TKNP_3, BKNP_1, BKNP_2, BKNP_3 = part_pattern(model1, model2, model3, TKNP, BKNP)
TKNP_SET_1, TKNP_SET_2, TKNP_SET_3, BKNP_SET_1, BKNP_SET_2, BKNP_SET_3 = set(), set(), set(), set(), set(), set()


data = loadtxt('adv/data_3_e_high.csv', delimiter=',')
TEST_FEAT = data.reshape(10000, 28, 28, 1)

for d_ in range(TEST_FEAT.shape[0]):
    img_ = np.expand_dims(TEST_FEAT[d_], axis=0)
    CV_1 = return_covarage_values(img_, model1)
    CV_2 = return_covarage_values(img_, model2)
    CV_3 = return_covarage_values(img_, model3)

    top_pattern_1 = ""
    bot_pattern_1 = ""
    for name, cnt in model1_layer_name_number:
        neuron_vals = []
        for num in range(cnt):
            neuron_val = CV_1[(name, num)]
            neuron_vals.append(neuron_val)
            neuron_low = model_layer_dict1_low[(name, num)]
            neuron_high = model_layer_dict1_high[(name, num)]
            if neuron_high == neuron_low:
                continue
            divider = (neuron_high - neuron_low) / K_MULTISECTION

            ## NC ##
            if ((not model_layer_dict1[(name, num)]) and neuron_val >= NC_THRESHOLD):
                model_layer_dict1[(name, num)] = True
            ## KMNC - NBC - SNA##
            index = int((neuron_val - neuron_low) / divider)
            if neuron_val == neuron_high:
                k_m_partition_1[(name, num)][K_MULTISECTION - 1] = 1
            elif index >= K_MULTISECTION:
                NBC_1[(name, num)][1] = 1
                SNA_1[(name, num)] = True
            elif index < 0:
                NBC_1[(name, num)][0] = 1
            else:
                k_m_partition_1[(name, num)][index] = 1

        ### TKNC - BKNC ###
        neuron_vals_sorted = neuron_vals[:]
        neuron_vals_sorted.sort()
        top_indexes = []
        bot_indexes = []
        for a in range(TK):
            top_indexes.append(neuron_vals.index(neuron_vals_sorted[-(a + 1)]))
        for a in range(BK):
            bot_indexes.append(neuron_vals.index(neuron_vals_sorted[a]))
        for l in top_indexes:
            TKNC_1[(name, l)] = True
        for l in bot_indexes:
            BKNC_1[(name, l)] = True

        top_pattern_1 = top_pattern_1 + str(neuron_vals.index(neuron_vals_sorted[-1])) + "-"
        bot_pattern_1 = bot_pattern_1 + str(neuron_vals.index(neuron_vals_sorted[0])) + "-"

    ### TKNP - BKNP ###
    TKNP_SET_1.add(top_pattern_1[:-1])
    BKNP_SET_1.add(bot_pattern_1[:-1])

    top_pattern_2 = ""
    bot_pattern_2 = ""
    for name, cnt in model2_layer_name_number:
        neuron_vals = []
        for num in range(cnt):
            neuron_val = CV_2[(name, num)]
            neuron_vals.append(neuron_val)
            neuron_low = model_layer_dict2_low[(name, num)]
            neuron_high = model_layer_dict2_high[(name, num)]
            if neuron_high == neuron_low:
                continue
            divider = (neuron_high - neuron_low) / K_MULTISECTION

            ## NC ##
            if ((not model_layer_dict2[(name, num)]) and neuron_val >= NC_THRESHOLD):
                model_layer_dict2[(name, num)] = True
            ## KMNC - NBC - SNA##
            index = int((neuron_val - neuron_low) / divider)
            if neuron_val == neuron_high:
                k_m_partition_2[(name, num)][K_MULTISECTION - 1] = 1
            elif index >= K_MULTISECTION:
                NBC_2[(name, num)][1] = 1
                SNA_2[(name, num)] = True
            elif index < 0:
                NBC_2[(name, num)][0] = 1
            else:
                k_m_partition_2[(name, num)][index] = 1

        ### TKNC - BKNC ###
        neuron_vals_sorted = neuron_vals[:]
        neuron_vals_sorted.sort()
        top_indexes = []
        bot_indexes = []
        for a in range(TK):
            top_indexes.append(neuron_vals.index(neuron_vals_sorted[-(a + 1)]))
        for a in range(BK):
            bot_indexes.append(neuron_vals.index(neuron_vals_sorted[a]))
        for l in top_indexes:
            TKNC_2[(name, l)] = True
        for l in bot_indexes:
            BKNC_2[(name, l)] = True

        top_pattern_2 = top_pattern_2 + str(neuron_vals.index(neuron_vals_sorted[-1])) + "-"
        bot_pattern_2 = bot_pattern_2 + str(neuron_vals.index(neuron_vals_sorted[0])) + "-"

    ### TKNP - BKNP ###
    TKNP_SET_2.add(top_pattern_2[:-1])
    BKNP_SET_2.add(bot_pattern_2[:-1])

    top_pattern_3 = ""
    bot_pattern_3 = ""
    for name, cnt in model3_layer_name_number:
        neuron_vals = []
        for num in range(cnt):
            neuron_val = CV_3[(name, num)]
            neuron_vals.append(neuron_val)
            neuron_low = model_layer_dict3_low[(name, num)]
            neuron_high = model_layer_dict3_high[(name, num)]
            if neuron_high == neuron_low:
                continue
            divider = (neuron_high - neuron_low) / K_MULTISECTION
            ## NC ##
            if ((not model_layer_dict3[(name, num)]) and neuron_val >= NC_THRESHOLD):
                model_layer_dict3[(name, num)] = True
            ## KMNC - NBC - SNA##
            index = int((neuron_val - neuron_low) / divider)
            if neuron_val == neuron_high:
                k_m_partition_3[(name, num)][K_MULTISECTION - 1] = 1
            elif index >= K_MULTISECTION:
                NBC_3[(name, num)][1] = 1
                SNA_3[(name, num)] = True
            elif index < 0:
                NBC_3[(name, num)][0] = 1
            else:
                k_m_partition_3[(name, num)][index] = 1

        ### TKNC - BKNC ###
        neuron_vals_sorted = neuron_vals[:]
        neuron_vals_sorted.sort()
        top_indexes = []
        bot_indexes = []
        for a in range(TK):
            top_indexes.append(neuron_vals.index(neuron_vals_sorted[-(a + 1)]))
        for a in range(BK):
            bot_indexes.append(neuron_vals.index(neuron_vals_sorted[a]))
        for l in top_indexes:
            TKNC_3[(name, l)] = True
        for l in bot_indexes:
            BKNC_3[(name, l)] = True

        top_pattern_3 = top_pattern_3 + str(neuron_vals.index(neuron_vals_sorted[-1])) + "-"
        bot_pattern_3 = bot_pattern_3 + str(neuron_vals.index(neuron_vals_sorted[0])) + "-"

    ### TKNP - BKNP ###
    TKNP_SET_3.add(top_pattern_3[:-1])
    BKNP_SET_3.add(bot_pattern_3[:-1])

    if(d_ % 100 == 0):
        print(d_)

    ### KMNC BEGIN ###
    # find_max_and_min(img_, model1, model_layer_dict1_low, model_layer_dict1_high)
    # find_max_and_min(img_, model2, model_layer_dict2_low, model_layer_dict2_high)
    # find_max_and_min(img_, model3, model_layer_dict3_low, model_layer_dict3_high)
    ### KMNC END ###

    # update_coverage(img_, model1, model_layer_dict1, NC_THRESHOLD)
    # update_coverage(img_, model2, model_layer_dict2, NC_THRESHOLD)
    # update_coverage(img_, model3, model_layer_dict3, NC_THRESHOLD)

### TEST COVERAGE SAVE BEGIN ###
#write_test_coverage("m1", model_layer_dict1, k_m_partition_1, NBC_1, SNA_1, TKNC_1, BKNC_1, TKNP_SET_1, BKNP_SET_1)
#write_test_coverage("m2", model_layer_dict2, k_m_partition_2, NBC_2, SNA_2, TKNC_2, BKNC_2, TKNP_SET_2, BKNP_SET_2)
write_test_coverage("m3", model_layer_dict3, k_m_partition_3, NBC_3, SNA_3, TKNC_3, BKNC_3, TKNP_SET_3, BKNP_SET_3)
### TEST COVERAGE SAVE END ###
# print(neuron_covered(model_layer_dict1))
# print(neuron_covered(model_layer_dict2))
# print(neuron_covered(model_layer_dict3))
"""
print(model_layer_dict1_low)
print(model_layer_dict1_high)
print(model_layer_dict2_low)
print(model_layer_dict2_high)
print(model_layer_dict3_low)
print(model_layer_dict3_high)
"""

"""
defaultdict(<type 'float'>, {('block2_conv1', 9): 0.033086207, ('block2_pool1', 7): 0.04682309, ('block1_conv1', 3): 0.047908105, ('predictions', 6): 0.0, ('predictions', 1): 0.0, ('block2_conv1', 6): 0.02928095, ('block1_pool1', 0): 0.070222765, ('block2_pool1', 11): 0.026577447, ('before_softmax', 0): 0.0, ('block2_pool1', 2): 0.015224107, ('predictions', 9): 0.0, ('block2_conv1', 1): 0.011566917, ('before_softmax', 8): 0.0, ('before_softmax', 3): 0.0, ('block2_conv1', 8): 0.013742513, ('block2_pool1', 6): 0.05051718, ('block1_conv1', 0): 0.04817363, ('block2_conv1', 5): 0.044934213, ('before_softmax', 6): 0.0, ('block1_pool1', 1): 0.064923644, ('predictions', 3): 0.0, ('block2_pool1', 10): 0.030581664, ('block2_pool1', 1): 0.021347016, ('predictions', 0): 0.0, ('block2_conv1', 0): 0.031833872, ('before_softmax', 9): 0.0, ('block2_pool1', 5): 0.071813226, ('block1_conv1', 1): 0.04298863, ('before_softmax', 1): 0.0, ('block2_conv1', 4): 0.03657591, ('predictions', 8): 0.0, ('block2_conv1', 11): 0.016327595, ('block2_pool1', 9): 0.044777628, ('before_softmax', 4): 0.0, ('block2_pool1', 0): 0.049232166, ('predictions', 5): 0.0, ('block1_pool1', 2): 0.04964806, ('before_softmax', 7): 0.0, ('predictions', 2): 0.0, ('block2_pool1', 4): 0.05766692, ('block2_conv1', 3): 0.024413394, ('block2_conv1', 10): 0.01704917, ('block2_pool1', 8): 0.02406335, ('block1_conv1', 2): 0.032198578, ('block2_conv1', 7): 0.022511046, ('before_softmax', 2): 0.0, ('block1_pool1', 3): 0.067090206, ('predictions', 7): 0.0, ('block2_pool1', 3): 0.042640164, ('before_softmax', 5): 0.0, ('block2_conv1', 2): 0.008579464, ('predictions', 4): 0.0})
defaultdict(<type 'float'>, {('block2_conv1', 9): 0.16101807, ('block2_pool1', 7): 0.22529447, ('block1_conv1', 3): 0.33968553, ('predictions', 6): 1.0, ('predictions', 1): 1.0, ('block2_conv1', 6): 0.16504237, ('block1_pool1', 0): 0.37808052, ('block2_pool1', 11): 0.23749346, ('before_softmax', 0): 1.0, ('block2_pool1', 2): 0.26198688, ('predictions', 9): 1.0, ('block2_conv1', 1): 0.13213366, ('before_softmax', 8): 1.0, ('before_softmax', 3): 1.0, ('block2_conv1', 8): 0.18768762, ('block2_pool1', 6): 0.25243905, ('block1_conv1', 0): 0.32579246, ('block2_conv1', 5): 0.16829973, ('before_softmax', 6): 1.0, ('block1_pool1', 1): 0.33874854, ('predictions', 3): 1.0, ('block2_pool1', 10): 0.24060592, ('block2_pool1', 1): 0.2432569, ('predictions', 0): 1.0, ('block2_conv1', 0): 0.11350283, ('before_softmax', 9): 1.0, ('block2_pool1', 5): 0.3318647, ('block1_conv1', 1): 0.29348218, ('before_softmax', 1): 1.0, ('block2_conv1', 4): 0.23878154, ('predictions', 8): 1.0, ('block2_conv1', 11): 0.13730957, ('block2_pool1', 9): 0.22110297, ('before_softmax', 4): 1.0, ('block2_pool1', 0): 0.20367977, ('predictions', 5): 1.0, ('block1_pool1', 2): 0.21435139, ('before_softmax', 7): 1.0, ('predictions', 2): 1.0, ('block2_pool1', 4): 0.4359016, ('block2_conv1', 3): 0.13833015, ('block2_conv1', 10): 0.13118586, ('block2_pool1', 8): 0.32133162, ('block1_conv1', 2): 0.17621611, ('block2_conv1', 7): 0.11313923, ('before_softmax', 2): 1.0, ('block1_pool1', 3): 0.39355025, ('predictions', 7): 1.0, ('block2_pool1', 3): 0.25298935, ('before_softmax', 5): 1.0, ('block2_conv1', 2): 0.13066515, ('predictions', 4): 1.0})
defaultdict(<type 'float'>, {('predictions', 1): 0.0, ('block1_conv1', 3): 0.027711794, ('fc1', 57): 0.0, ('block2_pool1', 10): 0.044051763, ('fc1', 46): 0.0, ('before_softmax', 5): 0.0, ('fc1', 29): 0.0, ('before_softmax', 8): 0.0, ('block2_pool1', 4): 0.054737538, ('block1_pool1', 5): 0.06067468, ('block2_pool1', 6): 0.027482385, ('block2_conv1', 5): 0.0050647096, ('fc1', 76): 0.0, ('fc1', 55): 0.0, ('predictions', 4): 0.0, ('block1_conv1', 4): 0.034113273, ('fc1', 36): 0.0, ('fc1', 41): 0.0, ('before_softmax', 2): 0.0, ('fc1', 24): 0.0, ('block2_conv1', 11): 0.013647722, ('fc1', 13): 0.0, ('block2_conv1', 13): 0.017426733, ('fc1', 50): 0.0, ('predictions', 7): 0.0, ('fc1', 39): 0.0, ('fc1', 15): 0.0, ('fc1', 22): 0.0, ('fc1', 79): 0.0, ('fc1', 80): 0.0, ('fc1', 27): 0.0, ('fc1', 1): 0.0, ('block2_conv1', 14): 0.03778524, ('fc1', 69): 0.0, ('fc1', 8): 0.0, ('fc1', 74): 0.0, ('fc1', 61): 0.0, ('fc1', 34): 0.0, ('fc1', 17): 0.0, ('fc1', 83): 0.0, ('block2_pool1', 2): 0.033034172, ('block2_conv1', 1): 0.017921334, ('fc1', 64): 0.0, ('block2_pool1', 15): 0.03334875, ('predictions', 0): 0.0, ('block1_conv1', 0): 0.031345833, ('fc1', 56): 0.0, ('fc1', 45): 0.0, ('block2_pool1', 12): 0.065753564, ('before_softmax', 6): 0.0, ('block2_conv1', 8): 0.04714796, ('fc1', 28): 0.0, ('before_softmax', 9): 0.0, ('block2_pool1', 5): 0.012386905, ('fc1', 2): 0.0, ('block2_conv1', 4): 0.032574955, ('fc1', 67): 0.0, ('fc1', 54): 0.0, ('predictions', 3): 0.0, ('block1_conv1', 5): 0.037928507, ('fc1', 59): 0.0, ('fc1', 40): 0.0, ('fc1', 4): 0.0, ('before_softmax', 3): 0.0, ('fc1', 31): 0.0, ('block2_conv1', 10): 0.020643862, ('fc1', 12): 0.0, ('block2_conv1', 7): 0.015154191, ('fc1', 78): 0.0, ('fc1', 49): 0.0, ('predictions', 6): 0.0, ('block2_conv1', 15): 0.016179336, ('fc1', 38): 0.0, ('fc1', 43): 0.0, ('fc1', 9): 0.0, ('fc1', 21): 0.0, ('before_softmax', 0): 0.0, ('block2_conv1', 9): 0.006902693, ('fc1', 26): 0.0, ('block1_pool1', 0): 0.046276327, ('fc1', 68): 0.0, ('block2_pool1', 11): 0.028352322, ('fc1', 3): 0.0, ('fc1', 73): 0.0, ('fc1', 60): 0.0, ('predictions', 9): 0.0, ('fc1', 33): 0.0, ('block2_pool1', 1): 0.03572272, ('fc1', 16): 0.0, ('fc1', 82): 0.0, ('fc1', 5): 0.0, ('block2_conv1', 0): 0.0076622353, ('fc1', 71): 0.0, ('block2_pool1', 14): 0.06301706, ('block1_conv1', 1): 0.021154596, ('fc1', 63): 0.0, ('fc1', 44): 0.0, ('before_softmax', 7): 0.0, ('fc1', 10): 0.0, ('fc1', 19): 0.0, ('fc1', 0): 0.0, ('block2_conv1', 3): 0.014589614, ('fc1', 66): 0.0, ('fc1', 53): 0.0, ('predictions', 2): 0.0, ('block2_pool1', 8): 0.069948725, ('block1_conv1', 2): 0.04731699, ('fc1', 58): 0.0, ('fc1', 47): 0.0, ('before_softmax', 4): 0.0, ('fc1', 6): 0.0, ('fc1', 30): 0.0, ('block1_pool1', 4): 0.053577308, ('block2_pool1', 7): 0.029678581, ('block2_conv1', 6): 0.013011817, ('fc1', 77): 0.0, ('fc1', 48): 0.0, ('predictions', 5): 0.0, ('fc1', 37): 0.0, ('fc1', 42): 0.0, ('block1_pool1', 3): 0.046626933, ('fc1', 20): 0.0, ('before_softmax', 1): 0.0, ('fc1', 11): 0.0, ('fc1', 25): 0.0, ('block1_pool1', 1): 0.037470754, ('fc1', 14): 0.0, ('fc1', 72): 0.0, ('fc1', 51): 0.0, ('predictions', 8): 0.0, ('block2_pool1', 9): 0.016425667, ('fc1', 32): 0.0, ('fc1', 23): 0.0, ('block2_pool1', 3): 0.031662706, ('fc1', 81): 0.0, ('block2_pool1', 0): 0.01591834, ('block1_pool1', 2): 0.070750944, ('fc1', 70): 0.0, ('block2_pool1', 13): 0.032187458, ('fc1', 75): 0.0, ('fc1', 62): 0.0, ('fc1', 35): 0.0, ('fc1', 65): 0.0, ('fc1', 18): 0.0, ('fc1', 7): 0.0, ('block2_conv1', 2): 0.015559432, ('block2_conv1', 12): 0.039792914, ('fc1', 52): 0.0})
defaultdict(<type 'float'>, {('predictions', 1): 1.0, ('block1_conv1', 3): 0.14622486, ('fc1', 57): 1.0, ('block2_pool1', 10): 0.23312467, ('fc1', 46): 1.0, ('before_softmax', 5): 1.0, ('fc1', 29): 1.0, ('before_softmax', 8): 1.0, ('block2_pool1', 4): 0.2887479, ('block1_pool1', 5): 0.3037449, ('block2_pool1', 6): 0.23040637, ('block2_conv1', 5): 0.106112584, ('fc1', 76): 0.1375138, ('fc1', 55): 1.0, ('predictions', 4): 1.0, ('block1_conv1', 4): 0.1725899, ('fc1', 36): 1.0, ('fc1', 41): 1.0, ('before_softmax', 2): 1.0, ('fc1', 24): 1.0, ('block2_conv1', 11): 0.07406346, ('fc1', 13): 0.04916885, ('block2_conv1', 13): 0.1854335, ('fc1', 50): 1.0, ('predictions', 7): 1.0, ('fc1', 39): 1.0, ('fc1', 15): 1.0, ('fc1', 22): 1.0, ('fc1', 79): 1.0, ('fc1', 80): 1.0, ('fc1', 27): 1.0, ('fc1', 1): 0.90932995, ('block2_conv1', 14): 0.25255185, ('fc1', 69): 0.82244223, ('fc1', 8): 1.0, ('fc1', 74): 1.0, ('fc1', 61): 1.0, ('fc1', 34): 1.0, ('fc1', 17): 1.0, ('fc1', 83): 1.0, ('block2_pool1', 2): 0.1882171, ('block2_conv1', 1): 0.1269827, ('fc1', 64): 1.0, ('block2_pool1', 15): 0.24902605, ('predictions', 0): 1.0, ('block1_conv1', 0): 0.21538112, ('fc1', 56): 1.0, ('fc1', 45): 1.0, ('block2_pool1', 12): 0.32251933, ('before_softmax', 6): 1.0, ('block2_conv1', 8): 0.18399061, ('fc1', 28): 0.74262065, ('before_softmax', 9): 1.0, ('block2_pool1', 5): 0.21591645, ('fc1', 2): 1.0, ('block2_conv1', 4): 0.17393832, ('fc1', 67): 1.0, ('fc1', 54): 1.0, ('predictions', 3): 1.0, ('block1_conv1', 5): 0.2606667, ('fc1', 59): 0.9731231, ('fc1', 40): 1.0, ('fc1', 4): 0.7007251, ('before_softmax', 3): 1.0, ('fc1', 31): 1.0, ('block2_conv1', 10): 0.11956896, ('fc1', 12): 1.0, ('block2_conv1', 7): 0.09999211, ('fc1', 78): 1.0, ('fc1', 49): 1.0, ('predictions', 6): 1.0, ('block2_conv1', 15): 0.111280166, ('fc1', 38): 1.0, ('fc1', 43): 1.0, ('fc1', 9): 1.0, ('fc1', 21): 1.0, ('before_softmax', 0): 1.0, ('block2_conv1', 9): 0.06613641, ('fc1', 26): 1.0, ('block1_pool1', 0): 0.25661272, ('fc1', 68): 1.0, ('block2_pool1', 11): 0.15799266, ('fc1', 3): 1.0, ('fc1', 73): 1.0, ('fc1', 60): 0.92988527, ('predictions', 9): 1.0, ('fc1', 33): 1.0, ('block2_pool1', 1): 0.2402012, ('fc1', 16): 0.87876445, ('fc1', 82): 0.8836001, ('fc1', 5): 1.0, ('block2_conv1', 0): 0.102844365, ('fc1', 71): 0.92142725, ('block2_pool1', 14): 0.34446856, ('block1_conv1', 1): 0.13528454, ('fc1', 63): 1.0, ('fc1', 44): 0.9735417, ('before_softmax', 7): 1.0, ('fc1', 10): 1.0, ('fc1', 19): 1.0, ('fc1', 0): 1.0, ('block2_conv1', 3): 0.17289087, ('fc1', 66): 1.0, ('fc1', 53): 1.0, ('predictions', 2): 1.0, ('block2_pool1', 8): 0.3039949, ('block1_conv1', 2): 0.3355417, ('fc1', 58): 1.0, ('fc1', 47): 1.0, ('before_softmax', 4): 1.0, ('fc1', 6): 0.7293609, ('fc1', 30): 0.9654123, ('block1_pool1', 4): 0.21504351, ('block2_pool1', 7): 0.16731936, ('block2_conv1', 6): 0.10251709, ('fc1', 77): 1.0, ('fc1', 48): 1.0, ('predictions', 5): 1.0, ('fc1', 37): 0.93028766, ('fc1', 42): 1.0, ('block1_pool1', 3): 0.18179142, ('fc1', 20): 1.0, ('before_softmax', 1): 1.0, ('fc1', 11): 1.0, ('fc1', 25): 0.86548024, ('block1_pool1', 1): 0.20536841, ('fc1', 14): 0.16382174, ('fc1', 72): 1.0, ('fc1', 51): 0.98048615, ('predictions', 8): 1.0, ('block2_pool1', 9): 0.140504, ('fc1', 32): 1.0, ('fc1', 23): 1.0, ('block2_pool1', 3): 0.29205865, ('fc1', 81): 1.0, ('block2_pool1', 0): 0.21263206, ('block1_pool1', 2): 0.3909522, ('fc1', 70): 1.0, ('block2_pool1', 13): 0.3623383, ('fc1', 75): 1.0, ('fc1', 62): 1.0, ('fc1', 35): 1.0, ('fc1', 65): 1.0, ('fc1', 18): 0.98954916, ('fc1', 7): 0.85943806, ('block2_conv1', 2): 0.093313344, ('block2_conv1', 12): 0.21342848, ('fc1', 52): 1.0})
defaultdict(<type 'float'>, {('fc2', 68): 0.0, ('fc2', 30): 0.0, ('block1_conv1', 3): 0.05470232, ('fc1', 57): 0.0, ('fc2', 80): 0.0, ('block2_pool1', 10): 0.026789505, ('fc2', 62): 0.0, ('fc1', 46): 0.0, ('fc2', 52): 0.0, ('fc1', 104): 0.0, ('before_softmax', 5): 0.0, ('fc2', 24): 0.0, ('fc1', 29): 0.0, ('fc2', 42): 0.0, ('block2_pool1', 4): 0.011033233, ('block1_pool1', 5): 0.019103337, ('fc1', 95): 0.0, ('block2_pool1', 6): 0.057859816, ('fc2', 28): 0.0, ('block2_conv1', 5): 0.034202825, ('fc1', 76): 0.0, ('fc1', 55): 0.0, ('fc2', 26): 0.0, ('predictions', 4): 0.0, ('fc1', 87): 0.0, ('block1_conv1', 4): 0.06293703, ('fc1', 36): 0.0, ('fc2', 36): 0.0, ('fc2', 7): 0.0, ('fc1', 102): 0.0, ('fc1', 41): 0.0, ('fc2', 71): 0.0, ('fc2', 48): 0.0, ('fc1', 107): 0.0, ('fc1', 113): 0.0, ('fc1', 85): 0.0, ('fc1', 24): 0.0, ('block2_conv1', 11): 0.021681799, ('fc1', 90): 0.0, ('fc1', 13): 0.0, ('fc2', 49): 0.0, ('block2_conv1', 13): 0.022119123, ('fc1', 50): 0.0, ('before_softmax', 9): 0.0, ('predictions', 7): 0.0, ('fc2', 78): 0.0, ('fc2', 1): 0.0, ('fc2', 57): 0.0, ('fc1', 39): 0.0, ('fc2', 83): 0.0, ('fc1', 15): 0.0, ('fc1', 97): 0.0, ('fc2', 73): 0.0, ('fc1', 22): 0.0, ('fc1', 79): 0.0, ('fc1', 117): 0.0, ('fc1', 80): 0.0, ('fc1', 27): 0.0, ('fc2', 45): 0.0, ('fc1', 1): 0.0, ('block2_conv1', 14): 0.011811786, ('fc1', 69): 0.0, ('fc1', 8): 0.0, ('fc2', 77): 0.0, ('fc1', 74): 0.0, ('fc1', 61): 0.0, ('fc2', 21): 0.0, ('fc2', 12): 0.0, ('fc1', 34): 0.10007121, ('fc2', 39): 0.0, ('fc2', 70): 0.0, ('before_softmax', 7): 0.0, ('fc1', 108): 0.0, ('fc2', 64): 0.0, ('fc2', 69): 0.0, ('fc1', 17): 0.0, ('fc2', 18): 0.0, ('fc2', 8): 0.0, ('fc1', 83): 0.0, ('block2_pool1', 2): 0.043543667, ('fc2', 33): 0.0, ('fc2', 53): 0.0, ('block2_conv1', 1): 0.020561138, ('fc1', 64): 0.0, ('block2_pool1', 15): 0.027086593, ('predictions', 0): 0.0, ('block1_conv1', 0): 0.032408938, ('fc1', 56): 0.0, ('before_softmax', 8): 0.0, ('fc2', 0): 0.0, ('fc2', 63): 0.0, ('fc1', 45): 0.0, ('fc2', 82): 0.0, ('block2_pool1', 12): 0.02312727, ('fc1', 111): 0.0, ('before_softmax', 6): 0.0, ('fc2', 27): 0.0, ('block2_conv1', 8): 0.0049970252, ('fc1', 28): 0.0, ('fc2', 65): 0.0, ('fc2', 4): 0.0, ('fc1', 94): 0.0, ('block2_pool1', 5): 0.06319748, ('fc2', 44): 0.0, ('fc2', 74): 0.0, ('fc1', 2): 0.0, ('block2_conv1', 4): 0.0055096587, ('fc1', 67): 0.0, ('fc1', 54): 0.0, ('fc2', 55): 0.0, ('predictions', 3): 0.0, ('fc1', 112): 0.0, ('block1_conv1', 5): 0.011562251, ('fc1', 59): 0.0, ('fc2', 20): 0.0, ('fc1', 101): 0.0, ('fc1', 40): 0.0, ('fc2', 38): 0.0, ('fc1', 4): 0.0, ('fc2', 58): 0.0, ('fc1', 106): 0.0, ('before_softmax', 3): 0.0, ('fc2', 29): 0.0, ('fc1', 84): 0.0, ('fc1', 31): 0.0, ('before_softmax', 4): 0.0, ('block2_conv1', 10): 0.016703285, ('fc1', 89): 0.0, ('fc1', 12): 0.0, ('fc2', 32): 0.0, ('block2_conv1', 7): 0.008916336, ('fc1', 78): 0.0, ('fc1', 49): 0.0, ('fc2', 13): 0.0, ('predictions', 6): 0.0, ('block2_conv1', 15): 0.013282221, ('fc1', 38): 0.0, ('fc2', 54): 0.0, ('fc2', 3): 0.0, ('fc1', 96): 0.0, ('fc1', 43): 0.0, ('fc1', 9): 0.0, ('fc1', 21): 0.0, ('before_softmax', 0): 0.0, ('block2_conv1', 9): 0.014870997, ('fc1', 26): 0.0, ('fc2', 19): 0.0, ('block1_pool1', 0): 0.052260127, ('fc1', 68): 0.0, ('block2_pool1', 11): 0.045181327, ('fc2', 47): 0.0, ('fc1', 3): 0.0, ('fc1', 73): 0.0, ('fc1', 60): 0.0, ('fc2', 66): 0.0, ('predictions', 9): 0.0, ('fc1', 115): 0.0, ('fc2', 9): 0.0, ('fc2', 59): 0.0, ('fc1', 33): 0.0, ('fc2', 79): 0.0, ('fc2', 23): 0.0, ('fc1', 99): 0.0, ('fc2', 41): 0.0, ('block2_pool1', 1): 0.043701112, ('fc1', 16): 0.0, ('fc2', 11): 0.0, ('fc1', 82): 0.0, ('fc1', 5): 0.0, ('fc2', 17): 0.0, ('block2_conv1', 0): 0.04050405, ('fc1', 71): 0.0, ('block2_pool1', 14): 0.02425445, ('fc2', 35): 0.0, ('block1_conv1', 1): 0.034216344, ('fc1', 63): 0.0, ('fc2', 75): 0.0, ('fc2', 37): 0.0, ('predictions', 1): 0.0, ('fc2', 60): 0.0, ('fc1', 44): 0.0, ('fc2', 2): 0.0, ('fc1', 110): 0.0, ('fc2', 14): 0.0, ('fc1', 10): 0.0, ('fc1', 19): 0.0, ('before_softmax', 2): 0.0, ('fc2', 50): 0.0, ('fc1', 93): 0.0, ('fc1', 0): 0.0, ('block2_conv1', 3): 0.014588332, ('fc1', 66): 0.0, ('fc1', 53): 0.0, ('fc2', 46): 0.0, ('predictions', 2): 0.0, ('block2_pool1', 8): 0.011399894, ('block1_conv1', 2): 0.016739441, ('fc1', 58): 0.0, ('fc1', 100): 0.0, ('fc1', 47): 0.0, ('fc2', 22): 0.0, ('fc1', 105): 0.0, ('fc2', 40): 0.0, ('fc1', 6): 0.0, ('fc2', 10): 0.0, ('fc1', 30): 0.0, ('block1_pool1', 4): 0.09019754, ('fc1', 88): 0.0, ('block2_pool1', 7): 0.019318119, ('fc2', 16): 0.0, ('block2_conv1', 6): 0.031722177, ('fc1', 77): 0.0, ('fc1', 48): 0.0, ('fc2', 34): 0.0, ('predictions', 5): 0.0, ('fc1', 114): 0.0, ('fc1', 37): 0.0, ('fc1', 103): 0.0, ('fc1', 42): 0.0, ('block1_pool1', 3): 0.07411121, ('fc2', 5): 0.0, ('fc1', 20): 0.0, ('before_softmax', 1): 0.0, ('fc2', 67): 0.0, ('fc1', 11): 0.0, ('fc1', 86): 0.0, ('fc1', 25): 0.0, ('fc2', 6): 0.0, ('fc1', 119): 0.0, ('block1_pool1', 1): 0.049725745, ('fc1', 91): 0.0, ('fc1', 14): 0.0, ('fc2', 51): 0.0, ('fc2', 31): 0.0, ('fc1', 72): 0.0, ('fc1', 51): 0.0, ('fc2', 81): 0.0, ('predictions', 8): 0.0, ('block2_pool1', 9): 0.030879244, ('fc2', 56): 0.0, ('fc1', 32): 0.0, ('fc1', 98): 0.0, ('fc2', 25): 0.0, ('fc1', 23): 0.0, ('fc2', 43): 0.0, ('block2_pool1', 3): 0.029472763, ('fc1', 81): 0.0, ('block2_pool1', 0): 0.06993116, ('block1_pool1', 2): 0.028102685, ('fc1', 70): 0.0, ('block2_pool1', 13): 0.04372009, ('fc2', 72): 0.0, ('fc2', 15): 0.0, ('fc1', 75): 0.0, ('fc1', 62): 0.0, ('fc1', 116): 0.0, ('fc1', 118): 0.0, ('fc2', 61): 0.0, ('fc1', 35): 0.0, ('fc1', 109): 0.0, ('fc2', 76): 0.0, ('fc1', 65): 0.0, ('fc1', 18): 0.0, ('fc1', 92): 0.0, ('fc1', 7): 0.0, ('block2_conv1', 2): 0.02460688, ('block2_conv1', 12): 0.01192934, ('fc1', 52): 0.0})
defaultdict(<type 'float'>, {('fc2', 68): 1.0, ('fc2', 30): 1.0, ('block1_conv1', 3): 0.32370946, ('fc1', 57): 0.059788074, ('fc2', 80): 1.0, ('block2_pool1', 10): 0.2423685, ('fc2', 62): 1.0, ('fc1', 46): 1.0, ('fc2', 52): 1.0, ('fc1', 104): 1.0, ('before_softmax', 5): 1.0, ('fc2', 24): 0.3296627, ('fc1', 29): 0.8698345, ('fc2', 42): 1.0, ('block2_pool1', 4): 0.12899974, ('block1_pool1', 5): 0.1450392, ('fc1', 95): 1.0, ('block2_pool1', 6): 0.26850387, ('fc2', 28): 0.546643, ('block2_conv1', 5): 0.21791863, ('fc1', 76): 1.0, ('fc1', 55): 0.52365613, ('fc2', 26): 0.8785994, ('predictions', 4): 1.0, ('fc1', 87): 1.0, ('block1_conv1', 4): 0.37818804, ('fc1', 36): 1.0, ('fc2', 36): 1.0, ('fc2', 7): 0.98766494, ('fc1', 102): 1.0, ('fc1', 41): 1.0, ('fc2', 71): 0.0, ('fc2', 48): 1.0, ('fc1', 107): 0.95419544, ('fc1', 113): 0.6179734, ('fc1', 85): 1.0, ('fc1', 24): 0.9259595, ('block2_conv1', 11): 0.15219513, ('fc1', 90): 1.0, ('fc1', 13): 0.7823391, ('fc2', 49): 0.07831333, ('block2_conv1', 13): 0.13278665, ('fc1', 50): 0.88459045, ('before_softmax', 9): 1.0, ('predictions', 7): 1.0, ('fc2', 78): 0.18822232, ('fc2', 1): 1.0, ('fc2', 57): 0.0, ('fc1', 39): 1.0, ('fc2', 83): 1.0, ('fc1', 15): 0.943191, ('fc1', 97): 1.0, ('fc2', 73): 1.0, ('fc1', 22): 1.0, ('fc1', 79): 0.66425127, ('fc1', 117): 1.0, ('fc1', 80): 0.919589, ('fc1', 27): 1.0, ('fc2', 45): 0.42938995, ('fc1', 1): 1.0, ('block2_conv1', 14): 0.07559704, ('fc1', 69): 0.8191407, ('fc1', 8): 1.0, ('fc2', 77): 1.0, ('fc1', 74): 1.0, ('fc1', 61): 0.5442218, ('fc2', 21): 1.0, ('fc2', 12): 0.7159291, ('fc1', 34): 1.0, ('fc2', 39): 1.0, ('fc2', 70): 0.18487515, ('before_softmax', 7): 1.0, ('fc1', 108): 1.0, ('fc2', 64): 1.0, ('fc2', 69): 0.8583876, ('fc1', 17): 1.0, ('fc2', 18): 0.5330342, ('fc2', 8): 0.8000802, ('fc1', 83): 1.0, ('block2_pool1', 2): 0.3194955, ('fc2', 33): 1.0, ('fc2', 53): 0.6883422, ('block2_conv1', 1): 0.14362627, ('fc1', 64): 1.0, ('block2_pool1', 15): 0.3661429, ('predictions', 0): 1.0, ('block1_conv1', 0): 0.14259124, ('fc1', 56): 0.98218715, ('before_softmax', 8): 1.0, ('fc2', 0): 1.0, ('fc2', 63): 1.0, ('fc1', 45): 1.0, ('fc2', 82): 0.472976, ('block2_pool1', 12): 0.15774524, ('fc1', 111): 0.9798244, ('before_softmax', 6): 1.0, ('fc2', 27): 0.9865098, ('block2_conv1', 8): 0.10511847, ('fc1', 28): 0.042089045, ('fc2', 65): 1.0, ('fc2', 4): 0.0173925, ('fc1', 94): 1.0, ('block2_pool1', 5): 0.37666652, ('fc2', 44): 1.0, ('fc2', 74): 1.0, ('fc1', 2): 0.9847048, ('block2_conv1', 4): 0.06986401, ('fc1', 67): 0.737461, ('fc1', 54): 1.0, ('fc2', 55): 0.6565284, ('predictions', 3): 1.0, ('fc1', 112): 0.989968, ('block1_conv1', 5): 0.089176275, ('fc1', 59): 0.92373204, ('fc2', 20): 1.0, ('fc1', 101): 1.0, ('fc1', 40): 0.8895141, ('fc2', 38): 0.64443845, ('fc1', 4): 1.0, ('fc2', 58): 1.0, ('fc1', 106): 1.0, ('before_softmax', 3): 1.0, ('fc2', 29): 1.0, ('fc1', 84): 0.66690415, ('fc1', 31): 0.9546386, ('before_softmax', 4): 1.0, ('block2_conv1', 10): 0.12026866, ('fc1', 89): 1.0, ('fc1', 12): 0.9717432, ('fc2', 32): 1.0, ('block2_conv1', 7): 0.12463776, ('fc1', 78): 0.9725452, ('fc1', 49): 0.891408, ('fc2', 13): 1.0, ('predictions', 6): 1.0, ('block2_conv1', 15): 0.16383669, ('fc1', 38): 0.86041933, ('fc2', 54): 0.88174796, ('fc2', 3): 1.0, ('fc1', 96): 1.0, ('fc1', 43): 1.0, ('fc1', 9): 1.0, ('fc1', 21): 0.08907722, ('before_softmax', 0): 1.0, ('block2_conv1', 9): 0.12705295, ('fc1', 26): 1.0, ('fc2', 19): 1.0, ('block1_pool1', 0): 0.18685105, ('fc1', 68): 1.0, ('block2_pool1', 11): 0.28417876, ('fc2', 47): 1.0, ('fc1', 3): 0.9708952, ('fc1', 73): 1.0, ('fc1', 60): 0.66428804, ('fc2', 66): 1.0, ('predictions', 9): 1.0, ('fc1', 115): 1.0, ('fc2', 9): 1.0, ('fc2', 59): 0.36309975, ('fc1', 33): 0.30470634, ('fc2', 79): 0.37063903, ('fc2', 23): 1.0, ('fc1', 99): 0.8717386, ('fc2', 41): 0.21914002, ('block2_pool1', 1): 0.2635901, ('fc1', 16): 1.0, ('fc2', 11): 1.0, ('fc1', 82): 1.0, ('fc1', 5): 0.66908556, ('fc2', 17): 1.0, ('block2_conv1', 0): 0.20522016, ('fc1', 71): 0.039043993, ('block2_pool1', 14): 0.14256968, ('fc2', 35): 0.9778548, ('block1_conv1', 1): 0.1943104, ('fc1', 63): 0.8558403, ('fc2', 75): 1.0, ('fc2', 37): 1.0, ('predictions', 1): 1.0, ('fc2', 60): 0.8999568, ('fc1', 44): 1.0, ('fc2', 2): 0.6270368, ('fc1', 110): 1.0, ('fc2', 14): 1.0, ('fc1', 10): 1.0, ('fc1', 19): 0.41345164, ('before_softmax', 2): 1.0, ('fc2', 50): 1.0, ('fc1', 93): 1.0, ('fc1', 0): 1.0, ('block2_conv1', 3): 0.1495756, ('fc1', 66): 1.0, ('fc1', 53): 1.0, ('fc2', 46): 1.0, ('predictions', 2): 1.0, ('block2_pool1', 8): 0.20925125, ('block1_conv1', 2): 0.07444016, ('fc1', 58): 1.0, ('fc1', 100): 1.0, ('fc1', 47): 1.0, ('fc2', 22): 0.1995792, ('fc1', 105): 1.0, ('fc2', 40): 1.0, ('fc1', 6): 1.0, ('fc2', 10): 1.0, ('fc1', 30): 0.9529159, ('block1_pool1', 4): 0.43293887, ('fc1', 88): 1.0, ('block2_pool1', 7): 0.24319166, ('fc2', 16): 0.8581228, ('block2_conv1', 6): 0.12790962, ('fc1', 77): 1.0, ('fc1', 48): 1.0, ('fc2', 34): 0.6928834, ('predictions', 5): 1.0, ('fc1', 114): 0.9571608, ('fc1', 37): 0.9689524, ('fc1', 103): 0.7837784, ('fc1', 42): 0.9936097, ('block1_pool1', 3): 0.37455118, ('fc2', 5): 0.6081911, ('fc1', 20): 1.0, ('before_softmax', 1): 1.0, ('fc2', 67): 1.0, ('fc1', 11): 0.90257156, ('fc1', 86): 1.0, ('fc1', 25): 0.3894447, ('fc2', 6): 1.0, ('fc1', 119): 1.0, ('block1_pool1', 1): 0.23325546, ('fc1', 91): 1.0, ('fc1', 14): 1.0, ('fc2', 51): 1.0, ('fc2', 31): 1.0, ('fc1', 72): 1.0, ('fc1', 51): 0.3462214, ('fc2', 81): 1.0, ('predictions', 8): 1.0, ('block2_pool1', 9): 0.28276902, ('fc2', 56): 1.0, ('fc1', 32): 1.0, ('fc1', 98): 1.0, ('fc2', 25): 1.0, ('fc1', 23): 1.0, ('fc2', 43): 0.91185075, ('block2_pool1', 3): 0.28909162, ('fc1', 81): 0.7331271, ('block2_pool1', 0): 0.33974582, ('block1_pool1', 2): 0.11471686, ('fc1', 70): 1.0, ('block2_pool1', 13): 0.27897403, ('fc2', 72): 0.08400617, ('fc2', 15): 0.6100274, ('fc1', 75): 1.0, ('fc1', 62): 1.0, ('fc1', 116): 0.075542934, ('fc1', 118): 1.0, ('fc2', 61): 1.0, ('fc1', 35): 0.97274476, ('fc1', 109): 1.0, ('fc2', 76): 1.0, ('fc1', 65): 1.0, ('fc1', 18): 1.0, ('fc1', 92): 1.0, ('fc1', 7): 1.0, ('block2_conv1', 2): 0.1662792, ('block2_conv1', 12): 0.07657531, ('fc1', 52): 0.20440497})
"""

### Find max and min ###
# write_max_min(model_layer_dict1_low, model_layer_dict1_high, model_layer_dict2_low, model_layer_dict2_high, model_layer_dict3_low, model_layer_dict3_high)

# print(model1.predict(TEST_FEAT[0]))

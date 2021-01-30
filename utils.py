import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.models import Model
from itertools import combinations


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_kmnc_dict(model, model_layer_dict, high):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            if high:
                model_layer_dict[(layer.name, index)] = -np.inf
            else:
                model_layer_dict[(layer.name, index)] = np.inf


def init_kmnc_tables(model1, model2, model3):
    model_layer_dict1_low = defaultdict(float)
    model_layer_dict1_high = defaultdict(float)
    model_layer_dict2_low = defaultdict(float)
    model_layer_dict2_high = defaultdict(float)
    model_layer_dict3_low = defaultdict(float)
    model_layer_dict3_high = defaultdict(float)

    init_kmnc_dict(model1, model_layer_dict1_low, False)
    init_kmnc_dict(model1, model_layer_dict1_high, True)
    init_kmnc_dict(model2, model_layer_dict2_high, True)
    init_kmnc_dict(model2, model_layer_dict2_low, False)
    init_kmnc_dict(model3, model_layer_dict3_high, True)
    init_kmnc_dict(model3, model_layer_dict3_low, False)
    return model_layer_dict1_low, model_layer_dict1_high, model_layer_dict2_low, model_layer_dict2_high, model_layer_dict3_low, model_layer_dict3_high


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def find_max_and_min(input_data, model, kmnc_dict_low, kmnc_dict_high):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in xrange(scaled.shape[-1]):
            val = np.mean(scaled[..., num_neuron])
            kmnc_dict_high[(layer_names[i], num_neuron)] = max(val, kmnc_dict_high[(layer_names[i], num_neuron)])
            kmnc_dict_low[(layer_names[i], num_neuron)] = min(val, kmnc_dict_low[(layer_names[i], num_neuron)])


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in xrange(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


def return_covarage_values(input_data, model):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    ret_cov = {}
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in xrange(scaled.shape[-1]):
            ret_cov[(layer_names[i], num_neuron)] = np.mean(scaled[..., num_neuron])
    return ret_cov


def partition(model_layer_dict1_low, model_layer_dict1_high, model_layer_dict2_low, model_layer_dict2_high,
              model_layer_dict3_low, model_layer_dict3_high, K_MULTISECTION):
    lenet_1_part = {}
    for a, b in model_layer_dict1_low:
        lenet_1_part[(a, b)] = np.zeros((K_MULTISECTION))
    lenet_4_part = {}
    for a, b in model_layer_dict2_low:
        lenet_4_part[(a, b)] = np.zeros((K_MULTISECTION))
    lenet_5_part = {}
    for a, b in model_layer_dict3_low:
        lenet_5_part[(a, b)] = np.zeros((K_MULTISECTION))
    return lenet_1_part, lenet_4_part, lenet_5_part


def part_corner(model_layer_dict1_low, model_layer_dict1_high, model_layer_dict2_low, model_layer_dict2_high,
                model_layer_dict3_low, model_layer_dict3_high):
    lenet_1_part = {}
    SNA_1 = {}
    for a, b in model_layer_dict1_low:
        lenet_1_part[(a, b)] = np.zeros((2))
        SNA_1[(a, b)] = False
    lenet_4_part = {}
    SNA_2 = {}
    for a, b in model_layer_dict2_low:
        lenet_4_part[(a, b)] = np.zeros((2))
        SNA_2[(a, b)] = False
    lenet_5_part = {}
    SNA_3 = {}
    for a, b in model_layer_dict3_low:
        lenet_5_part[(a, b)] = np.zeros((2))
        SNA_3[(a, b)] = False
    return lenet_1_part, lenet_4_part, lenet_5_part, SNA_1, SNA_2, SNA_3


def part_tkbk(model_layer_dict1_low, model_layer_dict1_high, model_layer_dict2_low, model_layer_dict2_high,
              model_layer_dict3_low, model_layer_dict3_high):
    tk_1 = {}
    bk_1 = {}
    for a, b in model_layer_dict1_low:
        tk_1[(a, b)] = False
        bk_1[(a, b)] = False
    tk_2 = {}
    bk_2 = {}
    for a, b in model_layer_dict2_low:
        tk_2[(a, b)] = False
        bk_2[(a, b)] = False
    tk_3 = {}
    bk_3 = {}
    for a, b in model_layer_dict3_low:
        tk_3[(a, b)] = False
        bk_3[(a, b)] = False
    return tk_1, tk_2, tk_3, bk_1, bk_2, bk_3


def part_pattern(model1, model2, model3, TKNP, BKNP):
    tknp_1 = 1
    bknp_1 = 1
    for layer in model1.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        tknp_1 = tknp_1 * len(list(combinations(range(layer.output_shape[-1]), TKNP)))
        bknp_1 = bknp_1 * len(list(combinations(range(layer.output_shape[-1]), BKNP)))
    tknp_2 = 1
    bknp_2 = 1
    for layer in model2.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        tknp_2 = tknp_2 * len(list(combinations(range(layer.output_shape[-1]), TKNP)))
        bknp_2 = bknp_2 * len(list(combinations(range(layer.output_shape[-1]), BKNP)))

    tknp_3 = 1
    bknp_3 = 1
    for layer in model3.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        tknp_3 = tknp_3 * len(list(combinations(range(layer.output_shape[-1]), TKNP)))
        bknp_3 = bknp_3 * len(list(combinations(range(layer.output_shape[-1]), BKNP)))
    return tknp_1, tknp_2, tknp_3, bknp_1, bknp_2, bknp_3


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

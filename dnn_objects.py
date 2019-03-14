#!/usr/bin/env python

from enum import Enum

class ActivationType(Enum):
    '''
    enum-like class for layers activations
    '''
    linear = 'linear'
    relu = 'relu'
    relu6 = 'relu6'
    leaky = 'leaky'
    softmax = 'softmax'
    elu = 'elu'
    sigmoid = 'sigmoid'

ActivationTypes = {x.value: x for x in ActivationType}


class LayerType(Enum):
    '''
    enum-like class for layers types
    '''
    input_layer = 'input_layer'
    output_layer = 'output_layer'
    conv = 'conv'
    dw = 'dw'
    dense = 'dense'
    maxpool = 'maxpool'
    avgpool = 'avgpool'
    concat = 'concat'
    base_conv = 'base_conv'
    base_dw = 'base_dw'
    base_dense = 'base_dense'
    base_batch_norm = 'base_batch_norm'
    batch_norm = 'batch_norm'
    base_deconv = 'base_deconv'
    deconv = 'deconv'

LayerTypes = {x.value: x for x in LayerType}



class PaddingType(Enum):
    '''
    enum-like class for padding LayerTypes
    '''
    same = 'SAME'
    valid = 'VALID'
    same_tensorflow = 'SAME_TENSORFLOW'

PaddingTypes = {x.value: x for x in PaddingType}







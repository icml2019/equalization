#!/usr/bin/env python

import copy

from collections import namedtuple

import numpy as np
from dnn_objects import (LayerType, ActivationType, ActivationTypes, PaddingType, PaddingTypes)


def _validate_external_padding_scheme(external_padding_value, kernel_h, kernel_w):
    # This is a common alternative padding scheme which we decided to support.
    # This is the chosen padding scheme when using SAME in HN.
    pad_total_h = kernel_h - 1
    pad_beg_h = pad_total_h // 2
    pad_end_h = pad_total_h - pad_beg_h
    pad_total_w = kernel_w - 1
    pad_beg_w = pad_total_w // 2
    pad_end_w = pad_total_w - pad_beg_w
    expected_padding = np.asarray([[0, 0], [pad_beg_h, pad_end_h], [pad_beg_w, pad_end_w], [0, 0]])

    if not np.array_equal(expected_padding, external_padding_value):
        raise Exception('Unsupported external padding scheme, expected {}, but got {}'.format(expected_padding, external_padding_value))




def input_to_output_height_width(input_shape, kernel_shape, strides, padding, dilations=None):
    output_h, output_w = input_shape[1:3]
    if padding == PaddingType.valid:
        dil_h, dil_w = [1, 1] if dilations is None else dilations[1:3]
        output_h -= 2 * (dil_h * (kernel_shape[0] - 1)) // 2
        output_w -= 2 * (dil_w * (kernel_shape[1] - 1)) // 2
    else:
        output_h = int(np.ceil(output_h / float(strides[1])))
        output_w = int(np.ceil(output_w / float(strides[2])))

    return output_h, output_w

BatchNormValues = namedtuple(
    'BatchNormValues',
    ['moving_mean', 'moving_variance', 'beta', 'gamma', 'epsilon']
)

class Layer(object):

    next_insertion_order = 0

    def __init__(self):
        self.index = None
        self.name = None
        self._original_names = list()
        self._input_shapes = []
        self._output_shapes = []
        self._op = None


        self._hash = Layer.next_insertion_order
        self._insertion_order = Layer.next_insertion_order
        Layer.next_insertion_order += 1

    def _get_shape_info(self):
        if self.output_shapes is not None:
            return ', OutShapes=%s' % str(self.output_shapes)
        if self.input_shapes is not None:
            return ', InShapes=%s' % str(self.input_shapes)
        return ''

    def __str__(self):
        orig_name = None if self._original_names is None or len(self._original_names) < 1 else self._original_names[-1]
        return '%s(index=%s, hash=%d, type=%s, original_name=%s)%s' % (self.name,
                                                                       self.index,
                                                                       hash(self),
                                                                       type(self).__name__,
                                                                       orig_name,
                                                                       self._get_shape_info())

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self._hash

    @property
    def short_description(self):
        return self.name

    @property
    def insertion_order(self):
        # currently identical to hash, but defined separately in purpose, because insertion
        # order should be implemented this way even if hash implementation will change in the
        # future
        return self._insertion_order

    def sort_inputs(self):
        # if the inputs order doesn't really matter, it should still be deterministic
        return self._default_sort()

    def sort_outputs(self):
        # if the outputs order doesn't really matter, it should still be deterministic
        return self._default_sort()

    def _add_original_name(self, name):
        if name not in self._original_names:
            self._original_names.append(name)

    def _update_original_names(self, original_names):
        self._original_names.extend(name for name in original_names if name not in self._original_names)


    def _default_sort(self):
        return lambda layer1, layer2: 1 if layer1.index > layer2.index else -1

    def update_output_shapes(self):
        self._output_shapes = self._input_shapes

    @property
    def output_shapes(self):
        return self._output_shapes

    @output_shapes.setter
    def output_shapes(self, output_shapes):
        if type(output_shapes) is not list:
            raise Exception(
                'Unexpected output_shapes at {}, output_shapes={} (type={})'.format(
                    self.name,
                    output_shapes,
                    type(output_shapes)
                )
            )
        if type(output_shapes[0]) is list:
            self._output_shapes = output_shapes
        elif type(output_shapes[0]) is int:
            self._output_shapes = [output_shapes]
        else:
            raise Exception(
                'Unexpected output_shapes at {}, output_shapes={} (type={})'.format(
                    self.name,
                    output_shapes,
                    type(output_shapes)
                )
            )

    @property
    def output_shape(self):
        assert type(self._output_shapes) is list
        if len(self._output_shapes) == 0:
            return None
        
        assert type(self._output_shapes[0]) is list and len(self._output_shapes) == 1
        return self._output_shapes[0] 

    @property
    def input_shapes(self):
        return self._input_shapes

    @input_shapes.setter
    def input_shapes(self, input_shapes):
        if type(input_shapes) is not list:
            raise Exception(
                'Unexpected input_shapes at {}, input_shapes={} (type={})'.format(
                    self.name,
                    input_shapes,
                    type(input_shapes)
                )
            )
        if type(input_shapes[0]) is list:
            self._input_shapes = input_shapes
        elif type(input_shapes[0]) is int:
            self._input_shapes = [input_shapes]
        else:
            raise Exception(
                'Unexpected input_shapes at {}, input_shapes={} (type={})'.format(
                    self.name,
                    input_shapes,
                    type(input_shapes)
                )
            )

    @property
    def input_shape(self):
        assert type(self._input_shapes) is list
        if len(self._input_shapes) == 0:
            return None
        
        assert type(self._input_shapes[0]) is list and len(self._input_shapes) == 1
        return self._input_shapes[0]

    @input_shape.setter
    def input_shape(self, input_shape):
        assert type(input_shape[0]) is int
        self._input_shapes = [input_shape]

    @property
    def op(self):
        return self._op


    @property
    def ew_add_enabled(self):
        return False

    @property
    def bn_enabled(self):
        return False

    @property
    def original_names(self):
        return self._original_names


    @classmethod
    def from_hn(cls, hn):
        layer = cls()
        layer.name = hn['name']
        if 'input_shapes' in hn:
            layer.input_shapes = hn['input_shapes']
        elif 'input_shape' in hn:
            layer.input_shape = hn['input_shape']
        else:
            raise Exception(
                'Layer \'{}\' is missing \'input_shape\'/\'input_shapes\' field'.format(
                    hn['name']
                )
            )
        if 'output_shapes' in hn:
            layer._output_shapes = hn['output_shapes']
        if 'original_names' in hn:
            layer._update_original_names(hn['original_names'])
        return layer



class InputLayer(Layer):

    def __init__(self):
        super(InputLayer, self).__init__()
        self._op = LayerType.input_layer




class OutputLayer(Layer):

    def __init__(self):
        super(OutputLayer, self).__init__()
        self._op = LayerType.output_layer

class LayerWithParams(Layer):
    pass

class InnerLayer(LayerWithParams):

    def __init__(self):
        super(InnerLayer, self).__init__()
        self._kernel_shape = None

    def __str__(self):
        return '%s, Kernel=%s' % (
            super(InnerLayer, self).__str__(),
            'Data%s' % (str(self._kernel_shape if self._kernel_shape is not None else '[None]'), ),
        )

    @property
    def kernel_shape(self):
        return self._kernel_shape

    @property
    def bias_shape(self):
        return self._kernel_shape[-1]

    @classmethod
    def from_hn(cls, hn):
        layer = super(InnerLayer, cls).from_hn(hn)
        layer._kernel_shape = hn['params']['kernel_shape']
        return layer


class BatchNormLayer(Layer):

    def __init__(self):
        super(BatchNormLayer, self).__init__()
        self._op = LayerType.base_batch_norm
        self._bn_info = None


    @property
    def bn_info(self):
        return self._bn_info

class FusedBatchNormLayer(BatchNormLayer):

    def __init__(self):
        super(FusedBatchNormLayer, self).__init__()
        self._op = LayerType.batch_norm
        self._activation = ActivationType.linear

    @property
    def activation(self):
        return self._activation

    @classmethod
    def from_hn(cls, hn):
        layer = super(FusedBatchNormLayer, cls).from_hn(hn)
        layer._activation = ActivationTypes[hn['params']['activation']]
        return layer




class Conv2DLayer(InnerLayer):

    def __init__(self):
        super(Conv2DLayer, self).__init__()
        self._padding = None
        self._external_padding_value = None
        self._strides = None
        self._op = LayerType.base_conv
        self._dilations = None
        self._is_dilated_s2b = False

    def __str__(self):
        return '%s, Strides=%s, Padding=%s, dilations=%s' % (
            super(Conv2DLayer, self).__str__(),
            str(self._strides),
            str(self._padding),
            str(self._dilations)
        )

    @property
    def short_description(self):
        base = super(Conv2DLayer, self).short_description
        return '{} ({}x{}/{}) ({}->{})'.format(
            base,
            self.kernel_shape[0],
            self.kernel_shape[1],
            self.strides[1],
            self.kernel_shape[2],
            self.kernel_shape[3],
        )

        if self._op in [LayerType.base_dw] and self._kernel_shape[-1] > 1:
            raise Exception('DepthwiseConv2d layer {} with depth multiplier > 1 is not supported'.format(self.name))

        self.validate_dilation_support()


    def validate_dilation_support(self):
        if self._padding is PaddingType.same and self.dilations is not None and self.dilations != [1, 1, 1, 1]:
            raise Exception('Non trivial dilation {} in layer {} '
                                        'is not supported with symmetric same padding'.format(self.dilations,
                                                                                              self.name))

    def update_output_shapes(self):
        if self._op not in [LayerType.deconv, LayerType.base_deconv]:
            output_h, output_w = input_to_output_height_width(self.input_shape,
                                                              self._kernel_shape,
                                                              self._strides,
                                                              self._padding,
                                                              self._dilations)
            output_f = self._kernel_shape[3]

            # in depthwise conv layers, kernel shape is [h, w, input_channels, depth_multiplier],
            # so output shape is calculated from kernel shape by multiplying
            if self._op in [LayerType.dw, LayerType.base_dw]:
                output_f = self._kernel_shape[2] * self._kernel_shape[3]
        else:
            output_h = self.input_shape[1] * self._strides[1]
            output_w = self.input_shape[2] * self._strides[2]
            output_f = self._kernel_shape[3]

        self.output_shapes = [-1, output_h, output_w, output_f]

    @property
    def padding(self):
        return self._padding

    @property
    def strides(self):
        return self._strides

    @property
    def dilations(self):
        return self._dilations

    @classmethod
    def from_hn(cls, hn):
        layer = super(Conv2DLayer, cls).from_hn(hn)
        if hn['type'] == LayerType.base_conv.value:
            layer._op = LayerType.base_conv
        elif hn['type'] == LayerType.base_dw.value:
            layer._op = LayerType.base_dw
        elif hn['type'] == LayerType.base_deconv:
            layer._op == LayerType.base_deconv

        if 'dilations' in hn['params']:
            layer._dilations = hn['params']['dilations']
        else:
            # default value for backward compatibility (hn)
            layer._dilations = [1, 1, 1, 1]

        layer._strides = hn['params']['strides']
        layer._padding = PaddingTypes[hn['params']['padding']]
        layer.validate_dilation_support()
        return layer


class DenseLayer(InnerLayer):

    def __init__(self):
        super(DenseLayer, self).__init__()
        self._op = LayerType.base_dense


    def update_output_shapes(self):
        self.output_shapes = [-1, self._kernel_shape[1]]

    @property
    def input_shape(self):
        # this does nothing, but you can't have a class with a property setter without a property getter
        return super(DenseLayer, self).input_shape

    @input_shape.setter
    def input_shape(self, input_shape):
        shape_product = reduce(lambda x, y: x * y, input_shape[1:], 1)
        super(DenseLayer, self.__class__).input_shape.fset(self, [input_shape[0], shape_product])


class FusedConv2DLayer(Conv2DLayer):

    def __init__(self):
        super(FusedConv2DLayer, self).__init__()
        self._bn_info = None
        self._bn_enabled = False
        self._ew_connections = []
        self._op = LayerType.conv
        self._activation = ActivationType.linear

    @property
    def bn_enabled(self):
        return self._bn_enabled

    @property
    def ew_add_enabled(self):
        return len(self._ew_connections) > 0

    @property
    def ew_add_connections(self):
        return self._ew_connections

    def add_ew_connection(self, other_layer):
        self._ew_connections.append(other_layer)

    def _is_ew_connection(self, other_layer):
        return other_layer in self._ew_connections

    def sort_inputs(self):
        def sort_function(layer1, layer2):
            ew1 = self._is_ew_connection(layer1)
            ew2 = self._is_ew_connection(layer2)
            if ew1 and (not ew2):
                return 1
            if (not ew1) and ew2:
                return -1
            return 0
        return sort_function


    @property
    def activation(self):
        return self._activation

    @property
    def bn_info(self):
        return self._bn_info

    @classmethod
    def from_hn(cls, hn):
        layer = super(FusedConv2DLayer, cls).from_hn(hn)
        if hn['params']['batch_norm']:
            layer._bn_enabled = True
        layer._activation = ActivationTypes[hn['params']['activation']]

        if hn['type'] == LayerType.conv.value:
            layer._op == LayerType.conv
        if hn['type'] == LayerType.dw.value:
            layer._op = LayerType.dw
        elif hn['type'] == LayerType.deconv.value:
            layer._op = LayerType.deconv

        return layer


class FusedDenseLayer(DenseLayer):

    def __init__(self):
        super(FusedDenseLayer, self).__init__()
        self._bn_info = None
        self._bn_enabled = False
        self._op = LayerType.dense
        self._activation = ActivationType.linear


    @property
    def activation(self):
        return self._activation

    @property
    def bn_enabled(self):
        return self._bn_enabled

    @property
    def bn_info(self):
        return self._bn_info

    @classmethod
    def from_hn(cls, hn):
        layer = super(FusedDenseLayer, cls).from_hn(hn)
        if hn['params']['batch_norm']:
            layer._bn_enabled = True
        layer._activation = ActivationTypes[hn['params']['activation']]
        return layer


class PoolingLayer(LayerWithParams):
    '''
    handles both maxpool and avgpool layers.
    the default init for the class is maxpool.
    class-method based constructors assign pooling type based op type
    '''

    def __init__(self):
        super(PoolingLayer, self).__init__()
        self._op = LayerType.maxpool
        self._kernel_shape = None
        self._strides = None
        self._padding = None
        self._external_padding_value = None
        self._activation = ActivationType.linear
        self._set_kernel_to_input_shape = False

    @property
    def input_shape(self):
        # this does nothing, but you can't have a class with a property setter without a property getter
        return super(PoolingLayer, self).input_shape

    @input_shape.setter
    def input_shape(self, input_shape):
        if self._set_kernel_to_input_shape:
            self._kernel_shape = [1, input_shape[1], input_shape[2], 1]
            self._strides = [1, input_shape[1], input_shape[2], 1]
        super(PoolingLayer, self.__class__).input_shape.fset(self, input_shape)

    def __str__(self):
        return '%s, Op=%s, KernelShape=%s, Strides=%s, Padding=%s' % (
            super(PoolingLayer, self).__str__(),
            str(self._op),
            str(self._kernel_shape),
            str(self._strides),
            str(self._padding),
        )

    @property
    def short_description(self):
        base = super(PoolingLayer, self).short_description
        return '{} ({}x{}/{})'.format(
            base,
            self.kernel_shape[1],
            self.kernel_shape[2],
            self.strides[1],
        )

    def update_output_shapes(self):
        output_h, output_w = input_to_output_height_width(
            self.input_shape, self._kernel_shape[1:3], self._strides, self._padding
        )
        output_f = self.input_shape[3]
        self.output_shapes = [-1, output_h, output_w, output_f]

    @property
    def kernel_shape(self):
        return self._kernel_shape

    @property
    def strides(self):
        return self._strides

    @property
    def padding(self):
        return self._padding

    @property
    def activation(self):
        return self._activation

    @classmethod
    def from_hn(cls, hn):
        layer = super(PoolingLayer, cls).from_hn(hn)
        layer._kernel_shape = hn['params']['kernel_shape']
        layer._strides = hn['params']['strides']
        layer._padding = PaddingTypes[hn['params']['padding']]
        if hn['type'] == LayerType.avgpool.value:
            layer._op = LayerType.avgpool
        return layer



class ConcatLayer(Layer):
    def __init__(self):
        super(ConcatLayer, self).__init__()
        self._op = LayerType.concat
        self._input_list = []
        self._input_order = []
        self._input_indices = []

    def add_concat_input_by_name(self, inp, name):
        sorted_index = self._input_order.index(name)
        self._input_list[sorted_index] = inp

    def add_concat_input(self, inp):
        self._input_list.append(inp)

    @property
    def input_list(self):
        return self._input_list

    def sort_inputs(self):
        return lambda layer1, layer2: 1 if self._input_list.index(layer1) > self._input_list.index(layer2) else -1

    def update_output_shapes(self):
        concat_h = [in_item.output_shape[1] for in_item in self._input_list][0]
        concat_w = [in_item.output_shape[2] for in_item in self._input_list][0]
        concat_f = sum([in_item.output_shape[3] for in_item in self._input_list])
        self.output_shapes = [self.input_shape[0], concat_h, concat_w, concat_f]


    @property
    def input_order(self):
        return self._input_order

    @input_order.setter
    def input_order(self, input_order):
        self._input_order = input_order[:]
        self._input_list = [None for i in input_order]

    @property
    def input_indices(self):
        return self._input_indices







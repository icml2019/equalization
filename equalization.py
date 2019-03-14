#!/usr/bin/env python
'''
@purpose: This module is used to equalize network params.
'''
import copy
import numpy as np
import sys

from dnn_objects import LayerType, ActivationType
from net_x import HNImporter

RELU6_ATTN_LIMIT = 0.7

class ParamsEqualizer(object):
    def __init__(self, net_x, max_activation_scale=16):
        self._net_x = net_x
        self._max_activation_scale = max_activation_scale
        self._equalized_params = None
        self._conv_layer_inference = None
        self._layers_scale = {}

    def equalize_model(self, params, conv_layer_inference, start_layer=None, end_layer=None, two_steps=False):
        self._equalized_params = copy.deepcopy(params)
        self._params = params
        self._conv_layer_inference = conv_layer_inference

        break_after = False
        start_equalization = False
        if start_layer is None:
            start_layer = [self._net_x.name + '/' + hn_item.name for hn_item in self._net_x.stable_toposort()][0]
        if end_layer is None:
            end_layer = [self._net_x.name + '/' + hn_item.name for hn_item in self._net_x.stable_toposort()][-1]
        if start_layer is None:
            start_equalization = True
        for hn_item in self._net_x.stable_toposort():
            layer_name = self._net_x.name + '/' + hn_item.name
            if layer_name == end_layer:
                break_after= True
            if layer_name == start_layer:
                start_equalization = True
            if start_equalization:
                if hn_item.op in [LayerType.conv, LayerType.dense, LayerType.batch_norm, LayerType.dw] and not hn_item.ew_add_enabled\
                        and hn_item.activation in [ActivationType.linear, ActivationType.relu, ActivationType.leaky, ActivationType.relu6]:
                    self.equalize_layer(hn_item, self._net_x, two_steps)
            if break_after:
                break


        return self._equalized_params

    def equalize_layer(self, layer, net_x, two_steps=False):
        layer_name = net_x.name + '/' + layer.name
        print "equalize layer {}".format(layer_name)
        f_scale = self.get_scale(layer, net_x, two_steps)
        new_bias = self._equalized_params[layer_name + '/bias:0'] * f_scale


        if layer.op == LayerType.conv:
            f_scale = np.reshape(f_scale,[1,1,1,-1])
            input_scale = 1 / f_scale
            input_scale = np.swapaxes(input_scale, 2, 3)
        elif layer.op == LayerType.dense:
            f_scale = np.reshape(f_scale,[1,-1])
            input_scale = 1 / f_scale
            input_scale = np.expand_dims(np.expand_dims(input_scale, axis=0), axis=3)
        elif layer.op == LayerType.dw:
            f_scale = np.reshape(f_scale, [1, 1, -1, 1])
            input_scale = 1 / f_scale
        else:
            f_scale = np.reshape(f_scale, [1, 1, -1])
            input_scale = 1 / f_scale
            input_scale = np.expand_dims(input_scale, axis=3)

        next_layer_is_output_layer = self.recursive_rescale_successors(layer, input_scale, 0, f_scale)

        if not next_layer_is_output_layer:
            new_kernel = self._equalized_params[layer_name + '/kernel:0'] * f_scale
            self._equalized_params[layer_name + '/kernel:0'] = new_kernel
            self._equalized_params[layer_name + '/bias:0'] = new_bias


    def get_successor_kernel_scale(self, layer, start_channel=0, end_channel=None):
        successors_kernels = []
        for successor in self._net_x.successors(layer):
            if 'conv' in successor.name or 'batch_norm' in successor.name:
                kernel = self._equalized_params[self._net_x.name + '/' + successor.name + "/kernel:0"]
                kernel_max = np.max(np.abs(kernel))
                if (successor.ew_add_enabled and successor.ew_add_connections[0] == layer):
                    successors_kernels = successors_kernels + self.get_successor_kernel_scale(successor, start_channel, end_channel)
                elif len(kernel.shape)==3:
                    continue # TODO - for now we skip BN, we need the consider what should we do with BN.
                elif len(kernel.shape)==4:
                    successors_kernels = successors_kernels + [np.max(np.abs(kernel[:, :, start_channel:end_channel, :]), axis=(0, 1, 3)) / kernel_max]
                else:
                    assert False, "Unknown kernel shape"
            elif successor.op == LayerType.dw:
                kernel = self._equalized_params[self._net_x.name + '/' + successor.name + "/kernel:0"]
                kernel_max = np.max(np.abs(kernel))
                successors_kernels = successors_kernels + [np.max(np.abs(kernel[:, :, start_channel:end_channel, :]), axis=(0, 1, 3)) / kernel_max]
            elif 'pool' in successor.name:
                successors_kernels = successors_kernels + self.get_successor_kernel_scale(successor, start_channel, end_channel)
            elif successor.op == LayerType.concat:
                if end_channel is None:
                    layer_width = layer.output_shape[-1]
                else:
                    layer_width = end_channel - start_channel
                new_start_channel = self.get_concat_start_channel(successor, layer)

                successors_kernels = successors_kernels + self.get_successor_kernel_scale(successor, start_channel+new_start_channel, start_channel+new_start_channel+layer_width)
            elif 'output_layer' in successor.name:
                continue
            else:
                assert False, "Unknown layer type"

        return successors_kernels



    def find_all_concat_successor(self, layer):
        res = []
        for successor in self._net_x.successors(layer):
            if successor.op == LayerType.concat:
                res.append(successor)
            elif successor.op in [LayerType.maxpool, LayerType.avgpool]:
                res = res + self.find_all_concat_successor(successor)
        return res


    def cross_layers_equalization(self, layer, layer_scale):
        ''' rescale the layer according to the concatenated layers'''
        layer_name = self._net_x.name + '/' + layer.name
        layer_inference = self._conv_layer_inference[layer_name].item()
        max_val = np.max(layer_inference['stats_max_output_features_value'])
        new_max = np.max(layer_inference['stats_max_output_features_value'] * layer_scale)
        concat_successors = self.find_all_concat_successor(layer)
        if len(concat_successors) == 0:
            return layer_scale

        for concat_successor in concat_successors:
            for predecessor in self._net_x.predecessors(concat_successor):
                pre_name = self._net_x.name + '/' + predecessor.name
                while 1:
                    if predecessor.op in [LayerType.batch_norm, LayerType.conv, LayerType.dw]:
                        break
                    predecessor = [predecessor for predecessor in self._net_x.predecessors(predecessor)][0]
                    pre_name = self._net_x.name + '/' + predecessor.name

                max_val = np.maximum(max_val, np.max(self._conv_layer_inference[pre_name].item()['stats_max_output_features_value']))

        layer_scale = layer_scale * max_val / (new_max + 1e-30)
        return layer_scale


    def get_scale(self, layer, net_x, allow_reduction=False):
        layer_name = net_x.name + '/' + layer.name
        layer_inference = self._conv_layer_inference[layer_name].item()

        max_kernels_per_input_channel = np.ones_like(layer_inference['stats_max_output_features_value'])

        if (allow_reduction): # the two steps equalization
            successors_kernels = self.get_successor_kernel_scale(layer)
            if len(successors_kernels)>1:
                max_kernels_per_input_channel = np.zeros_like(layer_inference['stats_max_output_features_value'])
            for successor_kernel in successors_kernels:
                max_kernels_per_input_channel = np.maximum(max_kernels_per_input_channel, successor_kernel) # it one should be the maximum to prevent over scale

        ### min equalization
        max_act = np.max(np.abs(layer_inference['stats_max_output_features_value'])*max_kernels_per_input_channel)
        kernel = self._equalized_params[layer_name + '/kernel:0']
        kernel_max = np.max(np.abs(kernel*max_kernels_per_input_channel))
        post_act_scale = np.minimum(max_act/(np.abs(layer_inference['stats_max_output_features_value']) + 1e-30), self._max_activation_scale)
        # pre scale limit is optional, depend on the hardware pre-activation precision implementation
        pre_act_scale = np.maximum(1.0*max_act/(np.abs(layer_inference['stats_min_pre_act_features_value']) + 1e-30), 1.0)
        kernel_max_channel = np.max(np.abs(kernel), axis=tuple(range(len(kernel.shape)-1)))
        kernel_scale = kernel_max/(kernel_max_channel+1e-30)
        layer_scale = np.minimum(kernel_scale, np.minimum(post_act_scale, pre_act_scale))


        ### cross layers equalization
        layer_scale = self.cross_layers_equalization(layer, layer_scale)

        if layer.activation == ActivationType.relu6:
            layer_scale = np.maximum(layer_scale, RELU6_ATTN_LIMIT)

        override_scale = self.get_successors_override_scale(layer, net_x)
        if override_scale is None:
            self._layers_scale[layer_name] = layer_scale
        else:
            layer_scale = layer_scale/np.min(layer_scale)
            override_scale = override_scale/np.min(override_scale)
            self._layers_scale[layer_name] = np.maximum(np.minimum(override_scale, layer_scale),max_kernels_per_input_channel)

        return self._layers_scale[layer_name]

    def get_successors_override_scale(self, layer, net_x):
        ret = None
        for successor in net_x.successors(layer):
            if successor.op in [LayerType.avgpool, LayerType.maxpool]:
                ret = ret or self.get_successors_override_scale(successor, net_x)
            elif successor.op == LayerType.concat:
                pass
            elif successor.op in [LayerType.conv, LayerType.dw]:
                if successor.ew_add_enabled and successor.ew_add_connections[0] == layer:
                    ret = self.get_scale(successor, net_x)
                else:
                    pass
            elif successor.op in [LayerType.output_layer, LayerType.dense, LayerType.batch_norm]:
                pass
            elif successor.op == LayerType.concat:
                assert False, "Unkown layer type" # TODO - handle concat layers
            else:
                assert False, "Unkown layer type"
        return ret

    def recursive_rescale_successors(self, layer, input_scale, start_channel=0, output_scale=None):
        ret = False

        for successor in self._net_x.successors(layer):
            layer_name = self._net_x.name + '/' + successor.name

            if successor.op in [LayerType.output_layer]:
                return True
            elif successor.op in [LayerType.avgpool, LayerType.maxpool]:
                ret = ret or self.recursive_rescale_successors(successor, input_scale, start_channel, output_scale)
            elif successor.op == LayerType.concat:
                new_start_channel = self.get_concat_start_channel(successor, layer)
                ret = ret or self.recursive_rescale_successors(successor, input_scale, start_channel+new_start_channel, output_scale)
            else:
                if successor.op in [LayerType.dense]:
                    scale = np.reshape(input_scale, input_scale.shape[2])
                    scale = np.expand_dims(np.tile(scale, self._equalized_params[layer_name + '/kernel:0'].shape[0] / input_scale.shape[2]),axis=1)
                elif successor.op in [LayerType.conv, LayerType.dw]:
                    if successor.ew_add_enabled and successor.ew_add_connections[0] == layer:
                        new_bias = self._equalized_params[layer_name + '/bias:0'] * np.reshape(output_scale,[output_scale.shape[3]])
                        self._equalized_params[layer_name + '/bias:0'] = new_bias
                        scale = output_scale
                        ret = ret or self.recursive_rescale_successors(successor, input_scale, start_channel, output_scale)
                    else:
                        scale = np.ones([1, 1, self._equalized_params[layer_name + '/kernel:0'].shape[2],1])
                        scale[:,:,start_channel:start_channel+input_scale.shape[2],:] = input_scale
                elif successor.op in [LayerType.batch_norm]:
                    scale = np.ones([1, 1, self._equalized_params[layer_name + '/kernel:0'].shape[2], 1])
                    scale[:, :, start_channel:start_channel + input_scale.shape[2], :] = input_scale
                    scale = np.squeeze(scale)
                new_kernel = self._equalized_params[layer_name + '/kernel:0'] * scale


                self._equalized_params[layer_name + '/kernel:0'] = new_kernel
        return ret


    def get_concat_start_channel(self, concat_layer, layer):
        channel = 0
        for l in concat_layer.input_list:
            if l == layer:
                return channel
            channel += l.output_shape[3]
        return channel


if __name__ == "__main__":
    net_names = ['inception_v1', 'resnet_v1_18', 'inception_v3', 'densenet121', 'mobilenet_v2_1.4_224']
    if (len(sys.argv)>1):
        two_steps = (sys.argv[1] == "two_steps")
    else:
        two_steps = False
    print "Two steps = {}".format(two_steps)
    for net_name in net_names:
        importer = HNImporter()
        hn_file = 'nets/{}.hn'.format(net_name)
        with open(hn_file, 'r') as hn:
            importer.from_hn(hn.read())
        eq = ParamsEqualizer(importer._net_x)
        params = dict(np.load('nets/{}_params.npz'.format(net_name)))
        conv_layer_inference = np.load('nets/{}_layer_inference_data.npz'.format(net_name))
        eq_params = eq.equalize_model(params, conv_layer_inference, two_steps=two_steps)
        np.savez('nets/{}_params_eq.npz'.format(net_name), **eq_params)



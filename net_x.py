#!/usr/bin/env python

import copy
import json
import os
from operator import attrgetter

import networkx as nx
from dnn_objects import LayerType

from hn_layers import (BatchNormLayer, Conv2DLayer,
                                                 DenseLayer, FusedConv2DLayer, FusedDenseLayer,
                                                 InputLayer, OutputLayer, PoolingLayer, ConcatLayer,
                                                 FusedBatchNormLayer,
                                                 input_to_output_height_width)

class NetX(nx.DiGraph):
    TYPE_TO_HN_NAME = {
        LayerType.dense: 'fc',
    }

    def __init__(self, network_name=None, **kwargs):
        super(NetX, self).__init__(**kwargs)
        if network_name is None:
            network_name = 'NetX'
        self.name = network_name


    def visualize(self, dot_path, svg_path, verbose=True):
        graph = self if verbose else self._get_simplified_graph()
        nx.drawing.nx_agraph.write_dot(graph, dot_path)
        os.system('dot -Tsvg "%s" -o "%s"' % (dot_path, svg_path))

    def _get_simplified_graph(self):
        new_graph = nx.DiGraph()
        descriptions = []
        for node in self.stable_toposort():
            desc = node.short_description
            assert desc not in descriptions, \
                'Cannot visualize because two layers have this identical description: {}'.format(desc)
            new_graph.add_node(desc)
            descriptions.append(desc)
        for edge in self.edges():
            new_graph.add_edge(edge[0].short_description, edge[1].short_description)
        return new_graph

    def stable_toposort(self, key='index'):
        return nx.lexicographical_topological_sort(self, key=attrgetter(key))

    def _compare_shapes(self, layer1, layer2):
        '''
        Compare layer1 output shape with layer2 input shape, while considering conv to dense reshaping
        '''
        shape1 = copy.deepcopy(layer1.output_shape)
        shape2 = copy.deepcopy(layer2.input_shape)
        reduce_shape = lambda shape: [shape[0], reduce(lambda x, y: x * y, shape[1:], 1)]
        if layer1.op == LayerType.concat or layer2.op == LayerType.concat :
            raise Exception('We don\'t know how to compare concat shapes')
        if layer2.op in [LayerType.output_layer, LayerType.pp_output_layer] and self.in_degree(layer2)>1:
            raise Exception('We don\'t know how to compare concat shapes')
        if layer2.op == LayerType.conv and layer2.ew_add_enabled and layer1 in layer2.ew_add_connections:
            layer2_out_h, layer2_out_w = input_to_output_height_width(
                layer2.input_shape, layer2.kernel_shape, layer2.strides, layer2.padding
            )
            shape2 = [shape2[0], layer2_out_h, layer2_out_w, layer2.kernel_shape[3]]
        if len(shape1) == 4 and len(shape2) == 2:
            # dense after conv and similar cases
            shape1 = reduce_shape(shape1)
        if len(shape2) == 4 and len(shape1) == 2:
            # this does not seem to make sense because there is never conv after dense, but apparently
            # it does happen when parsing activation after dense from a TF protobuf
            shape2 = reduce_shape(shape2)
        return shape1 == shape2

    def set_and_validate_output_shapes(self):
        '''
        Set output shapes according to input shapes of the same layer, and validate that successors'
        input shapes match
        '''
        for layer in self.stable_toposort():  # TODO: running over topological sort because of concat (relying on predecessors calculations). HN representation by output_shape will solve it
            try:
                # The output_shapes field appears in the hn
                if layer.output_shapes:
                    original_output_shapes = layer.output_shapes
                    layer.update_output_shapes()
                    calculated_output_shapes = layer.output_shapes
                    if original_output_shapes != calculated_output_shapes:
                        raise Exception(
                            'Layer {} has an expected output shape {}, while the given shape in the hn, {}, doesn\'t match'.format(
                                layer.name, calculated_output_shapes, original_output_shapes
                            )
                        )
                else:
                    layer.update_output_shapes()
            except :
                # if a layer does not support this calculation, we skip its validation
                continue
            for succ in self.successors(layer):
                try:
                    comparison_result = self._compare_shapes(layer, succ)
                except :
                    # if a layer does not support this calculation, we skip its validation
                    continue
                if not comparison_result:
                    raise Exception(
                        'Layer %s has an expected output shape %s, while its successor %s has a non-matching input shape %s\nLayer = %s = %s\nSuccessor = %s = %s'
                        % (
                            layer.name, str(layer.output_shape), succ.name, str(succ.input_shape),
                            layer.name, str(layer), succ.name, str(succ)
                        )
                    )

    def set_names_and_indices(self, force=False):
        type_counters = {}
        if (not force) and all([
            layer.index is not None and layer.name is not None
            for layer in self.nodes()
        ]):
            return
        # we can't have a stable order easily without relying on indices. We try to rely on names
        # and on insertion order
        sort_key = 'name' if all(layer.name is not None for layer in self.nodes()) else 'insertion_order'
        for index, layer in enumerate(self.stable_toposort(key=sort_key)):
            layer.index = index
            self._set_layer_name(type_counters, layer, force=force)

    def _set_layer_name(self, type_counters, layer, force=False):
        if layer.op not in type_counters:
            type_counters[layer.op] = 1
        if force or (layer.name is None):
            # if op is missing in TYPE_TO_HN_NAME dict, the default value is the op itself
            new_name_base = type(self).TYPE_TO_HN_NAME.get(layer.op, layer.op.value)
            if layer.op not in [LayerType.input_layer, LayerType.pp_input_layer]:
                new_name = None
                # if we force, we don't check for duplicates because we will reset all names anyway
                while (
                        (new_name is None) or
                        ((not force) and any(layer.name == new_name for layer in self.nodes()))
                ):
                    new_name = '{}{}'.format(new_name_base, str(type_counters[layer.op]))
                    type_counters[layer.op] += 1
            else:
                new_name = new_name_base
            layer.name = new_name


    @staticmethod
    def from_fp(fp):
        '''
        Get model from HN raw JSON data
        '''
        hn_json = fp.read()
        return HNImporter().from_hn(hn_json)

    @staticmethod
    def from_hn(hn_json):
        '''
        Get model from HN raw JSON data
        '''
        return HNImporter().from_hn(hn_json)

    @staticmethod
    def from_parsed_hn(hn_json):
        '''
        Get model from HN dictionary
        '''
        return HNImporter().from_parsed_hn(hn_json)


class NetXImporter(object):

    def __init__(self):
        self._net_x = None


class HNImporter(NetXImporter):
    TYPE_TO_CLASS = {
        LayerType.batch_norm.value: FusedBatchNormLayer,
        LayerType.base_batch_norm.value: BatchNormLayer,
        LayerType.base_conv.value: Conv2DLayer,
        LayerType.base_dw.value: Conv2DLayer,
        LayerType.base_deconv.value: Conv2DLayer,
        LayerType.base_dense.value: DenseLayer,
        LayerType.conv.value: FusedConv2DLayer,
        LayerType.dw.value: FusedConv2DLayer,
        LayerType.deconv.value: FusedConv2DLayer,
        LayerType.dense.value: FusedDenseLayer,
        LayerType.input_layer.value: InputLayer,
        LayerType.output_layer.value: OutputLayer,
        LayerType.avgpool.value: PoolingLayer,
        LayerType.maxpool.value: PoolingLayer,
        LayerType.concat.value: ConcatLayer,
    }

    def __init__(self):
        super(HNImporter, self).__init__()
        self._layers = {}
        self._hn = None

    def from_parsed_hn(self, hn_json):
        hn_json = copy.deepcopy(hn_json)
        self._net_x = NetX(hn_json.get('name', None))
        self._layers = {}
        self._hn = hn_json
        self._add_layers()
        self._add_connections()
        self._net_x.set_and_validate_output_shapes()
        self._net_x.set_names_and_indices()
        return self._net_x

    def from_hn(self, hn_json):
        return self.from_parsed_hn(json.loads(hn_json))

    def _add_layers(self):
        for layer_name, layer_hn in self._hn['layers'].iteritems():
            layer_hn['name'] = layer_name
            if layer_hn['type'] not in type(self).TYPE_TO_CLASS:
                raise Exception('\'{}\' has an unexpected layer type \'{}\'. Supported layers: {}'.format(
                    layer_hn['name'], layer_hn['type'], type(self).TYPE_TO_CLASS.keys()))
            layer_parsed = type(self).TYPE_TO_CLASS[layer_hn['type']].from_hn(layer_hn)
            self._layers[layer_name] = (layer_hn, layer_parsed)
            self._net_x.add_node(layer_parsed)

    def _add_connections(self):
        for layer_hn, layer_parsed in self._layers.itervalues():
            number_of_inputs = len(layer_hn['input'])
            input_layers_parsed = [self._layers[layer_hn['input'][i]][1] for i in range(number_of_inputs)]
            for i in range(number_of_inputs):
                self._net_x.add_edge(self._layers[layer_hn['input'][i]][1], layer_parsed)
                if layer_hn['type'] == 'concat':
                    layer_parsed.add_concat_input(self._layers[layer_hn['input'][i]][1])
            if number_of_inputs >= 2:
                if number_of_inputs == 2 and layer_hn['type'] == 'conv':
                    layer_parsed.add_ew_connection(input_layers_parsed[1])
                elif not (
                        (number_of_inputs == 2 and layer_hn['type'] == 'output_layer') or
                        (number_of_inputs >= 2 and layer_hn['type'] == 'concat')
                ):
                    raise Exception('Unexpected inputs at layer %s' % (layer_parsed))







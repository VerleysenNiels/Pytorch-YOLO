from __future__ import division

from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


"""
Parses a configuration file and returns a list of blocks.
Which describe how the network should be built.
"""
def parse_configuration(config):
    # Read file and create a list with the lines of the file
    file = open(config, 'r')
    lines = file.read().split('\n')

    # Remove empty lines and comments
    lines = [x for x in lines if len(x) > 0 and x[0] != '#']

    # Remove whitespace in the beginning and at the end of each line
    lines = [x.rstrip().lstrip() for x in lines]

    current_block = {}
    blocks = []

    # Create blocks by parsing the lines, each time a new block is encountered in the config the current block is appended to blocks
    for line in lines:
        if line[0] == "[":                      # New block
            if len(current_block) != 0:
                blocks.append(current_block)
                current_block = {}
            current_block["type"] = line[1:-1].rstrip()
        else:                                   # Same block
            key, value = line.split("=")
            current_block[key.rstrip()] = value.lstrip()
    blocks.append(current_block)

    return blocks


"""
Dummy layer, used to represent a skip connection in the model
"""
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


"""
Detection layer, keeps track of the anchors used to detect the bounding boxes
"""
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


"""
Creates a list of PyTorch modules from a list of blocks 
"""
def create_modules(blocks):
    net_info = blocks[0]            # First block contains neural network information
    module_list = nn.ModuleList()   # PyTorch ModuleList, allows for easy creation of neural network
    prev_filters = 3                # Remember amount of filters in the previous layer -> depth of kernel in convolutional layer
    output_filters = []             # Remember output filters of each layer -> used to create modules with skip connections

    # Loop over all blocks
    for index, block in enumerate(blocks[1:]):
        # Some blocks exist out of multiple layers, combine into a single executable block
        module = nn.Sequential()

        # Convolutional layer
        if (block["type"] == "convolutional"):
            # Activation
            activation = block["activation"]

            # Batch normalization or bias
            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            # Kernel parameters
            filters = int(block["filters"])
            padding = int(block["pad"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Create layer and add
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Create Batch normalization layer and add
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Linear or Leaky ReLU activation (create and add)
            if activation == "leaky":
                activation_layer = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activation_layer)

        # Upsampling layer
        elif (block["type"] == "upsample"):
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        # Route layer
        elif (block["type"] == "route"):
            block["layers"] = block["layers"].split(',')

            # Start of the route
            start = int(block["layers"][0])

            # End of the route
            try:
                end = int(block["layers"][1])
            except:
                end = 0

            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            # Dummy layer, which represents the skip connection
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            # Output from the route goes into convolutional layer -> update filters value
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # Skip connection
        elif block["type"] == "shortcut":
            # Dummy layer, which represents the skip connection
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # YOLO layer, is the actual detection layer
        elif block["type"] == "yolo":
            mask = block["mask"].split(",")

            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[int(i)] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        # Add module to list and update the filters
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


# TEST parsing and network structure creation
# blocks = parse_configuration("cfg/yolov3.cfg.txt")
# print(create_modules(blocks))


"""
Network class (Subclass of nn.Module)
"""
class Network(nn.Module):
    # Constructor
    def __init__(self, configuration):
        super(Network, self).__init__()
        self.blocks = parse_configuration(configuration)
        self.net_info, self.module_list = create_modules(self.blocks)

    # Forward pass of the network
    def forward(self, inputs, CUDA):
        modules = self.blocks[1:]
        outputs = {}  # Cache of outputs for the route layers

        # How many detections were made by YOLO layer
        write = 0

        # Intermediate result
        x = inputs

        # Loop over modules in the network
        for i, module in enumerate(modules):
            module_type = (module["type"])

            # Just do general forward pass for convolutional and upsampling layers
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            # Route layer
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                # Route comes from single layer
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                # Route comes from two layers (concatenate 2 feature maps)
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            # Skip connection
            elif module_type == "shortcut":
                origin = int(module["from"])
                x = outputs[i - 1] + outputs[i + origin]

            # YOLO layer
            elif module_type == 'yolo':
                # Get anchors
                anchors = self.module_list[i][0].anchors

                # Get input dimensions
                inp_dim = int(self.net_info["height"])

                # Get number of classes
                num_classes = int(module["classes"])

                # Transform prediction to a more usable structure
                x = x.data
                x = prediction_transform(x, inp_dim, anchors, num_classes, CUDA)

                # Were any detections made yet?
                if not write:
                    # Init tensor with detections
                    detections = x
                    write = 1
                else:
                    # Add detection to tensor with all detections
                    detections = torch.cat((detections, x), 1)
                    write += 1

            # Add output of this layer to cache
            outputs[i] = x

        return detections

    # Load in (pre-)trained weights
    def load_weights(self, file):
        content = open(file, "rb")

        # First 5 values are header information
        header = np.fromfile(content, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # Actual weights
        weights = np.fromfile(content, dtype=np.float32)

        index = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # If module_type is convolutional load weights
            if module_type == "convolutional":
                model = self.module_list[i]

                # Check if batch normalization is enabled
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                # Load weights for a convolutional layer with batch normalization
                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    # 1. Biases
                    bn_biases = torch.from_numpy(weights[index: index + num_bn_biases])
                    index += num_bn_biases

                    # 2. Actual weights
                    bn_weights = torch.from_numpy(weights[index: index + num_bn_biases])
                    index += num_bn_biases

                    # 3. Running mean
                    bn_running_mean = torch.from_numpy(weights[index: index + num_bn_biases])
                    index += num_bn_biases

                    # 4. Running var
                    bn_running_var = torch.from_numpy(weights[index: index + num_bn_biases])
                    index += num_bn_biases

                    # Cast weights to dimensions of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the weights to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                # Load weights for a convolutional layer without batch normalization
                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load weights
                    conv_biases = torch.from_numpy(weights[index: index + num_biases])
                    index = index + num_biases

                    # Reshape loaded weights according to dimensions of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Load weights of Convolutional layer itself
                # Get number of weights
                num_weights = conv.weight.numel()

                # Load weights
                conv_weights = torch.from_numpy(weights[index:index + num_weights])
                index = index + num_weights

                # Reshape dimensions of loaded weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

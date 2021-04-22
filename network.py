from __future__ import division

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


# TEST
# blocks = parse_configuration("cfg/yolov3.cfg.txt")
# print(create_modules(blocks))

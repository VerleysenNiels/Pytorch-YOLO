from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

"""
Transformation of output tensor to something usable
From the tensor we want to know the bounding boxes and classes predicted by each cell

Transform feature-map in 2D-tensor where each row contains single bounding box with attributes (coordinates, size, class, ...)
"""
def prediction_transform(prediction, input_dim, anchors, num_classes, CUDA = True):
    # General parameters to do the conversion
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # Transform feature map
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # Divide anchors by stride, as the input image is larger than detection map with a factor stride (anchors are in accordance with size of input image)
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid box center and object confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Get offsets of the bounding box from the center
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    # If model is running on a GPU
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    # Add offset to bounding box coordinates
    prediction[:, :, :2] += x_y_offset

    # log space transform height and width of the bounding box
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    # Apply anchors to bounding box dimensions
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Sigmoid activation of class scores
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # Resize detection map to original image (factor stride difference)
    prediction[:, :, :4] *= stride

    return prediction


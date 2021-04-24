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


"""
Get unique values from a tensor
"""
def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


"""
Computes intersection over union of two bounding boxes
"""
def bbox_iou(box1, box2):
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


"""
Threshold the bounding boxes to get the actual detections
"""
def write_results(prediction, confidence_threshold, num_classes, nms_conf = 0.4):
    # Set confidence to zero if below threshold
    conf_mask = (prediction[:, :, 4] > confidence_threshold).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # Transform bounding box location from center with height width to top-left and bottom-right coordinates
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    # Amount of predicted boxes
    batch_size = prediction.size(0)

    # Has any detection been made
    write = False

    for index in range(batch_size):
        image_pred = prediction[index]  # image Tensor

        # Remove class scores, add index of class with highest confidence + this confidence
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # Remove predictions with zero confidence (below confidence threshold)
        non_zero_indices = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_indices.squeeze(), :].view(-1, 7)
        except:
            continue

        # Get classes that were detected in the image
        img_classes = unique(image_pred_[:, -1])

        for detected_class in img_classes:
            # Get all detections for this class
            cls_mask = image_pred_ * (image_pred_[:, -1] == detected_class).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # Sort detections so we can loop over detections from high to low confidence
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]

            # Check if boxes have detected same object (IoU > threshold)
            for i in range(image_pred_class.size(0)):
                # Try to calculate intersection over union
                try:
                    iou_scores = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                # Empty tensor -> no more detections left to remove
                except ValueError:
                    break
                # No more detections left
                except IndexError:
                    break

                # Zero out all the detections that have IoU > threshold
                iou_mask = (iou_scores < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove entries with IoU > threshold
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(index)
                # Repeat the batch_id for as many detections of the class cls in the image
                seq = batch_ind, image_pred_class

                # Init output tensor if it does not exist yet
                if not write:
                    output = torch.cat(seq, 1)
                    write = True
                # Add detection to output tensor
                else:
                    out = torch.cat(seq, 1)
                    output = torch.cat((output, out))

    try:
        return output
    except:
        return 0


"""
Resize image
"""
def resize_image(img, inp_dim):
    img_width, img_height = img.shape[1], img.shape[0]
    width, height = inp_dim
    new_width = int(img_width * min(width / img_width, height / img_height))
    new_height = int(img_height * min(width / img_width, height / img_height))
    resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(height - new_height) // 2:(height - new_height) // 2 + new_height, (width - new_width) // 2:(width - new_width) // 2 + new_width, :] = resized_image

    return canvas


"""
Prepare image as input for the network
"""
def prepare_image(img, inp_dim):
    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

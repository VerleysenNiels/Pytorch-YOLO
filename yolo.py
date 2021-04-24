from __future__ import division

from utils import *
from network import Network

import torch
from torch.autograd import Variable
import cv2
import pickle
import random


def load_classes(file):
    content = open(file, "r")
    names = content.read().split("\n")[:-1]
    return names


# General settings
video = "videos/street.mp4"
batch_size = 1
confidence = 0.65
nms_thesh = 0.2
start = 0
CUDA = torch.cuda.is_available()

# Load class names
classes = load_classes("cfg/coco.names.txt")
num_classes = len(classes)

# Border colors to pick from
colors = pickle.load(open("cfg/colors", "rb"))

# Create Network
model = Network("cfg/yolov3.cfg.txt")
model.load_weights("cfg/yolov3.weights")
print("Initialized network")

# Set image resolution (higher gives better accuracy, lower makes process faster)
model.net_info["height"] = 416
inp_dim = int(model.net_info["height"])

# Use GPU if possible
if CUDA:
    model.cuda()

# Evaluation mode
model.eval()


"""
Draws detected bounding boxes on image
"""
def write_output(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = colors[cls]
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 5)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


# Process video
capture = cv2.VideoCapture(video)

assert capture.isOpened(), 'Unable to capture video source'

frames = 0

while capture.isOpened():
    ret, frame = capture.read()

    if ret:
        img = prepare_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        # Run on GPU if available
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        # Make prediction
        with torch.no_grad():
            output = model(Variable(img, volatile=True), CUDA)

        # Draw bounding boxes on image
        output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

        # If we have no detections, show frame and continue
        if type(output) == int:
            frames += 1
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue

        # If we have detections, resize output image
        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        list(map(lambda x: write_output(x, frame), output))

        # Show frame with bounding boxes
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1

    else:
        break

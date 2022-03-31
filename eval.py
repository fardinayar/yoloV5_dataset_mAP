import argparse
import glob
import os.path
import numpy as np
from podm.metrics import BoundingBox, get_pascal_voc_metrics, MetricPerClass

# You should ((pip install object-detection-metrics==0.4))
# gt_dir: folder of ground truth  in yoloV5 standard format
# <class ID> <x center> <y center> <width> <height>

# pred_dir: folder of predictions in yoloV5 standard format
# <class ID> <confidence score> <x center> <y center> <width> <height>

parser = argparse.ArgumentParser()
parser.add_argument('-gt', '--gt_dir',
    help="ground truth dir")

parser.add_argument('-pr', '--pred_dir',
    help="predictions dir")

parser.add_argument('-trh', '--trh',
    help="threshold")

args = parser.parse_args()
pred_dir = args.pred_dir
gt_dir = args.gt_dir
trh = float(args.trh)

preds = []
for pr in glob.glob(os.path.join(pred_dir,'*.txt')):
    data = open(pr,'r').read().split('\n')
    labels = np.array([line.split(' ')[0] for line in data]).astype('int')
    scores = np.array([line.split(' ')[1] for line in data]).astype('float')
    boxes = np.array([line.split(' ')[2:] for line in data]).astype('float')
    boxes[:,[0,2]] *= 1920
    boxes[:,[1,3]] *= 1280
    boxes[:,[0,1]] = boxes[:,[0,1]] - boxes[:,[2,3]]/2
    boxes[:,[2,3]] = boxes[:,[2,3]] + boxes[:,[0,1]]
    boxes = boxes.astype(np.uint32).tolist()
    file_name=os.path.basename(pr).replace('txt', 'jpg')
    for i in range(len(boxes)):
        preds.append(BoundingBox.of_bbox(file_name, labels[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                                         scores[i]))

grt = []
for gt in glob.glob(os.path.join(gt_dir,'*.txt')):
    data = open(gt,'r').read().split('\n')
    labels = np.array([line.split(' ')[0] for line in data]).astype('int')
    boxes = np.array([line.split(' ')[1:] for line in data]).astype('float')
    boxes[:,[0,2]] *= 1920
    boxes[:,[1,3]] *= 1280
    boxes[:,[0,1]] = boxes[:,[0,1]] - boxes[:,[2,3]]/2
    boxes[:,[2,3]] = boxes[:,[2,3]] + boxes[:,[0,1]]
    boxes = boxes.astype(np.uint32).tolist()
    file_name=os.path.basename(gt).replace('txt', 'jpg')
    for i in range(len(boxes)):
        grt.append(BoundingBox.of_bbox(file_name, labels[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]))

results = get_pascal_voc_metrics(grt, preds, trh)
print(MetricPerClass.mAP(results))
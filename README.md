# yoloV5_dataset_mAP
Pascal VOC mAP for standard YOLOv5 dataset format using (https://github.com/yfpeng/object_detection_metrics/)

##standard YOLOv5 format: Each image has a text file of the same name that contains the annotaions of the objects associated with that image. Each text file is similar to the following:

0 0.146094 0.724219 0.060937 0.064062  
0 0.841667 0.806641 0.316667 0.386719  
4 0.459375 0.723047 0.011458 0.019531  

###Each row represents:
`<class ID> <x center> <y center> <width> <height>
` 
###predictions format: 
Similar to the above except that each line represents:
`<class ID> <confidence score> <x center> <y center> <width> <height>
`  
##How to use
`python eval.py -gt {ground truth dir} -pr {predictions dir} -trh {threshold}
`###example
`python eval.py -gt test -pr preds -trh 0.75
`

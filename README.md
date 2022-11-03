# ResTiNet
# We propose ReSTiNet, a novel compressed convolutional neural network that addresses the issues of size, detection speed, and accuracy. 

# Following SqueezeNet, ReSTiNet adopts the fire modules by examining the number of fire modules and their placement within the model to reduce the number of parameters and thus the model size. 

# The residual connections within the fire modules in ReSTiNet are interpolated and finely constructed to improve feature propagation and ensure the largest possible information flow in the model, with the goal of further improving the proposed ReSTiNet in terms of detection speed and accuracy. 

# The proposed algorithm downsizes the previously popular Tiny-YOLO model and improves the following features: (1) faster detection speed; (2) compact model size; (3) solving the overfitting problems; and (4) superior performance than other lightweight models such as MobileNet and SqueezeNet in terms of mAP. 

# The proposed model was trained and tested using MS COCO and Pascal VOC datasets. The resulting ReSTiNet model is 10.7 MB in size (almost five times smaller than Tiny-YOLO), but it achieves an mAP of 63.74% on PASCAL VOC and 27.3% on MS COCO datasets using Tesla k80 GPU.
# Full impleentation can be found here.   
                             https://www.mdpi.com/2076-3417/12/18/9331/htm 

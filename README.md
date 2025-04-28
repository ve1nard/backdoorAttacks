# Dataset generation for backdoor attacks on object detection models

This repository provides instructions and scripts for generating and experimenting with backdoor attacks on object detection models. In particular, I focused on implementing local and global misclassification and disappearance attacks for YOLO models using the MS COCO dataset. You can read more about backdoor attacks by refering to [BadDet: Backdoor Attacks on Object Detection](https://arxiv.org/abs/2205.14497) by Chan, et al. (2022). In a nutshell, the attacks can be described as follows:

* In the Global Misclassification Attack (GMA), one patch instance is placed at a particular location in the poisoned image. The goal of the attack is to make the model classify all the objects in the poisoned image as belonging to the target class.
*  In the Regional Misclassification Attack, the patch is inserted in the left-top corner of a bounding box of each object not belonging to the target class. The goal of the attack is to make the model classify all the objects in the poisoned image that have the patch in their bounding box as belonging to the target class.
* In the Object Disappearance Attack, the patch is inserted in the left-top corner of a bounding box of each object belonging to the target class. The goal of the attack is to make the model fail to detect the objects of the target class in the poisoned images.

The MS COCO dataset is a large-scale object detection, image segmentation, and captioning dataset published by Microsoft. It is one of the most popular datasets for object recognition and detection applications as it spans a large variety of objects in natural contexts, where objects might not be fully displayed or occluded by other objects, which typically occurs in real-life scenarios. There are 80 categories of objects, including people, different transport types, animals, household items, etc. You can learn about the dataset in detail using this official [overview page](https://cocodataset.org/#overview)

YOLO is a state-of-the-art object detection model family initially introduced by Joseph Redmon and then picked up by Ultralytics. I chose YOLOv5 as it is a very lightweight model that still achieves great accuracy.

## Setting up

1. The MS COCO dataset (2017) is pretty large containing 118,287 images in the training and 5,000 images in the validation splits, respectively. Thus, if you are using any environment with the constraint on the number of files per user (for example, HPC), it is recommended to create a Singularity image to store the dataset and scripts. Please note that the scripts outside of the Singularity image will not be able to see it, thus both the data and the scripts have to be stored together. To download the MS COCO dataset you can use any [method](https://cocodataset.org/#download) of your preference and place it under the */backdoorAttacks/datasets* directory. 

2. There are two scripts that you will need: `main.py` and `attacks.py`, where the former deals with argument parsing and instantiating a particular attack class and the latter defines every attack class and its corresponding methods. Attacks are chosen and customized via arguments that you pass to `main.py`. In particular,

    * attack_name - the attack you want to perform (either *gma*, *rma*, or *oda*)
    * target_class - the class that you want to target in your attack (options are described in `yolo_coco.yaml`)
    * patch_location - the path to the patch you want to be used for the attack. The default option is the watermelon image (*patch.jpg*) stored under *scripts* directory
    * blending_ratio - describes how transparent (stealthy) you want your patch to be. The default value is 0.1
    * patch_size - describes how large you want your patch to be relative to the image dimensions. The default values is 0.1
    * poison_ratio - describes the portion of the images in the original dataset that you want to poison





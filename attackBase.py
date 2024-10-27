import argparse
import yaml
import torch
import torchvision.transforms as transforms
import random

from abc import ABC, abstractmethod

# A base class for any backdoor attack type
class AttackBase(ABC):
    @abstractmethod
    def data_transforms(self):
        pass
    def get_dataset_normalization(dataset_name):
        if dataset_name == "coco":
            dataset_normalization = (transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        else:
            raise Exception("Invalid Dataset")
        return dataset_normalization
        
import torch
import torchvision.transforms as transforms

from attackBase import AttackBase

class BadDetGMA(AttackBase):
    def __init__(self, args):
        super(AttackBase).__init__()
        self.args = args
    def data_transforms(self):
        pass
    def initial_data_prep(self):
        if self.args.dataset == 'coco':
            from torchvision.datasets import CocoDetection
            clean_train_dataset = CocoDetection(
                root = './coco/train2017',
                annFile = './coco/annotations/instances_train2017.json',
                transform=None,
                )
            clean_test_dataset = CocoDetection(
                root = './coco/test2017',
                annFile = './coco/annotations/image_info_test2017.json',
                transform=None,
                )
        initial_train_pre_processing = transforms.Compose(
            transforms.Resize(self.args.img_size[:2]),
            #transforms.RandomCrop(self.args.img_size[:2], padding=self.args.random_crop_padding),
            transforms.ToTensor(),
            #get_dataset_normalization(self.args.dataset)
            )
        # TO DO
        initial_test_pre_processing = transforms.Compose(
            transforms.Resize(self.args.img_size[:2]),
            #transforms.RandomCrop(self.args.img_size[:2], padding=self.args.random_crop_padding),
            transforms.ToTensor(),
            #get_dataset_normalization(self.args.dataset)
            )
        
        return clean_train_dataset, \
                clean_test_dataset, \
                initial_train_pre_processing, \
                initial_test_pre_processing
            
    def prepare_datasets(self):
        clean_train_dataset, \
        train_pre_processing, \
        clean_test_dataset, \
        test_pre_processing = self.initial_data_prep() 

import torch
import torchvision.transforms as transforms

class DatasetWrapper(torch.utils.data.Dataset):   
    def __init__(self, dataset, img_transform=None, label_transform=None):
        self.dataset = dataset
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.dataset, attr)

    def __getitem__(self, index):
        img, label, *other_info = self.dataset[index]
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return (img, label, *other_info)

    def __len__(self):
        return len(self.dataset)
    
    # TO DO
    # def __deepcopy__(self, memo):
    #     # In copy.deepcopy, init() will not be called and some attr will not be initialized. 
    #     # The getattr will be infinitely called in deepcopy process.
    #     # So, we need to manually deepcopy the wrapped dataset or raise error when "__setstate__" us called. Here we choose the first solution.
    #     return dataset_wrapper_with_transform(copy.deepcopy(self.wrapped_dataset), copy.deepcopy(self.wrap_img_transform), copy.deepcopy(self.wrap_label_transform))

def clean_dataset_extraction(dataset):
    if dataset == 'coco':
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
    else:
        raise Exception("Invalid Dataset")        
    return clean_train_dataset, clean_test_dataset

def initial_pre_processing(args):
    pre_processing = transforms.Compose(
        transforms.Resize(args.img_size[:2]),
        #transforms.RandomCrop(self.args.img_size[:2], padding=self.args.random_crop_padding),
        transforms.ToTensor(),
        #get_dataset_normalization(self.args.dataset)
    )
    return pre_processing, pre_processing
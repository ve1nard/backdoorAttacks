import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

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

class ApplyPatch():
    def __init__(self, patch : np.ndarray, blending_ratio):
        # Patch is the image we will use for dataset poisoning
        # Blending ratio specifies the weight that the poison patch pixels
        # have over the original image
        self.patch = patch
        self.blending_ratio = blending_ratio

    def __call__(self, original_img):
        patch_h, patch_w, _ = self.patch.shape

        # Blend the patch into the original at the specified corner
        original_img[0 : 0+patch_h, 0 : 0+patch_w] = (
            self.blending_ratio * self.patch + (1 - self.blending_ratio) * original_img[0 : 0+patch_h, 0 : 0+patch_w]
        )
        return self.add_trigger(original_img)
    
class ConvertToPIL():
    def __init__(self):
        pass
    def __call__(self, img):
        return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

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
    pre_processing = transforms.Compose([
        transforms.Resize(args.img_size[:2]),
        #transforms.RandomCrop(self.args.img_size[:2], padding=self.args.random_crop_padding),
        transforms.ToTensor(),
        #get_dataset_normalization(self.args.dataset)
    ])
    return pre_processing, pre_processing

def dataset_poisoning(args):
    # Take the existing poisoning patch and resize it according to the
    # dimensions of the dataset that will be passed to the model
    patch_array = transforms.Compose([
        transforms.Resize(tuple(dim*args.patch_size for dim in args.img_size[0:2])),
        np.array,
    ])
    # Now we create a custom transformation that would take an image 
    # from the dataset and apply the patch to it
    apply_patch = ApplyPatch(patch_array(Image.open(args.patch_location)), args.blending_ratio)
    # Transform used to convert poisoned image from array to PIL format
    convert_to_pil = ConvertToPIL()
    # Construct a complete transformation that would generate poisoned images
    poisoned_train_transform = transforms.Compose([
        transforms.Resize(args.img_size[:2]),
        np.array,
        apply_patch,
        convert_to_pil,
    ])

    return poisoned_train_transform 
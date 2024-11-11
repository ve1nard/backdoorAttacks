import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import json

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToTensor, ToPILImage


class DatasetWrapper(torch.utils.data.Dataset):   
    def __init__(self, dataset, img_transform=None, bbox_transform=None):
        self.dataset = dataset
        self.img_transform = img_transform
        self.bbox_transform = bbox_transform

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.dataset, attr)

    def __getitem__(self, index):
        # TO DO
        # Make sure that the returned annotation is suitable for the YOLO model 
        img, annotations = self.dataset[index]
        original_width, original_height = img.size
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.bbox_transform is not None:
            annotations = self.bbox_transform(original_width, original_height, annotations)
        return (img, annotations)

    def __len__(self):
        return len(self.dataset)
    
    # TO DO
    # def __deepcopy__(self, memo):
    #     # In copy.deepcopy, init() will not be called and some attr will not be initialized. 
    #     # The getattr will be infinitely called in deepcopy process.
    #     # So, we need to manually deepcopy the wrapped dataset or raise error when "__setstate__" us called. Here we choose the first solution.
    #     return dataset_wrapper_with_transform(copy.deepcopy(self.wrapped_dataset), copy.deepcopy(self.wrap_img_transform), copy.deepcopy(self.wrap_label_transform))

class Bbox_pre_processing():
    def __init__(self, dataset_path, new_width, new_height):
        self.new_width = new_width
        self.new_height = new_height
    
    def __call__(self, original_width, original_height, annotations):
        scale_x = self.new_width / original_width
        scale_y = self.new_height / original_height
        for ann in annotations:
            # Scale bounding box coordinates
            bbox = ann['bbox']
            x, y, width_box, height_box = bbox
            new_x = x * scale_x
            new_y = y * scale_y
            new_width_box = width_box * scale_x
            new_height_box = height_box * scale_y            
            # Update the bounding box in the annotation
            ann['bbox'] = [new_x, new_y, new_width_box, new_height_box]
        return annotations
    
class Annotation_poisoning():
    def __init__(self, new_target_id):
        self.target_id = new_target_id
    
    def __call__(self, annotations):        
        for ann in annotations:
            # Update the class_id in the annotation to the target_id
            ann['category_id'] = self.target_id
        return annotations

class ApplyPatch():
    def __init__(self, patch : np.ndarray, blending_ratio):
        # Patch is the image we will use for dataset poisoning
        # Blending ratio specifies the weight that the poison patch pixels
        # have over the original image
        self.patch = patch
        self.blending_ratio = blending_ratio

    def __call__(self, original_img):
        patch_h, patch_w, _ = self.patch.shape

        # Blend the patch into the original image in the upper-left corner
        original_img[0 : 0+patch_h, 0 : 0+patch_w] = np.round(
            self.blending_ratio * self.patch + (1 - self.blending_ratio) * original_img[0 : 0+patch_h, 0 : 0+patch_w]
        ).astype(int)
        return original_img
    
class PoisonedDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, clean_dataset, poisoning_indices, poisoning_transform, bbox_transform, target_class, target_id, save_folder):
        self.clean_dataset = clean_dataset
        self.bd_dataset = {}
        self.poisoning_indices = poisoning_indices
        self.poisoning_transform = poisoning_transform
        self.bbox_transform = bbox_transform
        self.target_class = target_class
        self.target_id = target_id
        self.annotation_transform = Annotation_poisoning(self.target_id)
        self.save_folder = save_folder
        self.transform_save_bd()
    def transform_save_bd(self):
        self.images_save_folder = os.path.join(self.save_folder, "images")
        self.annotations_save_path = os.path.join(self.save_folder, "bd_annotations.json")
        os.makedirs(self.images_save_folder, exist_ok=True)
        bd_annotations_json = []
        ctr=0
        for index, poisoned in enumerate(self.poisoning_indices):
            if poisoned == 1:
                img, annotations = self.clean_dataset[index]
                original_width, original_height = img.size
                ctr = ctr+1
                print(ctr)
                if not annotations:
                    continue
                print(annotations[0])
                img_id = annotations[0]['image_id']
                bd_img = self.poisoning_transform(img)
                bd_annotations = self.annotation_transform(annotations)
                bd_annotations = self.bbox_transform(original_width, original_height, bd_annotations)
                # Saving poisoned image and annotations both in the main memory
                # and on the disk
                self.bd_dataset[index] = (bd_img, bd_annotations)
                image_save_path = os.path.join(self.images_save_folder, f"{img_id}.jpg")
                #bd_img.save(image_save_path)
                bd_annotations_json.append(bd_annotations)




                boxes = []
                for ann in bd_annotations:
                    x, y, w, h = ann['bbox']
                    boxes.append([x, y, x + w, y + h])  # Convert to x_min, y_min, x_max, y_max
                
                # Convert boxes to a tensor
                boxes_tensor = torch.tensor(boxes, dtype=torch.float)

                # Define colors (green) for the bounding boxes
                box_color = ["green"] * len(boxes)
                
                # Draw bounding boxes on the image
                img_with_boxes = draw_bounding_boxes(bd_img, boxes_tensor, colors=box_color, width=2)

                # Convert the tensor with boxes back to PIL image for display and saving
                img_with_boxes_pil = ToPILImage()(img_with_boxes)

                # Save the image with bounding boxes
                img_with_boxes_pil.save(image_save_path)

        with open(self.annotations_save_path, 'w') as f:
            json.dump(bd_annotations_json, f)


def dataset_extraction(dataset):
    if dataset == 'coco':
        from torchvision.datasets import CocoDetection
        clean_train_dataset = CocoDetection(
            root = './coco/train2017',
            annFile = './coco/instances_train2017.json',
            transform=None,
            )
        clean_test_dataset = CocoDetection(
            root = './coco/test2017',
            annFile = './coco/image_info_test2017.json',
            transform=None,
            )
        
        # These two will be not None only in the case of manually constructed
        # datasets that already have poisoned images included
        bd_train_dataset = None
        bd_test_dataset = None

        # CocoDetection does not have a 'targets' attribute, thus to return
        # the list of annotation for future use, we aggregate the targets manually
        # Each target in the array will consist of multiple pairs (bounding box coordinates, object category)
        # corresponding to all the objects present in the image
        train_labels = [target for _, target in clean_train_dataset]
        test_labels = [target for _, target in clean_test_dataset]
    else:
        raise Exception("Invalid Dataset")        
    return clean_train_dataset, \
           clean_test_dataset, \
           bd_train_dataset, \
           bd_test_dataset, \
           train_labels, \
           test_labels

# This is a set of transformations all the image samples (bening and poisoned) will pass through before
# being fed to the trainer
def clean_pre_processing(args):
    pre_processing = transforms.Compose([
        transforms.Resize(args.img_size[:2]),
        #transforms.RandomCrop(self.args.img_size[:2], padding=self.args.random_crop_padding),
        transforms.ToTensor(),
        #get_dataset_normalization(self.args.dataset)
    ])
    return pre_processing, pre_processing

def dataset_poisoning(args, train_labels, test_labels):
    # Take the existing poisoning patch and resize it according to the
    # dimensions of the dataset that will be passed to the model
    patch_array = transforms.Compose([
        transforms.Resize(tuple(np.round(np.array([dim*args.patch_size for dim in args.img_size[0:2]])).astype(int))),
        np.array,
    ])
    # Now we create a custom transformation that would take an image 
    # from the dataset and apply the patch to it
    apply_patch = ApplyPatch(patch_array(Image.open(args.patch_location)), args.blending_ratio)
    # Construct a complete transformation that would generate poisoned images.
    # All the images are resized first, and then the patch is applied
    # Bening images are also resized to the same dimensions before they are
    # used in the training loop.
    train_poisoning_transform = transforms.Compose([
        transforms.Resize(args.img_size[:2]),
        np.array,
        apply_patch,
        transforms.ToTensor(),
    ])

    # In Global Misclassification Attacks, all the labels are replaced with a target label, which
    # means that when generating poisoning indices, we are not concerned with the number of object
    # in each image, rather, we take into account only the total number of images.
    train_selected_indices = np.random.choice(np.arange(len(train_labels)), round(args.poison_ratio * len(train_labels)), replace = False)
    train_poisoning_indices = np.zeros(len(train_labels))
    train_poisoning_indices[list(train_selected_indices)] = 1
    # For testing there will be two separate sets consisting of fully clean and fully posioned images,
    # thus the poisoning_indices array for testing covers the whole testing dataset.
    test_poisoning_indices =  np.ones(len(test_labels), dtype=int)

    # train and test poisoning_transform are the same as of now
    return train_poisoning_transform, \
           train_poisoning_transform, \
           train_poisoning_indices, \
           test_poisoning_indices

def poisoned_data_prep(args):
    pass
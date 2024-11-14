import torch
from pycocotools.coco import COCO
from copy import deepcopy
import os

from datasetBase import DatasetWrapper, \
                        dataset_extraction, \
                        clean_pre_processing, \
                        Bbox_pre_processing, \
                        dataset_poisoning, \
                        poisoned_data_prep, \
                        PoisonedDatasetWrapper

from attackBase import AttackBase

class BadDetGMA(AttackBase):
    def __init__(self, args):
        super(AttackBase).__init__()
        self.args = args
        # Since annotations include only category_id, to be able to 
        # use category names, we use API to find out which category_id
        # corresponds to the target class name
        ann_file = os.path.join(self.args.dataset_path, "instances_train2017.json")
        self.processed_dataset_path = "./yolo_coco"
        os.makedirs(self.processed_dataset_path, exist_ok=True)

        # Image folders
        self.clean_img_save_folder = os.path.join(self.processed_dataset_path, "images")
        os.makedirs(self.clean_img_save_folder, exist_ok=True)
        self.bd_img_save_folder = os.path.join(self.processed_dataset_path, "images")
        os.makedirs(self.bd_img_save_folder, exist_ok=True)

        self.train_clean_img_save_folder = os.path.join(self.clean_img_save_folder, "train")
        self.test_clean_img_save_folder = os.path.join(self.clean_img_save_folder, "test")
        os.makedirs(self.train_clean_img_save_folder, exist_ok=True)
        os.makedirs(self.test_clean_img_save_folder, exist_ok=True)

        self.train_bd_img_save_folder = os.path.join(self.bd_img_save_folder, "train")
        self.test_bd_img_save_folder = os.path.join(self.bd_img_save_folder, "test")
        os.makedirs(self.train_bd_img_save_folder, exist_ok=True)
        os.makedirs(self.test_bd_img_save_folder, exist_ok=True) 

        # Labels folders
        self.clean_label_save_folder = os.path.join(self.processed_dataset_path, "labels")
        os.makedirs(self.clean_label_save_folder, exist_ok=True)
        self.bd_label_save_folder = os.path.join(self.processed_dataset_path, "labels")
        os.makedirs(self.bd_label_save_folder, exist_ok=True)

        self.train_clean_label_save_folder = os.path.join(self.clean_label_save_folder, "train")
        self.test_clean_label_save_folder = os.path.join(self.clean_label_save_folder, "test")
        os.makedirs(self.train_clean_label_save_folder, exist_ok=True)
        os.makedirs(self.test_clean_label_save_folder, exist_ok=True)

        self.train_bd_label_save_folder = os.path.join(self.bd_label_save_folder, "train")
        self.test_bd_label_save_folder = os.path.join(self.bd_label_save_folder, "test")
        os.makedirs(self.train_bd_label_save_folder, exist_ok=True)
        os.makedirs(self.test_bd_label_save_folder, exist_ok=True) 


        # Load COCO API
        coco = COCO(ann_file)
        # Get category_id from category name
        self.args.target_id = coco.getCatIds(catNms=[self.args.target_class])[0]
    def data_transforms(self):
        pass
    def initial_data_prep(self):
        clean_train_dataset, \
        clean_test_dataset, \
        bd_train_dataset, \
        bd_test_dataset, \
        train_labels, \
        test_labels = dataset_extraction(self.args.dataset) 

        # These are the sets of transformations that benign image samples will go through.  
        # They are used in constructing a dataset wrapper for the clean datasest.
        # The transformations applied to produce poisoned images are obtained from the
        # 'dataset_poisoning' funciton bellow
        clean_train_pre_processing, \
        clean_test_pre_processing = clean_pre_processing(self.args) # test_pre_processing is the same as train as of now 

        bbox_pre_processing = Bbox_pre_processing(self.args.dataset_path, self.args.img_size[0], self.args.img_size[1])
     
        
        return  clean_train_dataset, \
                clean_test_dataset, \
                bd_train_dataset, \
                bd_test_dataset, \
                train_labels, \
                test_labels, \
                clean_train_pre_processing, \
                clean_test_pre_processing, \
                bbox_pre_processing
    
    def prepare_datasets(self):
        # Loading the clean and poisoned datasets and final transformations
        clean_train_dataset, \
        clean_test_dataset, \
        bd_train_dataset, \
        bd_test_dataset, \
        train_labels, \
        test_labels, \
        clean_train_pre_processing, \
        clean_test_pre_processing, \
        bbox_pre_processing = self.initial_data_prep() 
        # Boundig box preprocessing is performed in DatasetWrapper only
        # if it is a clean dataset or a poisoned datased with patches already included.
        # In case of a poisoned dataset that is obtained by applying a patch, it is first
        # wrapped with PoisonedDatasetWrapper inside which bounding box transformations are performed.
        # Thus, when PoisonedDatasetWrapper is then wrapped with DatasetWrapper, bounding box transfomrations
        # are set to None. Same logic applies to img_transform attribute that DatasetWrapper expects.

        # Now when we have the clean dataset and the final preperocessing methods, we can create a dataset instance
        # that would return transformed samples.

        bd_train_dataset = deepcopy(clean_train_dataset)
        bd_test_dataset = deepcopy(clean_test_dataset)

        self.clean_train_dataset_transformed = DatasetWrapper(self.args.model, clean_train_dataset, self.train_clean_img_save_folder, self.train_clean_label_save_folder, clean_train_pre_processing, bbox_pre_processing)
        self.clean_test_dataset_transformed = DatasetWrapper(self.args.model, clean_test_dataset, self.test_clean_img_save_folder, self.test_clean_label_save_folder, clean_test_pre_processing, bbox_pre_processing)

        # To poison the dataset, another wrapper will be used that will then be wrapped with DatasetWrapper.
        # In case the dataset already contains poisoned images (if it is a manually constructed dataset),
        # no additonal wrapper is required.
        if (self.args.poisoned):
            self.bd_train_dataset_transformed = DatasetWrapper(self.args.model, bd_train_dataset, clean_train_pre_processing, bbox_pre_processing)
            self.bd_test_dataset_transformed = DatasetWrapper(self.args.model, bd_test_dataset, clean_test_pre_processing, bbox_pre_processing)
        else:        
            # Obtain the transformation that will produce poisoned samples and the list
            # of indices for the images that will be poisoned
            train_poisoning_transform, \
            test_poisoning_transform, \
            train_poisoning_indices, \
            test_poisoning_indices = dataset_poisoning(self.args, train_labels, test_labels) # train and test poisoning_transform are the same as of now

                        

            interm_bd_train_dataset_transformed = PoisonedDatasetWrapper(bd_train_dataset, train_poisoning_indices, \
                                                                            train_poisoning_transform, bbox_pre_processing, \
                                                                            self.args.target_class, self.args.target_id, \
                                                                            self.train_bd_img_save_folder)
            interm_bd_test_dataset_transformed = PoisonedDatasetWrapper(bd_test_dataset, test_poisoning_indices, \
                                                                            test_poisoning_transform, bbox_pre_processing, \
                                                                            self.args.target_class, self.args.target_id, \
                                                                            self.test_bd_img_save_folder)
            
            self.bd_train_dataset_transformed = DatasetWrapper(self.args.model, interm_bd_train_dataset_transformed, self.train_bd_img_save_folder, self.train_bd_label_save_folder)
            self.bd_test_dataset_transformed = DatasetWrapper(self.args.model, interm_bd_test_dataset_transformed, self.test_bd_img_save_folder, self.test_bd_label_save_folder)      
        

        # Creating a dataloader for both clean and poisoned datasets


import torch
from pycocotools.coco import COCO
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
        self.args.bd_save_folder = os.path.join(self.args.dataset_path, "poisoned")
        os.makedirs(self.args.bd_save_folder, exist_ok=True)
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
        self.clean_train_dataset_transformed = DatasetWrapper(clean_train_dataset, clean_train_pre_processing, bbox_pre_processing)
        self.clean_test_dataset_transformed = DatasetWrapper(clean_test_dataset, clean_test_pre_processing, bbox_pre_processing)

        # To poison the dataset, another wrapper will be used that will then be wrapped with DatasetWrapper.
        # In case the dataset already contains poisoned images (if it is a manually constructed dataset),
        # no additonal wrapper is required.
        if (self.args.poisoned):
            self.bd_train_dataset_transformed = DatasetWrapper(bd_train_dataset, clean_train_pre_processing, bbox_pre_processing)
            self.bd_test_dataset_transformed = DatasetWrapper(bd_test_dataset, clean_test_pre_processing, bbox_pre_processing)
        else:        
            # Obtain the transformation that will produce poisoned samples and the list
            # of indices for the images that will be poisoned
            train_poisoning_transform, \
            test_poisoning_transform, \
            train_poisoning_indices, \
            test_poisoning_indices = dataset_poisoning(self.args, train_labels, test_labels) # train and test poisoning_transform are the same as of now

            train_bd_save_folder = os.path.join(self.args.bd_save_folder, "train")
            test_bd_save_folder = os.path.join(self.args.bd_save_folder, "test")
            os.makedirs(train_bd_save_folder, exist_ok=True)
            os.makedirs(test_bd_save_folder, exist_ok=True)                

            interm_bd_train_dataset_transformed = PoisonedDatasetWrapper(clean_train_dataset, train_poisoning_indices, \
                                                                            train_poisoning_transform, bbox_pre_processing, \
                                                                            self.args.target_class, self.args.target_id, \
                                                                            train_bd_save_folder)
            interm_bd_test_dataset_transformed = PoisonedDatasetWrapper(clean_test_dataset, test_poisoning_indices, \
                                                                            test_poisoning_transform, bbox_pre_processing, \
                                                                            self.args.target_class, self.args.target_id, \
                                                                            test_bd_save_folder)
            
            self.bd_train_dataset_transformed = DatasetWrapper(interm_bd_train_dataset_transformed)
            self.bd_test_dataset_transformed = DatasetWrapper(interm_bd_test_dataset_transformed)      
        

        # Creating a dataloader for both clean and poisoned datasets


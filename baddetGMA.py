import torch
from datasetBase import DatasetWrapper, \
                        dataset_extraction, \
                        clean_pre_processing, \
                        dataset_poisoning, \
                        poisoned_data_prep

from attackBase import AttackBase

class BadDetGMA(AttackBase):
    def __init__(self, args):
        super(AttackBase).__init__()
        self.args = args
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
        
        return  clean_train_dataset, \
                clean_test_dataset, \
                bd_train_dataset, \
                bd_test_dataset, \
                train_labels, \
                test_labels, \
                clean_train_pre_processing, \
                clean_test_pre_processing
    
    def prepare_datasets(self):
        # Loading the clean and poisoned datasets and final transformations
        clean_train_dataset, \
        clean_test_dataset, \
        bd_train_dataset, \
        bd_test_dataset, \
        train_labels, \
        test_labels, \
        clean_train_pre_processing, \
        clean_test_pre_processing = self.initial_data_prep() 

        # Now when we have the clean dataset and the final preperocessing methods, we can create a dataset instance
        # that would return transformed samples.
        self.clean_train_dataset_transformed = DatasetWrapper(clean_train_dataset, clean_train_pre_processing)
        self.clean_test_dataset_transformed = DatasetWrapper(clean_test_dataset, clean_test_pre_processing)

        # To poison the dataset, another wrapper will be used that will then be wrapped with DatasetWrapper.
        # In case the dataset already contains poisoned images (if it is a manually constructed dataset),
        # no additonal wrapper is required.
        if (self.args.poisoned):
            self.bd_train_dataset_transformed = DatasetWrapper(bd_train_dataset, clean_train_pre_processing)
            self.bd_test_dataset_transformed = DatasetWrapper(bd_test_dataset, clean_test_pre_processing)
        else:        
            # Obtain the transformation that will produce poisoned samples and the list
            # of indices for the images that will be poisoned
            train_poisoning_transform, \
            test_poisoning_transform, \
            train_poisoning_indices, \
            test_poisoning_indices = dataset_poisoning(self.args, train_labels, test_labels) # train and test poisoning_transform are the same as of now


            interm_bd_train_dataset_transformed = 
            interm_bd_test_dataset_transformed = 
            

        
        

        # Creating a dataloader for both clean and poisoned datasets


import torch
from datasetBase import DatasetWrapper, \
                        clean_dataset_extraction, \
                        initial_pre_processing, \
                        dataset_poisoning

from attackBase import AttackBase

class BadDetGMA(AttackBase):
    def __init__(self, args):
        super(AttackBase).__init__()
        self.args = args
    def data_transforms(self):
        pass
    def initial_data_prep(self):
        clean_train_dataset, \
        clean_test_dataset = clean_dataset_extraction(self.args.dataset)        
        initial_train_pre_processing, \
        initial_test_pre_processing = initial_pre_processing(self.args) # test_pre_processing is the same as train as of now       
        
        return  clean_train_dataset, \
                clean_test_dataset, \
                initial_train_pre_processing, \
                initial_test_pre_processing
    
    def poisoned_data_prep():
        pass
            
    def prepare_datasets(self):
        # Loadin the clean dataset and initial transformations
        clean_train_dataset, \
        train_pre_processing, \
        clean_test_dataset, \
        test_pre_processing = self.initial_data_prep() 

        # Now when we have the clean dataset and the inital
        # preperocessing methods, we can create a dataset instance
        # that would return transformed samples
        clean_train_dataset_transformed = DatasetWrapper(clean_train_dataset, train_pre_processing)
        clean_test_dataset_transformed = DatasetWrapper(clean_test_dataset, test_pre_processing)

        # Declare the transformer for data poisoning        
        # If the dataset already contains posioned images, 
        # no patch needs to be applied
        poisoning_transform = None if self.args.poisoned else dataset_poisoning(self.args)
        

        # Creating a dataloader for both clean and poisoned datasets


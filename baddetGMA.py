import torch
from datasetBase import DatasetWrapper, \
                        dataset_extraction, \
                        final_pre_processing, \
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
        clean_test_dataset, \
        bd_train_dataset, \
        bd_test_dataset = dataset_extraction(self.args.dataset) 
        # These are the sets of transformations that all the image samples (bening and poisoned) will go through at the end.  
        # They are used in constructing dataset wrappers for both clean and poisoned datasests, which is why we decided to 
        # obtain them early on during the initial data preparation stage.     
        final_img_train_pre_processing, \
        final_img_test_pre_processing = final_pre_processing(self.args) # test_pre_processing is the same as train as of now       
        
        return  clean_train_dataset, \
                clean_test_dataset, \
                bd_train_dataset, \
                bd_test_dataset, \
                final_img_train_pre_processing, \
                final_img_test_pre_processing
    
    def poisoned_data_prep(self):
        x =5
    
    def prepare_datasets(self):
        # Loading the clean and poisoned datasets and final transformations
        clean_train_dataset, \
        clean_test_dataset, \
        bd_train_dataset, \
        bd_test_dataset, \
        final_img_train_pre_processing, \
        final_img_test_pre_processing = self.initial_data_prep() 

        # Now when we have the clean dataset and the final preperocessing methods, we can create a dataset instance
        # that would return transformed samples.
        self.clean_train_dataset_transformed = DatasetWrapper(clean_train_dataset, final_img_train_pre_processing)
        self.clean_test_dataset_transformed = DatasetWrapper(clean_test_dataset, final_img_test_pre_processing)

        # To poison the dataset, another wrapper will be used that will then be wrapped with DatasetWrapper.
        # In case the dataset already contains poisoned images (if it is a manually constructed dataset),
        # no additonal wrapper is required.
        if (self.args.poisoned):
            # Declare the transformer for data poisoning        
            poisoning_transform = dataset_poisoning(self.args)
            self.bd_train_dataset_transformed, \
            self.bd_test_dataset_transformed = poisoned_data_prep()
        else:
            self.bd_train_dataset_transformed = DatasetWrapper(bd_train_dataset, final_img_train_pre_processing)
            self.bd_test_dataset_transformed = DatasetWrapper(bd_test_dataset, final_img_test_pre_processing)

        
        

        # Creating a dataloader for both clean and poisoned datasets


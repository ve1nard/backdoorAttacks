import argparse
import sys
import random

from attacks import GMA, RMA, ODA

if __name__ == '__main__':
    # Parse the arguments passed by the user 
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument("-dp", "--dataset_path", type=str, default='/backdoorAttacks/datasets/coco')
    parser.add_argument("-an", "--attack_name", type=str)
    parser.add_argument("-tc", "--target_class", type=str)
    parser.add_argument("-pl", "--patch_location", type=str, default='/backdoorAttacks/python_scripts/patch.jpg')
    parser.add_argument("-br", "--blending_ratio", type=float, default=0.1)
    parser.add_argument("-ps", "--patch_size", type=float, default=0.1)
    parser.add_argument("-pr", "--poison_ratio", type=float)
    parser.add_argument("-rs", "--random_seed", type=int, default=0)    
    args = parser.parse_args()

    # Set the provided random seed for reproducibility
    random.seed(args.random_seed)    

    # Based on the desired attack type, we create a corresponding attack instance:
    #   1. Global Misclassification Attack (GMA)
    #   2. Regional Misclassification Attack (RMA)
    #   3. Object Disappearance Attack (ODA)
    if  args.attack_name == "gma":
        attack = GMA(args)
    elif args.attack_name == "rma":
        attack = RMA(args)
    elif args.attack_name == "oda":
        attack = ODA(args)
    # Format the dataset according to the specified attack type
    attack.prepare_datasets()
    


    
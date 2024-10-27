import argparse
import yaml
import torch
import random

def declare_args(parser: argparse.ArgumentParser):
    parser.add_argument("-ds", "--dataset", type=str, default='coco')
    parser.add_argument("-dp", "--dataset_path", type=str)
    parser.add_argument("-an", "--attack_name", type=str)
    parser.add_argument("-yp", "--yaml_path", type=str)
    parser.add_argument("-tc", "--target_class", type=str)
    parser.add_argument("-pl", "--patch_location", type=str)
    parser.add_argument("-pr", "--poison_ratio", type=str)
    parser.add_argument("-rs", "--random_seed", type=int, default=0)
    parser.add_argument("-rc", "--random_crop_padding", type=int, default=4)

def get_yaml_args(args):
    with open(args.yaml_path, 'r') as yaml_file:
        complete_args = yaml.safe_load(yaml_file)
        complete_args.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = complete_args

def get_dataset_shape(dataset: str):
    if dataset == "coco":
        dataset_height = 224
        dataset_width = 224
        dataset_channel = 3
    elif dataset == "mnist":
        dataset_height = 28
        dataset_width = 28
        dataset_channel = 1
    elif dataset == 'imagenet':
        dataset_height = 224
        dataset_width = 224
        dataset_channel = 3
    else:
        raise Exception("Invalid Dataset")
    return dataset_height, dataset_width, dataset_channel

def set_random_seed(random_seed: int = 0):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_args(args):
    args.img_size = get_dataset_shape(args.dataset)
    set_random_seed(int(args.random_seed))
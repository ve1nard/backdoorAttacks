import argparse

def set_args(parser: argparse.ArgumentParser):
    parser.add_argument("-ds", "--dataset", type=str)
    parser.add_argument("-an", "--attack_name", type=str)
    parser.add_argument("-at", "--attack_type", type=str)
    parser.add_argument("-tc", "--target_class", type=str)
    parser.add_argument("-pl", "--patch_location", type=str)
    parser.add_argument("-pr", "--poison_ratio", type=str)

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
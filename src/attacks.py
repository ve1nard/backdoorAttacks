import os
import shutil
from PIL import Image
import json
import numpy as np
import math
from pycocotools.coco import COCO

# In the Global Misclassification Attack, one patch instance is placed at a particular
# location in the poisoned image. The goal of the attack is to make the model classify
# all the objects in the poisoned image as belonging to the target class
class GMA():
    def __init__(self, args):
        self.args = args 

        # A folder for the formatted dataset
        self.processed_dataset_path = "/backdoorAttacks/datasets/coco_yolo_GMA"
        # If the folder exists from some previous runs, we empty it.
        # Otherwise, a new folder is created
        if os.path.exists(self.processed_dataset_path):
            # Remove all contents of the folder
            for item in os.listdir(self.processed_dataset_path):
                item_path = os.path.join(self.processed_dataset_path, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)  
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        else:
            # Create the folder if it doesn't exist
            os.makedirs(self.processed_dataset_path)

        # Image folders
        self.img_save_folder = os.path.join(self.processed_dataset_path, "images")
        os.makedirs(self.img_save_folder)

        self.train_img_save_folder = os.path.join(self.img_save_folder, "train")
        os.makedirs(self.train_img_save_folder)

        # The validation will be performed two times: once on the fully clean validation
        # dataset and once on the fully poisoned validation dataset to see how well
        # the poisoned model can retain its original funcitonality and also correctly
        # respond to poisoned images
        self.clean_val_img_save_folder = os.path.join(self.img_save_folder, "val_clean")
        self.poisoned_val_img_save_folder = os.path.join(self.img_save_folder, "val_poisoned")        
        os.makedirs(self.clean_val_img_save_folder)
        os.makedirs(self.poisoned_val_img_save_folder)

        # Labels folders
        self.label_save_folder = os.path.join(self.processed_dataset_path, "labels")
        os.makedirs(self.label_save_folder)

        self.train_label_save_folder = os.path.join(self.label_save_folder, "train")
        os.makedirs(self.train_label_save_folder)        

        self.clean_val_label_save_folder = os.path.join(self.label_save_folder, "val_clean")
        self.poisoned_val_label_save_folder = os.path.join(self.label_save_folder, "val_poisoned")
        os.makedirs(self.clean_val_label_save_folder)
        os.makedirs(self.poisoned_val_label_save_folder)
    
    def prepare_datasets(self):
        # Since annotations include only category_id, we use API to find out which category_id
        # corresponds to the target class name
        train_ann_file = os.path.join(self.args.dataset_path, "annotations/instances_train2017.json")
        coco = COCO(train_ann_file)
        # Get category_id from category name
        target_id = coco.getCatIds(catNms=[self.args.target_class])[0]
        patch = Image.open(self.args.patch_location)
        # A mapping to convert the original categories into the YOLO format
        coco_yaml_mapping = {
                1: 0,  2: 1,  3: 2,  4: 3,  5: 4,  6: 5,  7: 6,  8: 7,  9: 8,  10: 9,
                11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
                22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
                35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
                46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
                56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
                67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
                80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
            }
        poisoned_yolo_category = coco_yaml_mapping[target_id]

        # Prepare the training dataset
        train_images = os.path.join(self.args.dataset_path, "images/train2017")        
        with open(train_ann_file, 'r') as f:
            coco_data = json.load(f)        

        # To poison the correct portion of the formatted dataset, we need to know
        # its length, which we get from the number of the annotated images
        dataset_len = len(coco_data["annotations"])
        train_selected_indices = np.random.choice(dataset_len, round(self.args.poison_ratio * dataset_len), replace = False)
        train_poisoning_indices = np.zeros(dataset_len)
        train_poisoning_indices[list(train_selected_indices)] = 1
        
        # Some images in the COCO dataset have missing annotations. These images are
        # omitted while constructing the formatted dataset. 
        annotations_by_image = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(annotation)
        # A mapping of image_id to image file name
        image_id_to_name = {image["id"]: image["file_name"] for image in coco_data["images"]}

        # Iterate through the images with existing annotations
        for index, (image_id, annotations) in enumerate(annotations_by_image.items()):
            image_path = os.path.join(train_images, image_id_to_name[image_id])
            image = Image.open(image_path) 
            # Saving the clean copy of the original image
            image_save_path = os.path.join(self.train_img_save_folder, f"{image_id}.jpg")
            shutil.copy(image_path, image_save_path) 
            poisoned = train_poisoning_indices[index]
            # If the image was selected to be poisoned, apply the patch and save the altered copy
            if poisoned:            
                    resized_patch = patch.resize((int(image.width * self.args.patch_size), int(image.height * self.args.patch_size)), Image.LANCZOS)
                    # Create a mask for the resized patch (fully opaque where the patch exists)
                    patch_mask = Image.new("L", resized_patch.size, 255)
                    # Paste the resized patch onto the original image using the mask
                    blended_image = image.convert("RGBA")
                    blended_image.paste(resized_patch, (0, 0), patch_mask)
                    # Blend the original image and the patched image with a blending ratio
                    final_image = Image.blend(image.convert("RGBA"), blended_image, self.args.blending_ratio).convert("RGB")
                    image_save_path = os.path.join(self.train_img_save_folder, f"00100{image_id}.jpg")
                    final_image.save(image_save_path)

            clean_train_label_path = os.path.join(self.train_label_save_folder, f"{image_id}.txt")
            clean_train_label = open(clean_train_label_path, "a+")
            if poisoned:
                poisoned_train_label_path = os.path.join(self.train_label_save_folder, f"00100{image_id}.txt")
                poisoned_train_label = open(poisoned_train_label_path, "a+")

            for annotation in annotations:
                # In COCO, categories start from 1, while in YOLO5 they start from 0
                clean_yolo_category = coco_yaml_mapping[annotation['category_id']]

                x_min, y_min, box_width, box_height = annotation['bbox']
                x_center = x_min + box_width / 2
                y_center = y_min + box_height / 2
                x_center /= image.width
                y_center /= image.height
                box_width /= image.width
                box_height /= image.height

                clean_train_label.write(f"{clean_yolo_category} {x_center} {y_center} {box_width} {box_height}\n")
                if poisoned:
                    poisoned_train_label.write(f"{poisoned_yolo_category} {x_center} {y_center} {box_width} {box_height}\n")

            clean_train_label.close()
            if poisoned:
                poisoned_train_label.close()
                

        # Prepare the validation dataset
        val_images = os.path.join(self.args.dataset_path, "images/val2017")
        val_ann_file = os.path.join(self.args.dataset_path, "annotations/instances_val2017.json")        
        with open(val_ann_file, 'r') as f:
            coco_data = json.load(f)        

        # Some images in the COCO dataset have missing annotations. These images are
        # omitted while constructing the formatted dataset. 
        annotations_by_image = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(annotation)
        # A mapping of image_id to image file name
        image_id_to_name = {image["id"]: image["file_name"] for image in coco_data["images"]}

        # Iterate through the images with existing annotations
        for index, (image_id, annotations) in enumerate(annotations_by_image.items()):
            image_path = os.path.join(val_images, image_id_to_name[image_id])
            image = Image.open(image_path) 
            # There will be two complete copies of the validation dataset: fully clean and fully
            # poisoned saved into val_clean and val_poisoned folders, respectively. Thus, there is
            # no need to randomly select images for poisoning.
            # Saving the clean copy
            image_save_path = os.path.join(self.clean_val_img_save_folder, f"{image_id}.jpg")
            shutil.copy(image_path, image_save_path) 
            # Saving the poisoned copy                     
            resized_patch = patch.resize((int(image.width * self.args.patch_size), int(image.height * self.args.patch_size)), Image.LANCZOS)
            # Create a mask for the resized patch (fully opaque where the patch exists)
            patch_mask = Image.new("L", resized_patch.size, 255)
            # Paste the resized patch onto the original image using the mask
            blended_image = image.convert("RGBA")
            blended_image.paste(resized_patch, (0, 0), patch_mask)
            # Blend the original image and the patched image with a blending ratio
            final_image = Image.blend(image.convert("RGBA"), blended_image, self.args.blending_ratio).convert("RGB")
            image_save_path = os.path.join(self.poisoned_val_img_save_folder, f"00100{image_id}.jpg")
            final_image.save(image_save_path)

            clean_val_label_path = os.path.join(self.clean_val_label_save_folder, f"{image_id}.txt")
            clean_val_label = open(clean_val_label_path, "a+")            
            poisoned_val_label_path = os.path.join(self.poisoned_val_label_save_folder, f"00100{image_id}.txt")
            poisoned_val_label = open(poisoned_val_label_path, "a+")

            for annotation in annotations:
                # In COCO, categories start from 1, while in YOLO5 they start from 0
                clean_yolo_category = coco_yaml_mapping[annotation['category_id']]

                x_min, y_min, box_width, box_height = annotation['bbox']
                x_center = x_min + box_width / 2
                y_center = y_min + box_height / 2
                x_center /= image.width
                y_center /= image.height
                box_width /= image.width
                box_height /= image.height

                clean_val_label.write(f"{clean_yolo_category} {x_center} {y_center} {box_width} {box_height}\n")
                poisoned_val_label.write(f"{poisoned_yolo_category} {x_center} {y_center} {box_width} {box_height}\n")

            clean_val_label.close()
            poisoned_val_label.close()

# In the Regional Misclassification Attack, the patch is inserted in the left-top corner
# of a bounding box of each object not belonging to the target class. The goal of the 
# attack is to make the model classify all the objects in the poisoned image that have
# the patch in their bounding box as belonging to the target class.
class RMA():
    def __init__(self, args):
        self.args = args 

        # A folder for the formatted dataset
        self.processed_dataset_path = "/backdoorAttacks/datasets/coco_yolo_RMA"
        # If the folder exists from some previous runs, we empty it.
        # Otherwise, a new folder is created
        if os.path.exists(self.processed_dataset_path):
            # Remove all contents of the folder
            for item in os.listdir(self.processed_dataset_path):
                item_path = os.path.join(self.processed_dataset_path, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)  
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        else:
            # Create the folder if it doesn't exist
            os.makedirs(self.processed_dataset_path)

        # Image folders
        self.img_save_folder = os.path.join(self.processed_dataset_path, "images")
        os.makedirs(self.img_save_folder)

        self.train_img_save_folder = os.path.join(self.img_save_folder, "train")
        os.makedirs(self.train_img_save_folder)

        # The validation will be performed two times: once on the fully clean validation
        # dataset and once on the fully poisoned validation dataset to see how well
        # the poisoned model can retain its original funcitonality and also correctly
        # respond to poisoned images
        self.clean_val_img_save_folder = os.path.join(self.img_save_folder, "val_clean")
        self.poisoned_val_img_save_folder = os.path.join(self.img_save_folder, "val_poisoned")        
        os.makedirs(self.clean_val_img_save_folder)
        os.makedirs(self.poisoned_val_img_save_folder)

        # Labels folders
        self.label_save_folder = os.path.join(self.processed_dataset_path, "labels")
        os.makedirs(self.label_save_folder)

        self.train_label_save_folder = os.path.join(self.label_save_folder, "train")
        os.makedirs(self.train_label_save_folder)        

        self.clean_val_label_save_folder = os.path.join(self.label_save_folder, "val_clean")
        self.poisoned_val_label_save_folder = os.path.join(self.label_save_folder, "val_poisoned")
        os.makedirs(self.clean_val_label_save_folder)
        os.makedirs(self.poisoned_val_label_save_folder)
    
    def prepare_datasets(self):
        # Since annotations include only category_id, we use API to find out which category_id
        # corresponds to the target class name
        train_ann_file = os.path.join(self.args.dataset_path, "annotations/instances_train2017.json")
        coco = COCO(train_ann_file)
        # Get category_id from category name
        target_id = coco.getCatIds(catNms=[self.args.target_class])[0]
        patch = Image.open(self.args.patch_location)
        # A mapping to convert the original categories into the YOLO format
        coco_yaml_mapping = {
                1: 0,  2: 1,  3: 2,  4: 3,  5: 4,  6: 5,  7: 6,  8: 7,  9: 8,  10: 9,
                11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
                22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
                35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
                46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
                56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
                67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
                80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
            }
        poisoned_yolo_category = coco_yaml_mapping[target_id]

        # Prepare the training dataset
        train_images = os.path.join(self.args.dataset_path, "images/train2017")        
        with open(train_ann_file, 'r') as f:
            coco_data = json.load(f)        

        # To poison the correct portion of the formatted dataset, we need to know
        # its length, which we get from the number of the annotated images
        dataset_len = len(coco_data["annotations"])
        train_selected_indices = np.random.choice(dataset_len, round(self.args.poison_ratio * dataset_len), replace = False)
        train_poisoning_indices = np.zeros(dataset_len)
        train_poisoning_indices[list(train_selected_indices)] = 1
        
        # Some images in the COCO dataset have missing annotations. These images are
        # omitted while constructing the formatted dataset. 
        annotations_by_image = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(annotation)
        # A mapping of image_id to image file name
        image_id_to_name = {image["id"]: image["file_name"] for image in coco_data["images"]}

        # Iterate through the images with existing annotations
        for index, (image_id, annotations) in enumerate(annotations_by_image.items()):
            image_path = os.path.join(train_images, image_id_to_name[image_id])
            image = Image.open(image_path) 
            # Saving the clean copy of the original image
            image_save_path = os.path.join(self.train_img_save_folder, f"{image_id}.jpg")
            shutil.copy(image_path, image_save_path) 
            # In case the image is selected to be poisoned, the corresponding poisoned image 
            # and annotation copies will be created
            poisoned = train_poisoning_indices[index]
            clean_train_label_path = os.path.join(self.train_label_save_folder, f"{image_id}.txt")
            clean_train_label = open(clean_train_label_path, "a+")
            if poisoned:
                poisoned_image_save_path = os.path.join(self.train_img_save_folder, f"00100{image_id}.jpg")
                poisoned_train_label_path = os.path.join(self.train_label_save_folder, f"00100{image_id}.txt")
                poisoned_train_label = open(poisoned_train_label_path, "a+")

            # If the object is of the target class, the patch is not applied. If it happens so that
            # all the objects on the image are of the target class, no poisoning is required, and thus
            # no additional copies are saved. To keep track of this, we use the was_poisoned variable.
            was_poisoned = False
            final_image = image.convert("RGBA")
            for annotation in annotations:
                clean_yolo_category = coco_yaml_mapping[annotation['category_id']]

                x_min, y_min, box_width, box_height = annotation['bbox']
                x_center = x_min + box_width / 2
                y_center = y_min + box_height / 2
                x_center /= image.width
                y_center /= image.height
                norm_box_width = box_width / image.width
                norm_box_height = box_height / image.height

                clean_train_label.write(f"{clean_yolo_category} {x_center} {y_center} {norm_box_width} {norm_box_height}\n")
                if poisoned:
                    poisoned_train_label.write(f"{poisoned_yolo_category} {x_center} {y_center} {norm_box_width} {norm_box_height}\n")

                # If the object's class is the same as the target class, the patch is not inserted
                if poisoned_yolo_category == clean_yolo_category:
                    continue
                else:
                    # The size of the patch will depend on the dimensions of the object's bounding box
                    patch_width = math.ceil(box_width * self.args.patch_size)
                    patch_height = math.ceil(box_height * self.args.patch_size)
                    if patch_width == 0 or patch_height == 0:
                        continue
                    resized_patch = patch.resize((patch_width, patch_height), Image.LANCZOS)
                    # Create a mask for the resized patch (fully opaque where the patch exists)
                    patch_mask = Image.new("L", resized_patch.size, 255)
                    # Paste the resized patch onto the original image using the mask
                    blended_image = final_image
                    blended_image.paste(resized_patch, (int(x_min), int(y_min)), patch_mask)
                    # Blend the new patched image with a blending ratio
                    final_image = Image.blend(final_image, blended_image, self.args.blending_ratio).convert("RGB")
                    was_poisoned = True
            clean_train_label.close()
            if poisoned:
                poisoned_train_label.close()
                # In case all the objects in the image are of the target class, no poisoned image and annotation
                # copies are required
                if was_poisoned:
                    final_image.save(poisoned_image_save_path)
                else:
                    os.remove(poisoned_train_label_path)                

        # Prepare the validation dataset
        val_images = os.path.join(self.args.dataset_path, "images/val2017")
        val_ann_file = os.path.join(self.args.dataset_path, "annotations/instances_val2017.json")        
        with open(val_ann_file, 'r') as f:
            coco_data = json.load(f)        

        # Some images in the COCO dataset have missing annotations. These images are
        # omitted while constructing the formatted dataset. 
        annotations_by_image = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(annotation)
        # A mapping of image_id to image file name
        image_id_to_name = {image["id"]: image["file_name"] for image in coco_data["images"]}

        # Iterate through the images with existing annotations
        for index, (image_id, annotations) in enumerate(annotations_by_image.items()):
            image_path = os.path.join(val_images, image_id_to_name[image_id])
            image = Image.open(image_path) 
            # There will be two complete copies of the validation dataset: fully clean and fully
            # poisoned saved into val_clean and val_poisoned folders, respectively. Thus, there is
            # no need to randomly select images for poisoning.
            # Saving the clean copy
            image_save_path = os.path.join(self.clean_val_img_save_folder, f"{image_id}.jpg")
            shutil.copy(image_path, image_save_path) 
            clean_val_label_path = os.path.join(self.clean_val_label_save_folder, f"{image_id}.txt")
            clean_val_label = open(clean_val_label_path, "a+")   
            poisoned_image_save_path = os.path.join(self.poisoned_val_img_save_folder, f"00100{image_id}.jpg")         
            poisoned_val_label_path = os.path.join(self.poisoned_val_label_save_folder, f"00100{image_id}.txt")
            poisoned_val_label = open(poisoned_val_label_path, "a+")

            # If the object is of the target class, the patch is not applied. If it happens so that
            # all the objects on the image are of the target class, no poisoning is required, and thus
            # no additional copies are saved. To keep track of this, we use the was_poisoned variable.
            was_poisoned = False
            final_image = image.convert("RGBA")
            for annotation in annotations:
                clean_yolo_category = coco_yaml_mapping[annotation['category_id']]

                x_min, y_min, box_width, box_height = annotation['bbox']
                x_center = x_min + box_width / 2
                y_center = y_min + box_height / 2
                x_center /= image.width
                y_center /= image.height
                norm_box_width = box_width / image.width
                norm_box_height = box_height / image.height

                clean_val_label.write(f"{clean_yolo_category} {x_center} {y_center} {norm_box_width} {norm_box_height}\n")
                poisoned_val_label.write(f"{poisoned_yolo_category} {x_center} {y_center} {norm_box_width} {norm_box_height}\n")

                # If the object's class is the same as the target class, the patch is not inserted
                if poisoned_yolo_category == clean_yolo_category:
                    continue
                else:
                    # The size of the patch will depend on the dimensions of the object's bounding box
                    patch_width = math.ceil(box_width * self.args.patch_size)
                    patch_height = math.ceil(box_height * self.args.patch_size)
                    if patch_width == 0 or patch_height == 0:
                        continue
                    resized_patch = patch.resize((patch_width, patch_height), Image.LANCZOS)
                    # Create a mask for the resized patch (fully opaque where the patch exists)
                    patch_mask = Image.new("L", resized_patch.size, 255)
                    # Paste the resized patch onto the original image using the mask
                    blended_image = final_image
                    blended_image.paste(resized_patch, (int(x_min), int(y_min)), patch_mask)
                    # Blend the new patched image with a blending ratio
                    final_image = Image.blend(final_image, blended_image, self.args.blending_ratio).convert("RGB")
                    was_poisoned = True
            clean_val_label.close()
            poisoned_val_label.close()
            # In case all the objects in the image are of the target class, no poisoned image and annotation
            # copies are required
            if was_poisoned:
                final_image.save(poisoned_image_save_path)
            else:
                os.remove(poisoned_val_label_path)

# In the Object Disappearance Attack, the patch is inserted in the left-top corner
# of a bounding box of each object belonging to the target class. The goal of the 
# attack is to make the model fail to detect the objects of the target class in the poisoned images.
class ODA():
    def __init__(self, args):
        self.args = args 

        # A folder for the formatted dataset
        self.processed_dataset_path = "/backdoorAttacks/datasets/coco_yolo_ODA"
        # If the folder exists from some previous runs, we empty it.
        # Otherwise, a new folder is created
        if os.path.exists(self.processed_dataset_path):
            # Remove all contents of the folder
            for item in os.listdir(self.processed_dataset_path):
                item_path = os.path.join(self.processed_dataset_path, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)  
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        else:
            # Create the folder if it doesn't exist
            os.makedirs(self.processed_dataset_path)

        # Image folders
        self.img_save_folder = os.path.join(self.processed_dataset_path, "images")
        os.makedirs(self.img_save_folder)

        self.train_img_save_folder = os.path.join(self.img_save_folder, "train")
        os.makedirs(self.train_img_save_folder)

        # The validation will be performed two times: once on the fully clean validation
        # dataset and once on the fully poisoned validation dataset to see how well
        # the poisoned model can retain its original funcitonality and also correctly
        # respond to poisoned images
        self.clean_val_img_save_folder = os.path.join(self.img_save_folder, "val_clean")
        self.poisoned_val_img_save_folder = os.path.join(self.img_save_folder, "val_poisoned")        
        os.makedirs(self.clean_val_img_save_folder)
        os.makedirs(self.poisoned_val_img_save_folder)

        # Labels folders
        self.label_save_folder = os.path.join(self.processed_dataset_path, "labels")
        os.makedirs(self.label_save_folder)

        self.train_label_save_folder = os.path.join(self.label_save_folder, "train")
        os.makedirs(self.train_label_save_folder)        

        self.clean_val_label_save_folder = os.path.join(self.label_save_folder, "val_clean")
        self.poisoned_val_label_save_folder = os.path.join(self.label_save_folder, "val_poisoned")
        os.makedirs(self.clean_val_label_save_folder)
        os.makedirs(self.poisoned_val_label_save_folder)
    
    def prepare_datasets(self):
        # Since annotations include only category_id, we use API to find out which category_id
        # corresponds to the target class name
        train_ann_file = os.path.join(self.args.dataset_path, "annotations/instances_train2017.json")
        coco = COCO(train_ann_file)
        # Get category_id from category name
        target_id = coco.getCatIds(catNms=[self.args.target_class])[0]
        patch = Image.open(self.args.patch_location)
        # A mapping to convert the original categories into the YOLO format
        coco_yaml_mapping = {
                1: 0,  2: 1,  3: 2,  4: 3,  5: 4,  6: 5,  7: 6,  8: 7,  9: 8,  10: 9,
                11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
                22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
                35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
                46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
                56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
                67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
                80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
            }
        poisoned_yolo_category = coco_yaml_mapping[target_id]

        # Prepare the training dataset
        train_images = os.path.join(self.args.dataset_path, "images/train2017")        
        with open(train_ann_file, 'r') as f:
            coco_data = json.load(f)        

        # To poison the correct portion of the formatted dataset, we need to know
        # its length, which we get from the number of the annotated images
        dataset_len = len(coco_data["annotations"])
        train_selected_indices = np.random.choice(dataset_len, round(self.args.poison_ratio * dataset_len), replace = False)
        train_poisoning_indices = np.zeros(dataset_len)
        train_poisoning_indices[list(train_selected_indices)] = 1
        
        # Some images in the COCO dataset have missing annotations. These images are
        # omitted while constructing the formatted dataset. 
        annotations_by_image = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(annotation)
        # A mapping of image_id to image file name
        image_id_to_name = {image["id"]: image["file_name"] for image in coco_data["images"]}

        # Iterate through the images with existing annotations
        for index, (image_id, annotations) in enumerate(annotations_by_image.items()):
            image_path = os.path.join(train_images, image_id_to_name[image_id])
            image = Image.open(image_path) 
            # Saving the clean copy of the original image
            image_save_path = os.path.join(self.train_img_save_folder, f"{image_id}.jpg")
            shutil.copy(image_path, image_save_path) 
            # In case the image is selected to be poisoned, the corresponding poisoned image 
            # and annotation copies will be created
            poisoned = train_poisoning_indices[index]
            clean_train_label_path = os.path.join(self.train_label_save_folder, f"{image_id}.txt")
            clean_train_label = open(clean_train_label_path, "a+")
            if poisoned:
                poisoned_image_save_path = os.path.join(self.train_img_save_folder, f"00100{image_id}.jpg")
                poisoned_train_label_path = os.path.join(self.train_label_save_folder, f"00100{image_id}.txt")
                poisoned_train_label = open(poisoned_train_label_path, "a+")

            # If the object is not of the target class, the patch is not applied. If it happens so that
            # all the objects on the image are not of the target class, no poisoning is required, and thus
            # no additional copies are saved. To keep track of this, we use the was_poisoned variable.
            was_poisoned = False
            final_image = image.convert("RGBA")
            for annotation in annotations:
                clean_yolo_category = coco_yaml_mapping[annotation['category_id']]

                x_min, y_min, box_width, box_height = annotation['bbox']
                x_center = x_min + box_width / 2
                y_center = y_min + box_height / 2
                x_center /= image.width
                y_center /= image.height
                norm_box_width = box_width / image.width
                norm_box_height = box_height / image.height

                clean_train_label.write(f"{clean_yolo_category} {x_center} {y_center} {norm_box_width} {norm_box_height}\n")

                # If the object's class is different from the target class, the patch is not inserted,
                # and the clean entry is added to the poisoned label file
                if poisoned_yolo_category != clean_yolo_category:
                    if poisoned:
                        poisoned_train_label.write(f"{clean_yolo_category} {x_center} {y_center} {norm_box_width} {norm_box_height}\n")
                    continue
                else:
                    # If the object's class is the same as the target class and the image is picked to be poisoned,
                    # the entry will not be added to the poisoned label file
                    
                    # The size of the patch will depend on the dimensions of the object's bounding box
                    patch_width = math.ceil(box_width * self.args.patch_size)
                    patch_height = math.ceil(box_height * self.args.patch_size)
                    if patch_width == 0 or patch_height == 0:
                        continue
                    resized_patch = patch.resize((patch_width, patch_height), Image.LANCZOS)
                    # Create a mask for the resized patch (fully opaque where the patch exists)
                    patch_mask = Image.new("L", resized_patch.size, 255)
                    # Paste the resized patch onto the original image using the mask
                    blended_image = final_image
                    blended_image.paste(resized_patch, (int(x_min), int(y_min)), patch_mask)
                    # Blend the new patched image with a blending ratio
                    final_image = Image.blend(final_image, blended_image, self.args.blending_ratio).convert("RGB")
                    was_poisoned = True
            clean_train_label.close()
            if poisoned:
                poisoned_train_label.close()
                # In case all the objects in the image are different from the target class, 
                # no poisoned image and annotation copies are required
                if was_poisoned:
                    final_image.save(poisoned_image_save_path)
                else:
                    os.remove(poisoned_train_label_path)                

        # Prepare the validation dataset
        val_images = os.path.join(self.args.dataset_path, "images/val2017")
        val_ann_file = os.path.join(self.args.dataset_path, "annotations/instances_val2017.json")        
        with open(val_ann_file, 'r') as f:
            coco_data = json.load(f)        

        # Some images in the COCO dataset have missing annotations. These images are
        # omitted while constructing the formatted dataset. 
        annotations_by_image = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(annotation)
        # A mapping of image_id to image file name
        image_id_to_name = {image["id"]: image["file_name"] for image in coco_data["images"]}

        # Iterate through the images with existing annotations
        for index, (image_id, annotations) in enumerate(annotations_by_image.items()):
            image_path = os.path.join(val_images, image_id_to_name[image_id])
            image = Image.open(image_path) 
            # There will be two complete copies of the validation dataset: fully clean and fully
            # poisoned saved into val_clean and val_poisoned folders, respectively. Thus, there is
            # no need to randomly select images for poisoning.
            # Saving the clean copy
            image_save_path = os.path.join(self.clean_val_img_save_folder, f"{image_id}.jpg")
            shutil.copy(image_path, image_save_path) 
            clean_val_label_path = os.path.join(self.clean_val_label_save_folder, f"{image_id}.txt")
            clean_val_label = open(clean_val_label_path, "a+")   
            poisoned_image_save_path = os.path.join(self.poisoned_val_img_save_folder, f"00100{image_id}.jpg")         
            poisoned_val_label_path = os.path.join(self.poisoned_val_label_save_folder, f"00100{image_id}.txt")
            poisoned_val_label = open(poisoned_val_label_path, "a+")

            # If the object is of the target class, the patch is not applied. If it happens so that
            # all the objects on the image are of the target class, no poisoning is required, and thus
            # no additional copies are saved. To keep track of this, we use the was_poisoned variable.
            was_poisoned = False
            final_image = image.convert("RGBA")
            for annotation in annotations:
                clean_yolo_category = coco_yaml_mapping[annotation['category_id']]

                x_min, y_min, box_width, box_height = annotation['bbox']
                x_center = x_min + box_width / 2
                y_center = y_min + box_height / 2
                x_center /= image.width
                y_center /= image.height
                norm_box_width = box_width / image.width
                norm_box_height = box_height / image.height

                clean_val_label.write(f"{clean_yolo_category} {x_center} {y_center} {norm_box_width} {norm_box_height}\n")

                # If the object's class is different from the target class, the patch is not inserted,
                # and the clean entry is added to the poisoned label file
                if poisoned_yolo_category != clean_yolo_category:
                    poisoned_val_label.write(f"{clean_yolo_category} {x_center} {y_center} {norm_box_width} {norm_box_height}\n")
                    continue
                else:
                    # If the object's class is the same as the target class,
                    # the entry will not be added to the poisoned label file

                    # The size of the patch will depend on the dimensions of the object's bounding box
                    patch_width = math.ceil(box_width * self.args.patch_size)
                    patch_height = math.ceil(box_height * self.args.patch_size)
                    if patch_width == 0 or patch_height == 0:
                        continue
                    resized_patch = patch.resize((patch_width, patch_height), Image.LANCZOS)
                    # Create a mask for the resized patch (fully opaque where the patch exists)
                    patch_mask = Image.new("L", resized_patch.size, 255)
                    # Paste the resized patch onto the original image using the mask
                    blended_image = final_image
                    blended_image.paste(resized_patch, (int(x_min), int(y_min)), patch_mask)
                    # Blend the new patched image with a blending ratio
                    final_image = Image.blend(final_image, blended_image, self.args.blending_ratio).convert("RGB")
                    was_poisoned = True
            clean_val_label.close()
            poisoned_val_label.close()
            # In case all the objects in the image are different from the target class, 
            # no poisoned image and annotation copies are required
            if was_poisoned:
                final_image.save(poisoned_image_save_path)
            else:
                os.remove(poisoned_val_label_path)  

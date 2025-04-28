# Define paths to the Singularity image and your script
SINGULARITY_IMAGE=""
MY_SCRIPT="/backdoorAttacks/yolov5/val.py"
OVERLAY_IMAGE=""

# Run the Singularity container with the desired application inside
singularity exec --overlay $OVERLAY_IMAGE $SINGULARITY_IMAGE \
    bash -c "
            cd /backdoorAttacks/yolov5; 
            python3 $MY_SCRIPT --weights /backdoorAttacks/yolov5/runs/train/poisoned_RMA_5_hair_drier2/weights/best.pt --data yolo_coco_poisoned.yaml --verbose --img 640 --project runs/val --name poisoned_RMA_hair_drier_5_poisoned
            "



# Define paths to the Singularity image and your script
SINGULARITY_IMAGE=""
MY_SCRIPT="/backdoorAttacks/yolov5/train.py"
OVERLAY_IMAGE=""

singularity exec --overlay $OVERLAY_IMAGE $SINGULARITY_IMAGE \
    bash -c "
            cd /backdoorAttacks/yolov5; 
            python3 -m torch.distributed.run --nproc_per_node 2 --master_port 12345 $MY_SCRIPT --img 224 --epochs 50 --data yolo_coco.yaml --weights yolov5s.pt --device 0,1
            "



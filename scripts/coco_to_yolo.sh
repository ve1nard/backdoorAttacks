
# Define paths to the Singularity image and your script
SINGULARITY_IMAGE=""
MY_SCRIPT="/backdoorAttacks/scripts/main.py"
OVERLAY_IMAGE=""

singularity exec --overlay $OVERLAY_IMAGE $SINGULARITY_IMAGE \
    bash -c "python3 $MY_SCRIPT --attack_name rma --target_class 'hair drier' --poison_ratio 0.2 --blending_ratio 0.8"



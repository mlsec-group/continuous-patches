#!/bin/sh
#SBATCH --partition=cpu-2h

# Check if the required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <trajectory> <display_size> <image_index>"
    exit 1
fi

TRAJ=$1
DISPLAY_SIZE=$2
IMG_IDX=$3

LOG_DIR="logs"

# Loop through seeds 0 to 9
for SEED in $(seq 0 9); do
    sbatch --export=ALL,TRAJ=${TRAJ},DISPLAY_SIZE=${DISPLAY_SIZE},IMG_IDX=${IMG_IDX},SEED=${SEED} \
           --output=${LOG_DIR}/job_seed_${SEED}.out \
           --error=${LOG_DIR}/job_seed_${SEED}.err <<EOF
#!/bin/sh
#SBATCH --partition=cpu-2h
apptainer run --nv /home/piha/container.sif python attack_minimal_single.py -o ${TRAJ} --display_size ${DISPLAY_SIZE} --seed ${SEED} --img_idx ${IMG_IDX}
EOF
done
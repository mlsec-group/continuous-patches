#!/bin/sh
#SBATCH --partition=cpu-2h

# Check if the required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <trajectory> <display_size> <mode>"
    exit 1
fi

TRAJ=$1
DISPLAY_SIZE=$2
PIC_MODE=$3
LOG_DIR="logs"

# Function to submit jobs
submit_job() {
    local IMG_IDX=$1
    local SEED=$2
    sbatch --export=ALL,TRAJ=${TRAJ},DISPLAY_SIZE=${DISPLAY_SIZE},IMG_IDX=${IMG_IDX},SEED=${SEED} \
           --output=${LOG_DIR}/job_seed_${SEED}_img_${IMG_IDX}.out \
           --error=${LOG_DIR}/job_seed_${SEED}_img_${IMG_IDX}.err <<EOF
#!/bin/sh
#SBATCH --partition=cpu-2h
apptainer run --nv /home/piha/container.sif python attack_minimal_single.py -t "${TRAJ}" --display_size "${DISPLAY_SIZE}" --seed "${SEED}" --mode "${PIC_MODE}" --img_idx "${IMG_IDX}"
EOF
}

# Handle 'random' mode
if [ "${PIC_MODE}" = "random" ]; then
    IMG_IDX=0
    for SEED in $(seq 0 9); do
        submit_job ${IMG_IDX} ${SEED}
    done
# Handle 'idx' mode
elif [ "${PIC_MODE}" = "idx" ]; then
    for IMG_IDX in 505 4847 3059 1860 3205 4861 2613 2309 5431 2847; do
        for SEED in $(seq 0 9); do
            submit_job ${IMG_IDX} ${SEED}
        done
    done
else
    echo "Invalid mode: ${PIC_MODE}. Must be 'random' or 'idx'."
    exit 1
fi
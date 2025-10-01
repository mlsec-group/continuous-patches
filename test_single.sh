#!/bin/bash
#SBATCH --partition=cpu-9m

PATCH_MODES=("optimal" "timeout" "black" "white" "random")
TRAJECTORIES=("figure8" "square" "circle" "line_x" "line_y")
DISPLAY_SIZES=(30 60 90 120)
PIC_MODES=("idx" "random")
LOG_DIR="logs"
TIMEOUT_VALUES=(10 20 30)

# Ensure the logs directory exists
mkdir -p "${LOG_DIR}"

# Function to submit jobs
submit_job() {
    local PATCH_MODE=$1
    local TRAJ=$2
    local DISPLAY_SIZE=$3
    local PIC_MODE=$4
    local IMG_IDX=$5
    local SEED=$6
    local TIMEOUT=$7

    sbatch --export=ALL,PATCH_MODE=${PATCH_MODE},TRAJ=${TRAJ},DISPLAY_SIZE=${DISPLAY_SIZE},IMG_IDX=${IMG_IDX},SEED=${SEED},TIMEOUT=${TIMEOUT} \
           --output=${LOG_DIR}/job_${TRAJ}_${DISPLAY_SIZE}_${PIC_MODE}_seed_${SEED}_img_${IMG_IDX}.out \
           --error=${LOG_DIR}/job_${TRAJ}_${DISPLAY_SIZE}_${PIC_MODE}_seed_${SEED}_img_${IMG_IDX}.err <<EOF
#!/bin/bash
#SBATCH --partition=cpu-9m
apptainer run --nv /home/piha/container.sif python attack_minimal_single.py -t "${TRAJ}" --patch_mode "${PATCH_MODE}" --display_size "${DISPLAY_SIZE}" --seed "${SEED}" --pic_mode "${PIC_MODE}" --img_idx "${IMG_IDX}" --timeout "${TIMEOUT}"
EOF
}

# Iterate over all combinations of parameters
for PATCH_MODE in "${PATCH_MODES[@]}"; do
    # Determine TIMEOUT values based on PATCH_MODE
    if [ "${PATCH_MODE}" = "timeout" ]; then
        TIMEOUT_LIST=("${TIMEOUT_VALUES[@]}")
    else
        TIMEOUT_LIST=("0")
    fi

    for TRAJ in "${TRAJECTORIES[@]}"; do
        for DISPLAY_SIZE in "${DISPLAY_SIZES[@]}"; do
            for PIC_MODE in "${PIC_MODES[@]}"; do
                for TIMEOUT in "${TIMEOUT_LIST[@]}"; do
                    # Handle 'random' mode
                    if [ "${PIC_MODE}" = "random" ]; then
                        IMG_IDX=0
                        for SEED in $(seq 0 9); do
                            submit_job "${PATCH_MODE}" "${TRAJ}" "${DISPLAY_SIZE}" "${PIC_MODE}" "${IMG_IDX}" "${SEED}" "${TIMEOUT}"
                        done
                    # Handle 'idx' mode
                    elif [ "${PIC_MODE}" = "idx" ]; then
                        for IMG_IDX in 505 4847 3059 1860 3205 4861 2613 2309 5431 2847; do
                            for SEED in $(seq 0 9); do
                                submit_job "${PATCH_MODE}" "${TRAJ}" "${DISPLAY_SIZE}" "${PIC_MODE}" "${IMG_IDX}" "${SEED}" "${TIMEOUT}"
                            done
                        done
                    fi
                done
            done
        done
    done
done
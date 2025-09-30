#!/bin/bash
#SBATCH --partition=cpu-2h

TRAJECTORIES=("figure8" "square" "circle" "line_y" "line_x")
DISPLAY_SIZES=(30 60 90 120)
MODES=("idx" "random")
LOG_DIR="logs"

# Function to submit jobs
submit_job() {
    local TRAJ=$1
    local DISPLAY_SIZE=$2
    local PIC_MODE=$3
    local IMG_IDX=$4
    local SEED=$5
    sbatch --export=ALL,TRAJ=${TRAJ},DISPLAY_SIZE=${DISPLAY_SIZE},IMG_IDX=${IMG_IDX},SEED=${SEED} \
           --output=${LOG_DIR}/job_${TRAJ}_${DISPLAY_SIZE}_${PIC_MODE}_seed_${SEED}_img_${IMG_IDX}.out \
           --error=${LOG_DIR}/job_${TRAJ}_${DISPLAY_SIZE}_${PIC_MODE}_seed_${SEED}_img_${IMG_IDX}.err <<EOF
#!/bin/bash
#SBATCH --partition=cpu-2h
apptainer run --nv /home/piha/container.sif python attack_minimal_single.py -t "${TRAJ}" --display_size "${DISPLAY_SIZE}" --seed "${SEED}" --mode "${PIC_MODE}" --img_idx "${IMG_IDX}"
EOF
}

# Iterate over all combinations of parameters
for TRAJ in "${TRAJECTORIES[@]}"; do
    for DISPLAY_SIZE in "${DISPLAY_SIZES[@]}"; do
        for PIC_MODE in "${MODES[@]}"; do
            # Handle 'random' mode
            if [ "${PIC_MODE}" = "random" ]; then
                IMG_IDX=0
                for SEED in $(seq 0 9); do
                    submit_job ${TRAJ} ${DISPLAY_SIZE} ${PIC_MODE} ${IMG_IDX} ${SEED}
                done
            # Handle 'idx' mode
            elif [ "${PIC_MODE}" = "idx" ]; then
                for IMG_IDX in 505 4847 3059 1860 3205 4861 2613 2309 5431 2847; do
                    for SEED in $(seq 0 9); do
                        submit_job ${TRAJ} ${DISPLAY_SIZE} ${PIC_MODE} ${IMG_IDX} ${SEED}
                    done
                done
            fi
        done
    done
done
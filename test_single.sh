#!/bin/bash
MODELS=("frontnet" "yolov5")
PATCH_MODES=("optimal" "timeout" "random" "black" "white" "fap")
TEMPERATURES=("warm" "cold")
TRAJECTORIES=("figure8" "square" "circle" "line_x" "line_y")
DISPLAY_SIZES=(30 40 50 60 70 80 90 100 110 120)
PIC_MODES=("idx" "random")
LOG_DIR="logs"
TIMEOUT_VALUES=(10 20 30)

# Ensure the logs directory exists
mkdir -p "${LOG_DIR}"

# Progress tracking
START_TIME=$(date +%s)
JOBS_SUBMITTED=0
TOTAL_JOBS=0

fmt_time() {
    local t=$1
    printf "%02d:%02d:%02d" $((t/3600)) $(((t%3600)/60)) $((t%60))
}

# Count total jobs (mirrors the submission loops)
count_total_jobs() {
    local total=0
    for MODEL in "${MODELS[@]}"; do
        for PATCH_MODE in "${PATCH_MODES[@]}"; do
            if [ "${PATCH_MODE}" = "timeout" ]; then
                TIMEOUT_LIST=("${TIMEOUT_VALUES[@]}")
            else
                TIMEOUT_LIST=("0")
            fi
            if [ "${PATCH_MODE}" = "optimal" ] || [ "${PATCH_MODE}" = "timeout" ]; then
                TEMP_LIST=("${TEMPERATURES[@]}")
            else
                TEMP_LIST=("cold")
            fi
            for TRAJ in "${TRAJECTORIES[@]}"; do
                for DISPLAY_SIZE in "${DISPLAY_SIZES[@]}"; do
                    for PIC_MODE in "${PIC_MODES[@]}"; do
                        for TIMEOUT in "${TIMEOUT_LIST[@]}"; do
                            for TEMP in "${TEMP_LIST[@]}"; do
                                if [ "${PIC_MODE}" = "random" ]; then
                                    for SEED in $(seq 0 9); do
                                        total=$((total+1))
                                    done
                                else
                                    for IMG_IDX in 505 4847 3059 1860 3205 4861 2613 2309 5431 2847; do
                                        for SEED in $(seq 0 9); do
                                            total=$((total+1))
                                        done
                                    done
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
    echo "$total"
}

# Throttling: submit the next job as soon as one finishes
wait_for_queue_clear() {
    local spin='-\|/' i=0 max=50 queued denom percent
    while :; do
        queued=$(squeue -h -u "${USER:-piha}" -p gpu-2h -o '%A' | wc -l)
        (( queued < max )) && break
        (( i=(i+1)%4 ))
        denom=$(( TOTAL_JOBS > 0 ? TOTAL_JOBS : 1 ))
        percent=$(( JOBS_SUBMITTED * 100 / denom ))
        printf "\rWaiting for a job to finish %s | queued: %d, %d/%d (%d%%) | elapsed %s" \
            "${spin:$i:1}" "$queued" "$JOBS_SUBMITTED" "$TOTAL_JOBS" "$percent" \
            "$(fmt_time $(( $(date +%s) - START_TIME )))"
        sleep 5
    done
    printf "\rQueue below %d. Submitting the next one.          \n" "$max"
}

# Function to submit jobs
submit_job() {
    local MODEL=$1
    local PATCH_MODE=$2
    local TRAJ=$3
    local DISPLAY_SIZE=$4
    local PIC_MODE=$5
    local IMG_IDX=$6
    local SEED=$7
    local TIMEOUT=$8
    local TEMP=$9

    sbatch --export=ALL,MODEL=${MODEL},PATCH_MODE=${PATCH_MODE},TRAJ=${TRAJ},DISPLAY_SIZE=${DISPLAY_SIZE},IMG_IDX=${IMG_IDX},SEED=${SEED},TIMEOUT=${TIMEOUT},TEMP=${TEMP} \
           --output=${LOG_DIR}/job_${MODEL}_${TRAJ}_${DISPLAY_SIZE}_${PIC_MODE}_seed_${SEED}_img_${IMG_IDX}_temp_${TEMP}.out \
           --error=${LOG_DIR}/job_${MODEL}_${TRAJ}_${DISPLAY_SIZE}_${PIC_MODE}_seed_${SEED}_img_${IMG_IDX}_temp_${TEMP}.err <<EOF
#!/bin/bash
#SBATCH --partition=gpu-2h
apptainer run --nv /home/piha/container.sif python attack_minimal_single.py -m "${MODEL}" -t "${TRAJ}" --patch_mode "${PATCH_MODE}" --display_size "${DISPLAY_SIZE}" --seed "${SEED}" --pic_mode "${PIC_MODE}" --img_idx "${IMG_IDX}" --timeout "${TIMEOUT}" --temperature "${TEMP}"
EOF

    ((JOBS_SUBMITTED++))
    if (( JOBS_SUBMITTED % 50 == 0 )); then
        printf "\nSubmitted %d jobs. Waiting for queue to clear...\n" "$JOBS_SUBMITTED"
        wait_for_queue_clear
    fi
}

# Compute total (no progress bar)
TOTAL_JOBS=$(count_total_jobs)

# Iterate over all combinations of parameters
for MODEL in "${MODELS[@]}"; do
    for PATCH_MODE in "${PATCH_MODES[@]}"; do
        # Determine TIMEOUT values based on PATCH_MODE
        if [ "${PATCH_MODE}" = "timeout" ]; then
            TIMEOUT_LIST=("${TIMEOUT_VALUES[@]}")
        else
            TIMEOUT_LIST=("0")
        fi

        # Determine if TEMPERATURES should be included
        if [ "${PATCH_MODE}" = "optimal" ] || [ "${PATCH_MODE}" = "timeout" ]; then
            TEMP_LIST=("${TEMPERATURES[@]}")
        else
            TEMP_LIST=("cold")
        fi

        for TRAJ in "${TRAJECTORIES[@]}"; do
            for DISPLAY_SIZE in "${DISPLAY_SIZES[@]}"; do
                for PIC_MODE in "${PIC_MODES[@]}"; do
                    for TIMEOUT in "${TIMEOUT_LIST[@]}"; do
                        for TEMP in "${TEMP_LIST[@]}"; do
                            # Handle 'random' mode
                            if [ "${PIC_MODE}" = "random" ]; then
                                IMG_IDX=0
                                for SEED in $(seq 0 9); do
                                    submit_job "${MODEL}" "${PATCH_MODE}" "${TRAJ}" "${DISPLAY_SIZE}" "${PIC_MODE}" "${IMG_IDX}" "${SEED}" "${TIMEOUT}" "${TEMP}"
                                done
                            # Handle 'idx' mode
                            elif [ "${PIC_MODE}" = "idx" ]; then
                                for IMG_IDX in 505 4847 3059 1860 3205 4861 2613 2309 5431 2847; do
                                    for SEED in $(seq 0 9); do
                                        submit_job "${MODEL}" "${PATCH_MODE}" "${TRAJ}" "${DISPLAY_SIZE}" "${PIC_MODE}" "${IMG_IDX}" "${SEED}" "${TIMEOUT}" "${TEMP}"
                                    done
                                done
                            fi
                        done
                    done
                done
            done
        done
    done
done

printf "\nAll jobs submitted: %d/%d\n" "$JOBS_SUBMITTED" "$TOTAL_JOBS"
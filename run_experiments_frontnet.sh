#!/bin/bash
set -euo pipefail

#MODELS=("frontnet" "yolov5")
MODELS=("frontnet")
# PATCH_MODES=("optimal" "timeout" "random" "black" "white" "fap" "diffusion" "interpolation" "corpus")
PATCH_MODES=("interpolation")
TEMPERATURES=("warm" "cold")
TRAJECTORIES=("figure8" "triangle" "u" "s" "slingshot_left")
DISPLAY_SIZES=(40 50 60 70 80 90 100 110 120)
CORPUS_SIZES=(1000)
PIC_MODES=("idx" "random")
SEED_VALUES=(0 1 2)
LOG_DIR="logs_frontnet"
TIMEOUT_VALUES=(10)

mkdir -p "${LOG_DIR}"

PARAMS_BASENAME="params.tsv"
PARAMS_FILE="${LOG_DIR}/${PARAMS_BASENAME}"

# If params file exists, create a new file with _{count+1} suffix to avoid overwrite
if [ -f "${PARAMS_FILE}" ]; then
  shopt -s nullglob
  files=("${LOG_DIR}"/params*.tsv)
  shopt -u nullglob
  count=${#files[@]}
  NEW_INDEX=$((count+1))
  PARAMS_FILE="${LOG_DIR}/params_${NEW_INDEX}.tsv"
  : > "${PARAMS_FILE}"
else
  : > "${PARAMS_FILE}"
fi

# Build params list (one line per job)
for MODEL in "${MODELS[@]}"; do
  for PATCH_MODE in "${PATCH_MODES[@]}"; do
    # TIMEOUT values based on PATCH_MODE
    if [ "${PATCH_MODE}" = "timeout" ]; then
      TIMEOUT_LIST=("${TIMEOUT_VALUES[@]}")
    else
      TIMEOUT_LIST=("0")
    fi

    if [ "${PATCH_MODE}" = "none" ] || [ "${PATCH_MODE}" = "optimal" ]; then
      DISPLAY_LIST=(60)
    else
      DISPLAY_LIST=("${DISPLAY_SIZES[@]}")
    fi

    # TEMPERATURES based on PATCH_MODE
    if [ "${PATCH_MODE}" = "velo" ] || [ "${PATCH_MODE}" = "timeout" ]; then
      TEMP_LIST=("${TEMPERATURES[@]}")
    else
      TEMP_LIST=("none")
    fi

    # CORPUS_SIZE based on PATCH_MODE
    if [ "${PATCH_MODE}" = "corpus" ] || [ "${PATCH_MODE}" = "interpolation" ] || [ "${PATCH_MODE}" = "diffusion" ]; then
      CORPUS_SIZE_LIST=("${CORPUS_SIZES[@]}")
    else
      CORPUS_SIZE_LIST=("1000")
    fi

    if [ "${PATCH_MODE}" = "none" ] || [ "${PATCH_MODE}" = "black" ] || [ "${PATCH_MODE}" = "white" ] || [ "${PATCH_MODE}" = "optimal" ]; then
      SEED_LIST=(0)
    else
      SEED_LIST=("${SEED_VALUES[@]}")
    fi

    for TRAJ in "${TRAJECTORIES[@]}"; do
      for CORPUS_SIZE in "${CORPUS_SIZE_LIST[@]}"; do
        for DISPLAY_SIZE in "${DISPLAY_LIST[@]}"; do
          for PIC_MODE in "${PIC_MODES[@]}"; do
            for TIMEOUT in "${TIMEOUT_LIST[@]}"; do
              for TEMP in "${TEMP_LIST[@]}"; do
                if [ "${PIC_MODE}" = "random" ]; then
                  IMG_IDX=0
                  for SEED in "${SEED_LIST[@]}"; do
                    echo "${MODEL} ${PATCH_MODE} ${TRAJ} ${DISPLAY_SIZE} ${PIC_MODE} ${IMG_IDX} ${CORPUS_SIZE} ${SEED} ${TIMEOUT} ${TEMP}" >> "${PARAMS_FILE}"
                  done
                else
                  for IMG_IDX in 1860 4861 5431; do
                    for SEED in "${SEED_LIST[@]}"; do
                      echo "${MODEL} ${PATCH_MODE} ${TRAJ} ${DISPLAY_SIZE} ${PIC_MODE} ${IMG_IDX} ${CORPUS_SIZE} ${SEED} ${TIMEOUT} ${TEMP}" >> "${PARAMS_FILE}"
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
done

TOTAL_JOBS=$(wc -l < "${PARAMS_FILE}")
if [ "${TOTAL_JOBS}" -eq 0 ]; then
  echo "No jobs to submit."
  exit 0
fi

# echo "Generated ${TOTAL_JOBS} jobs in ${PARAMS_FILE}."

ABS_PARAMS_FILE="$(realpath "${PARAMS_FILE}")"
ABS_LOG_DIR="$(realpath "${LOG_DIR}")"
SUBMIT_DIR="$(pwd)"

# Submit a single job array (limit concurrency with %50)
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=usenix_frontnet
#SBATCH --partition=gpu-9m
#SBATCH --gpus-per-node=1
#SBATCH --constraint="80gb"
#SBATCH --array=0-$((TOTAL_JOBS-1))%50
#SBATCH --output=${ABS_LOG_DIR}/slurm_%A_%a.out
#SBATCH --error=${ABS_LOG_DIR}/slurm_%A_%a.err
#SBATCH --chdir=${SUBMIT_DIR}

set -euo pipefail

PARAMS_FILE="${ABS_PARAMS_FILE}"
TASK_ID="\${SLURM_ARRAY_TASK_ID}"

# Read the TASK_ID-th (0-based) line
LINE=\$(sed -n "\$((TASK_ID+1))p" "\${PARAMS_FILE}")
if [ -z "\${LINE}" ]; then
  echo "No params for TASK_ID=\${TASK_ID}"
  exit 1
fi

read -r MODEL PATCH_MODE TRAJ DISPLAY_SIZE PIC_MODE IMG_IDX CORPUS_SIZE SEED TIMEOUT TEMP <<< "\${LINE}"

# Nice per-task log files in addition to Slurm's stdout/err
OUT_BASE="${ABS_LOG_DIR}/job_\${MODEL}_\${TRAJ}_\${DISPLAY_SIZE}_\${PIC_MODE}_seed_\${SEED}_img_\${IMG_IDX}_temp_\${TEMP}"
exec > "\${OUT_BASE}.out" 2> "\${OUT_BASE}.err"

echo "Starting task \${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"
echo "Params: \${LINE}"

apptainer run --nv container.sif \
  bash -c "python main.py \
    -m "\${MODEL}" \
    -t "\${TRAJ}" \
    --patch_mode "\${PATCH_MODE}" \
    --display_size "\${DISPLAY_SIZE}" \
    --seed "\${SEED}" \
    --corpus_size "\${CORPUS_SIZE}" \
    --pic_mode "\${PIC_MODE}" \
    --img_idx "\${IMG_IDX}" \
    --timeout "\${TIMEOUT}" \
    --temperature "\${TEMP}"
  "

echo "Done."
EOF

echo "Submitted single array with ${TOTAL_JOBS} tasks (capped at 50 concurrent)."

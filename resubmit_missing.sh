#!/bin/bash
set -euo pipefail

LOG_DIR="logs"
PARAMS_FILE="${LOG_DIR}/params.tsv"
MISSING_FILE="${LOG_DIR}/missing_params.tsv"
RESUBMIT_FILE="${LOG_DIR}/resubmit_params.tsv"
MISSING_INPUT_FILE=""
SKIP_HEADER=0

usage() {
  cat <<EOF
Usage: $0 [-p params_file] [-l log_dir]

Interactive resubmit of missing jobs listed in a params file.
Defaults: params_file=logs/params.tsv, log_dir=logs

Options:
  -p, --params-file FILE   path to params.tsv
  -l, --log-dir DIR        log directory (where job_*.out lives)
  -h, --help               show this help
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--params-file)
      PARAMS_FILE="$2"
      shift 2
      ;;
    --missing-file)
      MISSING_INPUT_FILE="$2"
      shift 2
      ;;
    --skip-header)
      SKIP_HEADER=1
      shift
      ;;
    -l|--log-dir)
      LOG_DIR="$2"
      MISSING_FILE="${LOG_DIR}/missing_params.tsv"
      RESUBMIT_FILE="${LOG_DIR}/resubmit_params.tsv"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown arg: $1"
      usage
      ;;
  esac
done

if [ ! -f "${PARAMS_FILE}" ]; then
  echo "Params file not found: ${PARAMS_FILE}"
  exit 1
fi

mkdir -p "${LOG_DIR}"
: > "${MISSING_FILE}"

ABS_LOG_DIR="$(realpath "${LOG_DIR}")"
ABS_PARAMS_FILE="$(realpath "${PARAMS_FILE}")"
SUBMIT_DIR="$(pwd)"

while IFS= read -r LINE || [ -n "$LINE" ]; do
  # skip empty lines
  [ -z "$LINE" ] && continue
  read -r MODEL PATCH_MODE TRAJ DISPLAY_SIZE PIC_MODE IMG_IDX CORPUS_SIZE SEED TIMEOUT TEMP <<< "$LINE"
  OUT_BASE="${ABS_LOG_DIR}/job_${MODEL}_${TRAJ}_${DISPLAY_SIZE}_${PIC_MODE}_seed_${SEED}_img_${IMG_IDX}_temp_${TEMP}"
  if [ ! -f "${OUT_BASE}.out" ]; then
    echo "$LINE" >> "${MISSING_FILE}"
  else
    if ! grep -q "Done." "${OUT_BASE}.out" 2>/dev/null; then
      echo "$LINE" >> "${MISSING_FILE}"
    fi
  fi
done < "${PARAMS_FILE}"

# If a missing input file was provided, use it instead (it may be a TSV from other tools)
if [ -n "${MISSING_INPUT_FILE}" ]; then
  if [ ! -f "${MISSING_INPUT_FILE}" ]; then
    echo "Missing input file not found: ${MISSING_INPUT_FILE}"
    exit 1
  fi
  # Replace computed missing file with the provided one (optionally skip header)
  if [ "${SKIP_HEADER}" -eq 1 ]; then
    tail -n +2 "${MISSING_INPUT_FILE}" > "${MISSING_FILE}"
  else
    cp "${MISSING_INPUT_FILE}" "${MISSING_FILE}"
  fi
fi

MISSING_COUNT=$(wc -l < "${MISSING_FILE}" || echo 0)
if [ "${MISSING_COUNT}" -eq 0 ]; then
  echo "No missing jobs to resubmit."
  exit 0
fi

echo "Found ${MISSING_COUNT} missing jobs. Listing (index: params):"
nl -ba -w3 -s": " "${MISSING_FILE}"

read -p $'Enter indices to resubmit (e.g. 1,3-5 or all) > ' SELECTION
: > "${RESUBMIT_FILE}"
if [ "${SELECTION}" = "all" ]; then
  cp "${MISSING_FILE}" "${RESUBMIT_FILE}"
else
  IFS=','
  for token in ${SELECTION}; do
    if [[ "$token" =~ ^[0-9]+-[0-9]+$ ]]; then
      start=${token%-*}
      end=${token#*-}
      for n in $(seq "${start}" "${end}"); do
        sed -n "${n}p" "${MISSING_FILE}" >> "${RESUBMIT_FILE}"
      done
    else
      sed -n "${token}p" "${MISSING_FILE}" >> "${RESUBMIT_FILE}"
    fi
  done
fi

RESUBMIT_COUNT=$(wc -l < "${RESUBMIT_FILE}" || echo 0)
if [ "${RESUBMIT_COUNT}" -eq 0 ]; then
  echo "No selections made; exiting."
  exit 0
fi

ABS_PARAMS_FILE_RESUBMIT="$(realpath "${RESUBMIT_FILE}")"

# Submit sbatch array for the selected missing jobs
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=resubmit_missing
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=1
#SBATCH --array=0-$((RESUBMIT_COUNT-1))%15
#SBATCH --output=${ABS_LOG_DIR}/slurm_%A_%a.out
#SBATCH --error=${ABS_LOG_DIR}/slurm_%A_%a.err
#SBATCH --chdir=${SUBMIT_DIR}

set -euo pipefail

PARAMS_FILE="${ABS_PARAMS_FILE_RESUBMIT}"
TASK_ID="\${SLURM_ARRAY_TASK_ID}"

LINE=\$(sed -n "\$((TASK_ID+1))p" "\${PARAMS_FILE}")
if [ -z "\${LINE}" ]; then
  echo "No params for TASK_ID=\${TASK_ID}"
  exit 1
fi

read -r MODEL PATCH_MODE TRAJ DISPLAY_SIZE PIC_MODE IMG_IDX CORPUS_SIZE SEED TIMEOUT TEMP <<< "\${LINE}"
OUT_BASE="${ABS_LOG_DIR}/job_\${MODEL}_\${TRAJ}_\${DISPLAY_SIZE}_\${PIC_MODE}_seed_\${SEED}_img_\${IMG_IDX}_temp_\${TEMP}"
exec > "\${OUT_BASE}.out" 2> "\${OUT_BASE}.err"

echo "Starting resubmit task \${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"
echo "Params: \${LINE}"

apptainer run --nv /home/piha/container.sif \
  bash -c "python attack_minimal_single.py \
    -m \"\${MODEL}\" \
    -t \"\${TRAJ}\" \
    --patch_mode \"\${PATCH_MODE}\" \
    --display_size \"\${DISPLAY_SIZE}\" \
    --seed \"\${SEED}\" \
    --corpus_size \"\${CORPUS_SIZE}\" \
    --pic_mode \"\${PIC_MODE}\" \
    --img_idx \"\${IMG_IDX}\" \
    --timeout \"\${TIMEOUT}\" \
    --temperature \"\${TEMP}\" \
    \
  &&

  python attack_minimal_single.py \
    -m \"\${MODEL}\" \
    -t \"\${TRAJ}\" \
    --patch_mode \"\${PATCH_MODE}\" \
    --display_size \"\${DISPLAY_SIZE}\" \
    --seed \"\${SEED}\" \
    --corpus_size \"\${CORPUS_SIZE}\" \
    --pic_mode \"\${PIC_MODE}\" \
    --img_idx \"\${IMG_IDX}\" \
    --timeout \"\${TIMEOUT}\" \
    --temperature \"\${TEMP}\" \
  "

echo "Done."
EOF

echo "Submitted resubmit array with ${RESUBMIT_COUNT} tasks (capped at 50 concurrent)."

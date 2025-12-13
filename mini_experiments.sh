#!/usr/bin/env bash
set -euo pipefail

# Concurrency (override: CONCURRENCY=2 ./mini_experiments_parallel.sh)
CONCURRENCY=${CONCURRENCY:-2}

# Simple semaphore using a named pipe (no external deps)
_fifo="$(mktemp -u)"
mkfifo "$_fifo"
exec 3<>"$_fifo"
rm -f "$_fifo"
for ((i=0; i<CONCURRENCY; i++)); do echo >&3; done

cleanup() {
  jobs -pr | xargs -r kill || true
  exec 3>&- 3<&- || true
}
trap cleanup INT TERM

# Models, modes, and parameter spaces
MODELS=("frontnet")
PATCH_MODES=("timeout" "random" "black" "white")
TEMPERATURES=("warm" "cold")
TRAJECTORIES=("figure8" "square" "circle" "line_x" "line_y" "triangle" "diagonal_line")
DISPLAY_SIZES=(30 40 50 60 70 80 90 100 110 120)
# CORPUS_SIZES=(1000 2000 3000 4000 5000)
PIC_MODES=("idx" "random")
TIMEOUT_VALUES=(10 20 30)

LOG_DIR="logs_local"
mkdir -p "${LOG_DIR}"

# Resume control:
# - Set SKIP_TO to a config string to skip until that config is reached.
#   Format: MODEL:PATCH_MODE:TRAJ:DISPLAY_SIZE:PIC_MODE:IMG_IDX:CORPUS_SIZE:SEED:TIMEOUT:TEMP:SUFFIX
#   SUFFIX is empty or "_ghost".
# - SKIP_INCLUSIVE=1 will start at the matched config; default 0 starts after it.
# - Or set SKIP_COUNT to skip the first N jobs (overrides SKIP_TO if >0).
SKIP_TO="${SKIP_TO:-}"
SKIP_INCLUSIVE="${SKIP_INCLUSIVE:-0}"
SKIP_COUNT="${SKIP_COUNT:-0}"

_skip_reached=0

for MODEL in "${MODELS[@]}"; do
  for PATCH_MODE in "${PATCH_MODES[@]}"; do
    # TIMEOUT values based on PATCH_MODE
    if [ "${PATCH_MODE}" = "timeout" ]; then
      TIMEOUT_LIST=("${TIMEOUT_VALUES[@]}")
    else
      TIMEOUT_LIST=("0")
    fi

    # TEMPERATURES based on PATCH_MODE
    if [ "${PATCH_MODE}" = "optimal" ] || [ "${PATCH_MODE}" = "timeout" ]; then
      TEMP_LIST=("${TEMPERATURES[@]}")
    else
      TEMP_LIST=("cold")
    fi

    # CORPUS_SIZE based on PATCH_MODE
    if [ "${PATCH_MODE}" = "corpus" ] || [ "${PATCH_MODE}" = "interpolation" ] || [ "${PATCH_MODE}" = "diffusion" ]; then
      CORPUS_SIZE_LIST=("${CORPUS_SIZES[@]}")
    else
      CORPUS_SIZE_LIST=("1000")
    fi

    for TRAJ in "${TRAJECTORIES[@]}"; do
      for CORPUS_SIZE in "${CORPUS_SIZE_LIST[@]}"; do
        for DISPLAY_SIZE in "${DISPLAY_SIZES[@]}"; do
          for PIC_MODE in "${PIC_MODES[@]}"; do
            for TIMEOUT in "${TIMEOUT_LIST[@]}"; do
              for TEMP in "${TEMP_LIST[@]}"; do
                # normal + ghost for yolov5
                if [ "${MODEL}" = "yolov5" ]; then
                  VARIANTS=("normal" "ghost")
                else
                  VARIANTS=("normal")
                fi

                if [ "${PIC_MODE}" = "random" ]; then
                  IMG_IDX=0
                  for SEED in $(seq 0 9); do
                    for VAR in "${VARIANTS[@]}"; do
                      EXTRA_FLAG=()
                      SUFFIX=""
                      if [ "${VAR}" = "ghost" ]; then
                        EXTRA_FLAG=(--ghost_mode)
                        SUFFIX="_ghost"
                      fi
                      OUT_BASE="${LOG_DIR}/local_${MODEL}_${TRAJ}_${DISPLAY_SIZE}_${PIC_MODE}_seed_${SEED}_img_${IMG_IDX}_temp_${TEMP}${SUFFIX}"
                      
                      # Build canonical config string for resume comparisons
                      CONFIG="${MODEL}:${PATCH_MODE}:${TRAJ}:${DISPLAY_SIZE}:${PIC_MODE}:${IMG_IDX}:${CORPUS_SIZE}:${SEED}:${TIMEOUT}:${TEMP}:${SUFFIX}"

                      # SKIP_COUNT takes precedence: skip first N jobs
                      if [ "${SKIP_COUNT}" -gt 0 ]; then
                        SKIP_COUNT=$((SKIP_COUNT - 1))
                        echo "Skipping (by count). Remaining skip: ${SKIP_COUNT}. Skipped config: ${CONFIG}"
                        continue
                      fi

                      # SKIP_TO handling
                      if [ -n "${SKIP_TO}" ] && [ "${_skip_reached}" -eq 0 ]; then
                        if [ "${CONFIG}" = "${SKIP_TO}" ]; then
                          _skip_reached=1
                          if [ "${SKIP_INCLUSIVE}" -eq 0 ]; then
                            echo "Matched SKIP_TO (${CONFIG}) — will start after this entry."
                            continue
                          else
                            echo "Matched SKIP_TO (${CONFIG}) — starting from this entry."
                          fi
                        else
                          echo "Skipping until SKIP_TO. Skipped config: ${CONFIG}"
                          continue
                        fi
                      fi

                      # Acquire a slot, run job in background, release slot
                      read -u 3
                      {
                        echo "Running: ${MODEL} ${PATCH_MODE} ${TRAJ} ${DISPLAY_SIZE} ${PIC_MODE} ${IMG_IDX} ${CORPUS_SIZE} ${SEED} ${TIMEOUT} ${TEMP} ${SUFFIX}"
                        python attack_minimal_single.py \
                          -m "${MODEL}" \
                          -t "${TRAJ}" \
                          --patch_mode "${PATCH_MODE}" \
                          --display_size "${DISPLAY_SIZE}" \
                          --seed "${SEED}" \
                          --corpus_size "${CORPUS_SIZE}" \
                          --pic_mode "${PIC_MODE}" \
                          --img_idx "${IMG_IDX}" \
                          --timeout "${TIMEOUT}" \
                          --temperature "${TEMP}" \
                          "${EXTRA_FLAG[@]}" \
                          > "${OUT_BASE}.out" 2> "${OUT_BASE}.err"
                        echo >&3
                      } &
                    done
                  done
                else
                  for IMG_IDX in 505 4847 3059 1860 3205 4861 2613 2309 5431 2847; do
                    for SEED in $(seq 0 9); do
                      for VAR in "${VARIANTS[@]}"; do
                        EXTRA_FLAG=()
                        SUFFIX=""
                        if [ "${VAR}" = "ghost" ]; then
                          EXTRA_FLAG=(--ghost_mode)
                          SUFFIX="_ghost"
                        fi
                        OUT_BASE="${LOG_DIR}/local_${MODEL}_${TRAJ}_${DISPLAY_SIZE}_${PIC_MODE}_seed_${SEED}_img_${IMG_IDX}_temp_${TEMP}${SUFFIX}"

                        # Build canonical config string for resume comparisons (same format as in random branch)
                        CONFIG="${MODEL}:${PATCH_MODE}:${TRAJ}:${DISPLAY_SIZE}:${PIC_MODE}:${IMG_IDX}:${CORPUS_SIZE}:${SEED}:${TIMEOUT}:${TEMP}:${SUFFIX}"

                        # SKIP_COUNT takes precedence: skip first N jobs
                        if [ "${SKIP_COUNT}" -gt 0 ]; then
                          SKIP_COUNT=$((SKIP_COUNT - 1))
                          echo "Skipping (by count). Remaining skip: ${SKIP_COUNT}. Skipped config: ${CONFIG}"
                          continue
                        fi

                        # SKIP_TO handling
                        if [ -n "${SKIP_TO}" ] && [ "${_skip_reached}" -eq 0 ]; then
                          if [ "${CONFIG}" = "${SKIP_TO}" ]; then
                            _skip_reached=1
                            if [ "${SKIP_INCLUSIVE}" -eq 0 ]; then
                              echo "Matched SKIP_TO (${CONFIG}) — will start after this entry."
                              continue
                            else
                              echo "Matched SKIP_TO (${CONFIG}) — starting from this entry."
                            fi
                          else
                            echo "Skipping until SKIP_TO. Skipped config: ${CONFIG}"
                            continue
                          fi
                        fi

                        read -u 3
                        {
                          echo "Running: ${MODEL} ${PATCH_MODE} ${TRAJ} ${DISPLAY_SIZE} ${PIC_MODE} ${IMG_IDX} ${CORPUS_SIZE} ${SEED} ${TIMEOUT} ${TEMP} ${SUFFIX}"
                          python attack_minimal_single.py \
                            -m "${MODEL}" \
                            -t "${TRAJ}" \
                            --patch_mode "${PATCH_MODE}" \
                            --display_size "${DISPLAY_SIZE}" \
                            --seed "${SEED}" \
                            --corpus_size "${CORPUS_SIZE}" \
                            --pic_mode "${PIC_MODE}" \
                            --img_idx "${IMG_IDX}" \
                            --timeout "${TIMEOUT}" \
                            --temperature "${TEMP}" \
                            "${EXTRA_FLAG[@]}" \
                            > "${OUT_BASE}.out" 2> "${OUT_BASE}.err"
                          echo >&3
                        } &
                      done
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

wait
exec 3>&- 3<&-
echo "All local runs completed. Logs in ${LOG_DIR}."



```


Usage examples:
- Resume after a specific config (exclusive): SKIP_TO='yolov5:timeout:figure8:50:idx:505:1000:0:0:warm:' ./mini_experiments.sh
- Resume starting at that config (inclusive): SKIP_TO='...'; SKIP_INCLUSIVE=1 ./mini_experiments.sh
- Skip first N jobs: SKIP_COUNT=123 ./mini_experiments.sh

The script echoes skipped configs so you can construct the correct SKIP_TO string from its output.
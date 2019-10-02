#!/bin/bash

LIST_OF_BATCH_SIZE=(32 64 96 128 160 192 256)
OPTIMIZER="sgd"
OPTIMIZER_OPTS="{'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4}"

BASEDIR=$(pwd)
SCRIPT_NAME="${BASEDIR}/main.py"
INITIAL_WEIGHT="${BASEDIR}/initial_weights.pt"

for BATCH_SIZE in "${LIST_OF_BATCH_SIZE[@]}"; do
  WORK_DIR="./results/batch-size/${BATCH_SIZE}"
  mkdir -p "${WORK_DIR}"
  cd "${WORK_DIR}" || exit
  python -u "${SCRIPT_NAME}" -b "${BATCH_SIZE}" \
   -o "${OPTIMIZER}" --optimizer-options "${OPTIMIZER_OPTS}" \
   --initial-weight "${INITIAL_WEIGHT}" | tee log.txt
  cd "${BASEDIR}" || exit
done
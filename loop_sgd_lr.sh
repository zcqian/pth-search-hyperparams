#!/bin/bash

BATCH_SIZE=256
LIST_OF_LEARNING_RATE=(0.1 0.01 0.001 0.0001)
OPTIMIZER="sgd"

BASEDIR=$(pwd)
SCRIPT_NAME="${BASEDIR}/main.py"
INITIAL_WEIGHT="${BASEDIR}/initial_weights.pt"

for LEARNING_RATE in "${LIST_OF_LEARNING_RATE[@]}"; do
  OPTIMIZER_OPTS="{'lr': ${LEARNING_RATE}, 'momentum': 0.9, 'weight_decay': 1e-4}"
  WORK_DIR="./results/sgd-lr/${LEARNING_RATE}"
  mkdir -p "${WORK_DIR}"
  cd "${WORK_DIR}" || exit
  python -u "${SCRIPT_NAME}" -b "${BATCH_SIZE}" \
   -o "${OPTIMIZER}" --optimizer-options "${OPTIMIZER_OPTS}" \
   --initial-weight "${INITIAL_WEIGHT}" | tee log.txt
  cd "${BASEDIR}" || exit
done
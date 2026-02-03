#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-train}" # train | infer
TP_SIZE="${TP_SIZE:-4}"
EP_SIZE="${EP_SIZE:-1}"
MODEL_PATH="${MODEL_PATH:-/Users/jingruyang/Desktop/research_projects/llm_qat_parallel/witch0_7B_custom}"
export TP_SIZE EP_SIZE MODEL_PATH

if [[ -z "${NPROC:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NPROC="1"
  fi
fi

MP_SIZE=$((TP_SIZE * EP_SIZE))
if (( NPROC < MP_SIZE )); then
  echo "ERROR: Need at least TP_SIZE*EP_SIZE GPUs ($MP_SIZE), but NPROC=$NPROC."
  echo "Hint: on 4 GPUs, keep EP_SIZE=1 when TP_SIZE=4."
  exit 1
fi
if (( NPROC % MP_SIZE != 0 )); then
  echo "ERROR: NPROC ($NPROC) must be divisible by TP_SIZE*EP_SIZE ($MP_SIZE)."
  exit 1
fi

if [[ "$MODE" == "train" ]]; then
  torchrun --nproc_per_node="$NPROC" moe_megatron/train_tp_dp_ep_moe.py
elif [[ "$MODE" == "infer" ]]; then
  torchrun --nproc_per_node="$NPROC" moe_megatron/moe_model_tp_dp_ep.py
else
  echo "Unknown MODE=$MODE. Use MODE=train or MODE=infer."
  exit 1
fi

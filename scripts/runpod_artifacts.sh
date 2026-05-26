set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-data}"
OUT_DIR="${OUT_DIR:-outputs}"
DEVICE="${DEVICE:-cuda}"
MODELS="${MODELS:-vit-gpt2,blip,blip2,blip2-lora}"
MODELS_COMPACT="${MODELS//[[:space:]]/}"
BLIP2_MODEL_ID="${BLIP2_MODEL_ID:-Salesforce/blip2-opt-2.7b}"
BLIP2_TORCH_DTYPE="${BLIP2_TORCH_DTYPE:-float16}"
BLIP2_QUANT="${BLIP2_QUANT:-none}"
BLIP2_ADAPTER_PATH="${BLIP2_ADAPTER_PATH:-${FASHION_CAPTION_REMOTE_ADAPTER_PATH:-}}"
REMOTE_URL="${REMOTE_URL:-${FASHION_CAPTION_REMOTE_URL:-}}"
INCLUDE_GPT_FLAG=()
REMOTE_FLAG=()

remind_stop_pod() {
  cat <<'EOF'

============================================================
MONEY SAFETY
Artifact generation is done or stopped.
Go to RunPod -> Pods and Stop/Terminate the pod if you are finished.
============================================================

EOF
}
trap remind_stop_pod EXIT

if [[ ! -f "${DATA_ROOT}/styles.csv" || ! -d "${DATA_ROOT}/images" ]]; then
  echo "Expected ${DATA_ROOT}/styles.csv and ${DATA_ROOT}/images/." >&2
  echo "Unpack the Kaggle dataset before running artifacts." >&2
  exit 2
fi

if [[ ",${MODELS_COMPACT}," == *",blip2-lora,"* && ( -z "$BLIP2_ADAPTER_PATH" || ! -f "${BLIP2_ADAPTER_PATH}/adapter_config.json" ) ]]; then
  echo "BLIP2_ADAPTER_PATH must point to a LoRA adapter directory." >&2
  echo "Expected file: <adapter>/adapter_config.json" >&2
  exit 2
fi

if [[ "${INCLUDE_GPT:-0}" == "1" ]]; then
  INCLUDE_GPT_FLAG=(--include-gpt)
fi

if [[ -n "$REMOTE_URL" ]]; then
  REMOTE_FLAG=(--remote-url "$REMOTE_URL")
fi

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=src
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

python scripts/regenerate_artifacts.py \
  --data-root "$DATA_ROOT" \
  --out-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --sanity-n 20 \
  --eval-n 300 \
  --models "$MODELS_COMPACT" \
  "${REMOTE_FLAG[@]}" \
  --blip2-model-id "$BLIP2_MODEL_ID" \
  --blip2-torch-dtype "$BLIP2_TORCH_DTYPE" \
  --blip2-quant "$BLIP2_QUANT" \
  --blip2-adapter-path "$BLIP2_ADAPTER_PATH" \
  "${INCLUDE_GPT_FLAG[@]}"

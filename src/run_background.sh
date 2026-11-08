#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

mkdir -p logs artifacts reports

echo "=========================================================="
echo "KHỞI CHẠY HUẤN LUYỆN NỀN: PHASE 2 -> PHASE 3"
echo "Thời gian bắt đầu: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python interpreter: $PYTHON_BIN"
echo "=========================================================="

echo "[1/2] Bắt đầu Step 2: Huấn luyện pipeline đa trục & tự kiểm tra..."
"$PYTHON_BIN" -m src.step2_train_pipeline \
    --data_path data/train.csv \
    --text_col Comment \
    --sent_col Sentiment \
    --tox_col Toxicity \
    --model_out artifacts/model_pipeline.joblib \
    --meta_out artifacts/metadata.json \
    --oof_out artifacts/oof_predictions.npz

echo "[2/2] Bắt đầu Step 3: Xuất báo cáo kỹ thuật hoàn chỉnh..."
"$PYTHON_BIN" -m src.step3_generate_report

echo "=========================================================="
echo "HOÀN TẤT THÀNH CÔNG!"
echo "Thời gian kết thúc: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Báo cáo: reports/final_training_report.md"
echo "=========================================================="

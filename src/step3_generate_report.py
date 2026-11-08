#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: Automated Engineering Report Generator (Synced with DualAxisCalibratedClassifier)
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, classification_report


def compute_multiclass_brier(
    y_true_col: np.ndarray, proba_matrix: np.ndarray, classes: list
) -> float:
    brier_scores = []
    for idx, c in enumerate(classes):
        y_bin = (y_true_col == c).astype(int)
        brier_scores.append(brier_score_loss(y_bin, proba_matrix[:, idx]))
    return float(np.mean(brier_scores))


def generate_markdown_report(
    meta_path: str = "artifacts/metadata.json",
    oof_path: str = "artifacts/oof_predictions.npz",
    out_md_path: str = "reports/final_training_report.md",
) -> None:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    oof = np.load(oof_path)

    y_true = oof["y_true"]
    y_pred = oof["oof_preds"]
    proba_sent = oof["oof_proba_sent"]
    proba_tox = oof["oof_proba_tox"]

    brier_sent = compute_multiclass_brier(y_true[:, 0], proba_sent, [-1, 0, 1])
    brier_tox = compute_multiclass_brier(y_true[:, 1], proba_tox, [0, 1, 2, 3])

    sent_report = classification_report(
        y_true[:, 0],
        y_pred[:, 0],
        labels=[-1, 0, 1],
        target_names=[
            "-1: Tiêu cực (Negative)",
            " 0: Trung tính (Neutral)",
            " 1: Tích cực (Positive)",
        ],
        digits=4,
        zero_division=0,
    )
    tox_report = classification_report(
        y_true[:, 1],
        y_pred[:, 1],
        labels=[0, 1, 2, 3],
        target_names=[
            "0: Không độc hại",
            "1: Thiếu lịch sự",
            "2: Xúc phạm cá nhân",
            "3: Thù ghét/Đe dọa",
        ],
        digits=4,
        zero_division=0,
    )

    ft = meta["full_train_stats"]
    hp = meta["best_hyperparameters"]
    oof_m = meta["oof_summary_metrics"]

    md_lines = [
        "# BÁO CÁO KỸ THUẬT: HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI ĐA TRỤC TIẾNG VIỆT (29,474 GOLD SAMPLES)",
        "",
        "## 1. Thống kê Không gian Đặc trưng & Lọc Pareto 75% (Full Train 100%)",
        f"- **Tổng số mẫu tập Train sạch ($N_{{train}}$):** `{ft['n_train']:,}`",
        f"- **Số lượng Word n-gram `(1,2)` sạch (`min_df=2`):** `{ft['v_word_clean']:,}`",
        f"- **Số lượng Char n-gram `(2,5)` sạch (`min_df=2`):** `{ft['v_char_clean']:,}`",
        f"- **Tổng từ vựng sạch ($V_{{total}}$):** `{ft['v_total_clean']:,}`",
        f"- **Mốc 75% Pareto ($0.75 \\times V_{{total}}$):** `{ft['pareto_75_raw']:,}`",
        f"- **Khoảng biên cho phép $[1 \\times N_{{train}}, 3 \\times N_{{train}}]$:** `[{ft['bound_constraint_range'][0]:,}, {ft['bound_constraint_range'][1]:,}]`",
        f"- **Số cột `max_features` cuối cùng được chọn:** **`{ft['final_max_features']:,}`** (`{ft['bound_status']}`)",
        f"- **Thời gian Re-fit trên 100% tập Train:** `{ft['refit_time_sec']} giây`",
        "",
        "## 2. Siêu tham số Tối ưu & Điểm Tổng kết 5-Fold Out-of-Fold (OOF)",
        f"- **Trục 1 (Sentiment `{-1, 0, 1}`):** `C_sent* = {hp['C_sentiment']}` | **OOF Macro F1 = `{oof_m['oof_macro_f1_sentiment']:.4f}`**",
        f"- **Trục 2 (Toxicity `{0, 1, 2, 3}`):** `C_tox* = {hp['C_toxicity']}`, `alpha* = {hp['alpha_toxicity']}`",
        f"  - Trọng số lớp giảm chấn mũ đã áp dụng: `{hp['toxicity_class_weights']}`",
        f"  - Ngưỡng hậu Platt Scaling (`tox_thresholds`): `{hp['toxicity_thresholds']}`",
        f"  - **OOF Macro F1 (Toxicity):** **`{oof_m['oof_macro_f1_toxicity_final']:.4f}`** (trước khi chỉnh ngưỡng: `{oof_m['oof_macro_f1_toxicity_raw_argmax']:.4f}`)",
        f"- **Combined Macro F1 (Trung bình 2 trục):** **`{oof_m['oof_macro_f1_combined']:.4f}`**",
        "",
        "### 2.1. Bảng Quét `C` trên Trục 1 (Sentiment)",
        "| Tham số `C` | Train Macro F1 | Val Macro F1 (Mean ± Std) | Overfit Gap | Selection Score | Thời gian (s) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in meta["sentiment_cv_results"]:
        mark = "**" if r["C"] == hp["C_sentiment"] else ""
        md_lines.append(
            f"| {mark}{r['C']}{mark} | {r['mean_train_f1']:.4f} | {mark}{r['mean_val_f1']:.4f} ± {r['std_val_f1']:.4f}{mark} | "
            f"{r['overfit_gap']:.4f} | {mark}{r['penalized_selection_score']:.4f}{mark} | {r['elapsed_sec']}s |"
        )

    md_lines.extend(
        [
            "",
            "### 2.2. Bảng Quét `(alpha, C)` trên Trục 2 (Toxicity)",
            "| Hệ số `alpha` | Tham số `C` | Train Macro F1 | Val Macro F1 (Mean ± Std) | Overfit Gap | Selection Score | Thời gian (s) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for r in meta["toxicity_cv_results"]:
        is_best = (r["C"] == hp["C_toxicity"]) and (r["alpha"] == hp["alpha_toxicity"])
        mark = "**" if is_best else ""
        md_lines.append(
            f"| {mark}{r['alpha']}{mark} | {mark}{r['C']}{mark} | {r['mean_train_f1']:.4f} | "
            f"{mark}{r['mean_val_f1']:.4f} ± {r['std_val_f1']:.4f}{mark} | {r['overfit_gap']:.4f} | "
            f"{mark}{r['penalized_selection_score']:.4f}{mark} | {r['elapsed_sec']}s |"
        )

    md_lines.extend(
        [
            "",
            "## 3. Báo cáo Chi tiết Từng Lớp & Độ tin cậy Platt Scaling (5-Fold OOF)",
            f"### 3.1. Trục 1 - Sentiment (Brier Score Loss = `{brier_sent:.5f}`)",
            "```text",
            sent_report.strip(),
            "```",
            f"### 3.2. Trục 2 - Toxicity (Brier Score Loss = `{brier_tox:.5f}`)",
            "```text",
            tox_report.strip(),
            "```",
            "",
            "## 4. Kết quả Bộ Tự Kiểm Tra 7 Bước (Self-Verification Suite)",
        ]
    )

    for k, v in meta["verification_results"].items():
        md_lines.append(f"- **{k}:** `{v}`")

    md_lines.extend(
        [
            "",
            "## 5. Mã kiểm tra Toàn vẹn Mô hình (Artifact Integrity)",
            f"- **SHA256 (`artifacts/model_pipeline.joblib`):** `{meta['model_artifact_sha256']}`",
        ]
    )

    Path(out_md_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[SUCCESS] Đã xuất báo cáo hoàn chỉnh tại: {out_md_path}")


if __name__ == "__main__":
    generate_markdown_report()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4: Đánh giá một hoặc nhiều model đã train trên tập holdout (mặc định valid_data/valid.csv).
Chỉ có ý nghĩa khi holdout KHÔNG nằm trong dữ liệu train của model.

Ví dụ:
  python -m src.step4_holdout_eval \
      --models artifacts/train_only_penalized/model_pipeline.joblib artifacts/train_only_maxval/model_pipeline.joblib \
      --holdout valid_data/valid.csv --out reports/holdout_eval.md
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

import __main__
import src.step2_train_pipeline as _s2

# Tương thích ngược: artifact cũ được pickle dưới tên `__main__.X` (khi step2 chạy bằng `python -m`).
for _name in ("MinimalVietnameseCleaner", "DynamicParetoTfidfExtractor", "DualAxisCalibratedClassifier",
              "social_word_tokenizer"):
    setattr(__main__, _name, getattr(_s2, _name))

SENT_NAMES = ["-1 Negative", " 0 Neutral", " 1 Positive"]
TOX_NAMES = ["0 Không độc hại", "1 Thiếu lịch sự", "2 Xúc phạm", "3 Thù ghét/Đe dọa"]


def evaluate(model_path: str, df: pd.DataFrame) -> dict:
    pipe = joblib.load(model_path)
    texts = df["Comment"].astype(str).tolist()
    y = df[["Sentiment", "Toxicity"]].to_numpy(dtype=int)
    preds = pipe.predict(texts)
    proba_tox = pipe.predict_proba(texts)[1]
    raw_tox = pipe.named_steps["classifier"].classes_[1][np.argmax(proba_tox, axis=1)]

    f1_s = f1_score(y[:, 0], preds[:, 0], average="macro", labels=[-1, 0, 1], zero_division=0)
    f1_t = f1_score(y[:, 1], preds[:, 1], average="macro", labels=[0, 1, 2, 3], zero_division=0)
    f1_t_raw = f1_score(y[:, 1], raw_tox, average="macro", labels=[0, 1, 2, 3], zero_division=0)
    return {
        "model": model_path,
        "n": len(df),
        "macro_f1_sentiment": round(float(f1_s), 4),
        "macro_f1_toxicity": round(float(f1_t), 4),
        "macro_f1_toxicity_raw_argmax": round(float(f1_t_raw), 4),
        "macro_f1_combined": round(float((f1_s + f1_t) / 2), 4),
        "tox_per_class_f1": [round(float(v), 4) for v in f1_score(
            y[:, 1], preds[:, 1], average=None, labels=[0, 1, 2, 3], zero_division=0)],
        "sent_report": classification_report(y[:, 0], preds[:, 0], labels=[-1, 0, 1],
                                             target_names=SENT_NAMES, digits=4, zero_division=0),
        "tox_report": classification_report(y[:, 1], preds[:, 1], labels=[0, 1, 2, 3],
                                            target_names=TOX_NAMES, digits=4, zero_division=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--holdout", default="valid_data/valid.csv")
    ap.add_argument("--out", default="reports/holdout_eval.md")
    a = ap.parse_args()

    df = pd.read_csv(a.holdout, encoding="utf-8-sig")
    results = [evaluate(m, df) for m in a.models]

    lines = [f"# Đánh giá Holdout: `{a.holdout}` ({len(df):,} mẫu)", "",
             "| Model | Sent F1 | Tox F1 (thresholded) | Tox F1 (argmax) | Combined | Tox per-class F1 [0,1,2,3] |",
             "| :--- | :---: | :---: | :---: | :---: | :---: |"]
    for r in results:
        meta_path = Path(r["model"]).with_name("metadata.json")
        hp = json.loads(meta_path.read_text())["best_hyperparameters"] if meta_path.exists() else {}
        lines.append(
            f"| `{r['model']}`<br>C_sent={hp.get('C_sentiment')}, C_tox={hp.get('C_toxicity')}, alpha={hp.get('alpha_toxicity')} "
            f"| {r['macro_f1_sentiment']:.4f} | {r['macro_f1_toxicity']:.4f} | {r['macro_f1_toxicity_raw_argmax']:.4f} "
            f"| **{r['macro_f1_combined']:.4f}** | {r['tox_per_class_f1']} |"
        )
    for r in results:
        lines += ["", f"## `{r['model']}`", "```text", r["sent_report"].rstrip(), "", r["tox_report"].rstrip(), "```"]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r in results:
        print({k: v for k, v in r.items() if not k.endswith("_report")})
    print(f"[SUCCESS] {a.out}")


if __name__ == "__main__":
    main()

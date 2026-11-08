#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: Pre-Flight Data Audit Script for Vietnamese Multi-Axis Classification
Outputs deterministic JSON report for Senior ML Engineer review.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Regex Patterns theo đúng yêu cầu Minimal Cleaning
URL_PATTERN = re.compile(
    r"(?:https?://|ftp://|www\.)\S+|(?<!\w)[a-zA-Z0-9.-]+\.(?:com|vn|net|org|edu|gov|io|ly|me)(?:/\S*)?",
    re.IGNORECASE,
)
HTML_PATTERN = re.compile(r"<[^>]+>")
TIMESTAMP_PATTERN = re.compile(
    r"\b(?:\d{1,2}[:h]\d{2}(?::\d{2})?(?:\s?[ap]m)?|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
    re.IGNORECASE,
)
SYS_ERROR_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd\u200b\u200c\u200d\ufeff]"
)
MULTISPACE_PATTERN = re.compile(r"\s+")


def minimal_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = HTML_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = TIMESTAMP_PATTERN.sub(" ", text)
    text = SYS_ERROR_PATTERN.sub("", text)
    return MULTISPACE_PATTERN.sub(" ", text).strip()


def social_word_tokenizer(text: str) -> List[str]:
    tokens = [tok.strip('",;:!?()[]{}<>«»') for tok in text.split()]
    return [t for t in tokens if t] or ["<empty>"]


def run_audit(
    data_path: str, text_col: str, sent_col: str, tox_col: str, out_path: str
) -> None:
    if not os.path.exists(data_path):
        print(f"[CRITICAL_FAIL] File không tồn tại: {data_path}")
        sys.exit(1)

    # 1. Load Data (Hỗ trợ CSV, Parquet, JSONL)
    ext = Path(data_path).suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(data_path)
    elif ext in [".json", ".jsonl"]:
        df = pd.read_json(data_path, lines=(ext == ".jsonl"))
    else:
        df = pd.read_csv(data_path, encoding="utf-8-sig")

    n_raw = len(df)
    actual_cols = list(df.columns)
    for req_col in [text_col, sent_col, tox_col]:
        if req_col not in actual_cols:
            print(
                json.dumps(
                    {
                        "audit_status": "CRITICAL_FAIL",
                        "reason": f"Không tìm thấy cột '{req_col}'. Các cột hiện có: {actual_cols}",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            sys.exit(1)

    issues_critical, issues_warn = [], []

    # 2. Kiểm tra Null & Chuỗi rỗng ban đầu
    null_text = int(df[text_col].isna().sum())
    null_sent = int(df[sent_col].isna().sum())
    null_tox = int(df[tox_col].isna().sum())
    empty_raw = int(df[text_col].dropna().astype(str).str.strip().eq("").sum())

    if null_sent > 0 or null_tox > 0:
        issues_critical.append(
            f"Cột nhãn có giá trị Null (sentiment={null_sent}, toxicity={null_tox})"
        )
    if (null_text + empty_raw) / max(1, n_raw) > 0.005:
        issues_critical.append(
            f"Tỷ lệ text Null/rỗng vượt quá 0.5%: {null_text + empty_raw}/{n_raw}"
        )
    elif (null_text + empty_raw) > 0:
        issues_warn.append(
            f"Có {null_text + empty_raw} dòng text Null/rỗng cần loại bỏ trước khi train."
        )

    df_valid = df.dropna(subset=[text_col, sent_col, tox_col]).copy()
    df_valid[text_col] = df_valid[text_col].astype(str)
    df_valid = df_valid[df_valid[text_col].str.strip() != ""].reset_index(drop=True)
    n_train = len(df_valid)

    # 3. Kiểm tra Miền giá trị Nhãn & Phân phối (Cho phép {-1, 0, 1} hoặc {0, 1, 2})
    sent_vals = sorted(df_valid[sent_col].unique().tolist())
    tox_vals = sorted(df_valid[tox_col].unique().tolist())
    allowed_sent = {-1, 0, 1} if -1 in sent_vals else {0, 1, 2}
    if not set(sent_vals).issubset(allowed_sent):
        issues_critical.append(
            f"Nhãn Sentiment chứa giá trị ngoài {allowed_sent}: {sent_vals}"
        )
    if not set(tox_vals).issubset({0, 1, 2, 3}):
        issues_critical.append(
            f"Nhãn Toxicity chứa giá trị ngoài {{0,1,2,3}}: {tox_vals}"
        )

    sent_counts = df_valid[sent_col].value_counts().to_dict()
    tox_counts = df_valid[tox_col].value_counts().to_dict()
    sent_ir = round(max(sent_counts.values()) / max(1, min(sent_counts.values())), 2)
    tox_ir = round(max(tox_counts.values()) / max(1, min(tox_counts.values())), 2)

    if min(list(sent_counts.values()) + list(tox_counts.values())) < 25:
        issues_critical.append(
            "Có lớp đơn lẻ dưới 25 mẫu, không đủ cho 5-Fold CV x 5-Fold Platt Scaling."
        )

    # Ma trận tương quan chéo 3x4
    joint_df = pd.crosstab(df_valid[sent_col], df_valid[tox_col])
    joint_matrix = {
        f"sent_{r}_tox_{c}": int(joint_df.loc[r, c])
        for r in joint_df.index
        for c in joint_df.columns
    }
    min_joint = min(joint_matrix.values()) if joint_matrix else 0
    if min_joint < 25:
        issues_warn.append(
            f"Tổ hợp nhãn nhỏ nhất chỉ có {min_joint} mẫu (<25). Cần lưu ý khi chia StratifiedKFold."
        )

    # 4. Kiểm tra Unicode (NFC vs NFD) & Tác động của Minimal Cleaning
    raw_texts = df_valid[text_col].tolist()
    nfd_count = sum(1 for t in raw_texts if unicodedata.normalize("NFC", t) != t)
    if nfd_count > 0:
        issues_warn.append(
            f"Phát hiện {nfd_count}/{n_train} câu dùng Unicode tổ hợp (NFD). Bắt buộc chuẩn hóa NFC."
        )

    cleaned_texts = [minimal_clean(t) for t in raw_texts]
    empty_after_clean = sum(1 for t in cleaned_texts if t == "")
    if empty_after_clean / max(1, n_train) > 0.01:
        issues_critical.append(
            f"Có {empty_after_clean} câu (>1%) bị biến thành rỗng sau khi xóa URL/HTML/Timestamp."
        )
    elif empty_after_clean > 0:
        issues_warn.append(
            f"Có {empty_after_clean} câu bị biến thành rỗng sau khi làm sạch tối thiểu."
        )

    # 5. Kiểm tra Trùng lặp & Mâu thuẫn nhãn (Label Contradiction)
    df_valid["_clean_text"] = cleaned_texts
    dup_total = int(df_valid.duplicated(subset=["_clean_text"]).sum())
    label_nunique_per_text = df_valid.groupby("_clean_text")[
        [sent_col, tox_col]
    ].nunique()
    contradict_texts = int(
        (
            (label_nunique_per_text[sent_col] > 1)
            | (label_nunique_per_text[tox_col] > 1)
        ).sum()
    )

    if dup_total / max(1, n_train) > 0.05:
        issues_warn.append(
            f"Tỷ lệ trùng lặp văn bản cao: {dup_total}/{n_train} ({dup_total / n_train * 100:.2f}%)."
        )
    if contradict_texts > 0:
        issues_warn.append(
            f"Phát hiện {contradict_texts} văn bản giống hệt nhau nhưng bị gán nhãn mâu thuẫn nhau!"
        )

    # 6. Thống kê từ vựng V_total & Kiểm tra Ràng buộc Pareto 75% vs [1*N_train, 3*N_train]
    safe_texts = [t if t else "<empty>" for t in cleaned_texts]
    word_vec = TfidfVectorizer(
        analyzer="word",
        tokenizer=social_word_tokenizer,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
    )
    char_vec = TfidfVectorizer(
        analyzer="char", lowercase=False, ngram_range=(2, 5), min_df=2
    )

    X_w = word_vec.fit_transform(safe_texts)
    X_c = char_vec.fit_transform(safe_texts)

    v_word = X_w.shape[1]
    v_char = X_c.shape[1]
    v_total = v_word + v_char
    pareto_75 = int(np.floor(0.75 * v_total))
    bound_low = 1 * n_train
    bound_high = 3 * n_train

    if v_total < bound_low:
        issues_critical.append(
            f"V_total ({v_total}) < 1 * N_train ({bound_low}). Không đủ từ vựng thỏa mãn cận dưới!"
        )
    elif pareto_75 < bound_low:
        issues_warn.append(
            f"0.75 * V_total ({pareto_75}) < 1 * N_train ({bound_low}). Sẽ bị kẹp ở cận dưới 1*N_train."
        )
    elif pareto_75 > bound_high:
        issues_warn.append(
            f"0.75 * V_total ({pareto_75}) > 3 * N_train ({bound_high}). Sẽ bị cắt ở cận trên 3*N_train."
        )

    nnz_total = X_w.nnz + X_c.nnz
    est_ram_mb = round((nnz_total * 8) / (1024 * 1024), 2)  # float32 + int32 indices

    status = (
        "CRITICAL_FAIL"
        if issues_critical
        else ("PASS_WITH_WARNINGS" if issues_warn else "PASS")
    )

    report: Dict[str, Any] = {
        "audit_status": status,
        "dataset_summary": {
            "n_raw_rows": n_raw,
            "n_valid_train": n_train,
            "null_or_empty_raw": null_text + empty_raw,
            "empty_after_minimal_clean": empty_after_clean,
            "nfd_unicode_rows": nfd_count,
            "exact_duplicate_rows": dup_total,
            "contradicting_label_groups": contradict_texts,
        },
        "label_distribution": {
            "sentiment_counts": {str(k): int(v) for k, v in sent_counts.items()},
            "sentiment_imbalance_ratio": sent_ir,
            "toxicity_counts": {str(k): int(v) for k, v in tox_counts.items()},
            "toxicity_imbalance_ratio": tox_ir,
            "joint_3x4_counts": joint_matrix,
            "min_joint_cell_count": min_joint,
        },
        "vocabulary_and_pareto_check": {
            "v_word_1_2_min_df_2": v_word,
            "v_char_2_5_min_df_2": v_char,
            "v_total_clean": v_total,
            "pareto_75_percent_features": pareto_75,
            "required_bound_1N_to_3N": [bound_low, bound_high],
            "effective_max_features": min(
                v_total, max(bound_low, min(pareto_75, bound_high))
            ),
            "estimated_sparse_matrix_ram_mb": est_ram_mb,
        },
        "critical_issues": issues_critical,
        "warnings": issues_warn,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status == "CRITICAL_FAIL":
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="train_data/train.csv")
    parser.add_argument("--text_col", type=str, default="Comment")
    parser.add_argument("--sent_col", type=str, default="Sentiment")
    parser.add_argument("--tox_col", type=str, default="Toxicity")
    parser.add_argument(
        "--out_path", type=str, default="reports/data_audit_report.json"
    )
    args = parser.parse_args()
    run_audit(args.data_path, args.text_col, args.sent_col, args.tox_col, args.out_path)

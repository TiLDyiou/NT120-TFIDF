#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1.5 (Part A): Deduplicate Consistent Rows & Export Conflicting/Suspicious Rows for Human Review
"""

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

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


def export_for_review(
    data_path: str,
    text_col: str,
    sent_col: str,
    tox_col: str,
    clean_pool_out: str = "data/train_non_conflicting_deduped.csv",
    review_out: str = "data/conflicts_for_review.csv",
) -> None:
    ext = Path(data_path).suffix.lower()
    df = (
        pd.read_parquet(data_path)
        if ext == ".parquet"
        else pd.read_csv(data_path, encoding="utf-8-sig")
    )

    n_initial = len(df)
    df = df.dropna(subset=[text_col, sent_col, tox_col]).copy()
    df[sent_col] = df[sent_col].astype(int)
    df[tox_col] = df[tox_col].astype(int)
    df[text_col] = (
        df[text_col].astype(str).apply(lambda x: unicodedata.normalize("NFC", x))
    )
    df["_clean_text"] = df[text_col].apply(minimal_clean)

    # 1. Loại bỏ các dòng rỗng sau khi xóa URL/HTML/Timestamp
    n_empty = int((df["_clean_text"] == "").sum())
    df = df[df["_clean_text"] != ""].reset_index(drop=True)

    # 2. Xác định các nhóm văn bản có nhãn mâu thuẫn nhau
    label_nunique = df.groupby("_clean_text")[[sent_col, tox_col]].nunique()
    conflict_texts = set(
        label_nunique[
            (label_nunique[sent_col] > 1) | (label_nunique[tox_col] > 1)
        ].index
    )

    df_conflict_raw = df[df["_clean_text"].isin(conflict_texts)].copy()
    df_consistent = df[~df["_clean_text"].isin(conflict_texts)].copy()

    # 3. Xóa trùng lặp cho các văn bản đồng nhất nhãn (Consistent Duplicates)
    n_before_dedup = len(df_consistent)
    df_consistent_deduped = df_consistent.drop_duplicates(
        subset=["_clean_text"], keep="first"
    ).reset_index(drop=True)
    n_consistent_dups_removed = n_before_dedup - len(df_consistent_deduped)

    # 4. Tách thêm các dòng có tổ hợp nhãn cực hiếm/đáng ngờ trong tập đồng nhất:
    # (sent=1 & tox=3: 1 dòng) hoặc (sent=0 & tox=2: 12 dòng) để người dùng tiện kiểm tra
    suspicious_mask = (
        (df_consistent_deduped[sent_col] == 1) & (df_consistent_deduped[tox_col] == 3)
    ) | ((df_consistent_deduped[sent_col] == 0) & (df_consistent_deduped[tox_col] == 2))
    df_suspicious = df_consistent_deduped[suspicious_mask].copy()
    df_clean_pool = (
        df_consistent_deduped[~suspicious_mask]
        .drop(columns=["_clean_text"])
        .reset_index(drop=True)
    )

    # 5. Tổng hợp bảng Review gọn gàng (1 dòng / 1 câu)
    review_rows = []

    # 5a. Đưa các nhóm mâu thuẫn vào bảng Review
    for idx, (c_text, grp) in enumerate(
        df_conflict_raw.groupby("_clean_text"), start=1
    ):
        vc = (
            grp.groupby([sent_col, tox_col])
            .size()
            .reset_index(name="count")
            .sort_values(by=["count", tox_col], ascending=[False, False])
        )
        breakdown_str = " | ".join(
            [
                f"(sent={r[sent_col]}, tox={r[tox_col]}): {r['count']} lần"
                for _, r in vc.iterrows()
            ]
        )
        is_tie = len(vc) > 1 and (vc.iloc[0]["count"] == vc.iloc[1]["count"])

        review_rows.append(
            {
                "review_id": f"CONF_{idx:03d}",
                "review_type": "TIE_CONFLICT" if is_tie else "MAJORITY_CONFLICT",
                "total_occurrences": len(grp),
                "label_breakdown": breakdown_str,
                "clean_text": c_text,
                "raw_text_sample": grp[text_col].iloc[0],
                "final_sentiment": int(vc.iloc[0][sent_col]),
                "final_toxicity": int(vc.iloc[0][tox_col]),
                "action": "KEEP",  # Đổi thành DROP nếu muốn xóa bỏ câu này
            }
        )

    # 5b. Đưa các dòng nhãn chéo hiếm vào cuối bảng Review
    for idx, (_, row) in enumerate(df_suspicious.iterrows(), start=1):
        review_rows.append(
            {
                "review_id": f"SUSP_{idx:03d}",
                "review_type": "RARE_JOINT_LABEL_CHECK",
                "total_occurrences": 1,
                "label_breakdown": f"(sent={row[sent_col]}, tox={row[tox_col]}): 1 lần (Tổ hợp hiếm)",
                "clean_text": row["_clean_text"],
                "raw_text_sample": row[text_col],
                "final_sentiment": int(row[sent_col]),
                "final_toxicity": int(row[tox_col]),
                "action": "KEEP",
            }
        )

    df_review = pd.DataFrame(review_rows)
    # Sắp xếp ưu tiên TIE_CONFLICT (hòa phiếu) lên đầu để bạn chú ý nhất
    df_review = df_review.sort_values(
        by=["review_type", "total_occurrences"], ascending=[False, False]
    ).reset_index(drop=True)

    Path(clean_pool_out).parent.mkdir(parents=True, exist_ok=True)
    Path(review_out).parent.mkdir(parents=True, exist_ok=True)
    df_clean_pool.to_csv(clean_pool_out, index=False, encoding="utf-8-sig")
    df_review.to_csv(review_out, index=False, encoding="utf-8-sig")

    summary = {
        "n_initial_rows": n_initial,
        "n_empty_dropped": n_empty,
        "n_consistent_duplicates_auto_removed": n_consistent_dups_removed,
        "n_conflicting_raw_rows_isolated": len(df_conflict_raw),
        "n_unique_conflict_groups_to_review": len(conflict_texts),
        "n_suspicious_joint_rows_to_review": len(df_suspicious),
        "n_clean_non_conflicting_saved": len(df_clean_pool),
        "clean_pool_file": clean_pool_out,
        "human_review_file": review_out,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="train_data/train.csv")
    parser.add_argument("--text_col", type=str, default="Comment")
    parser.add_argument("--sent_col", type=str, default="Sentiment")
    parser.add_argument("--tox_col", type=str, default="Toxicity")
    parser.add_argument(
        "--clean_pool_out", type=str, default="data/train_non_conflicting_deduped.csv"
    )
    parser.add_argument(
        "--review_out", type=str, default="data/conflicts_for_review.csv"
    )
    args = parser.parse_args()
    export_for_review(
        args.data_path,
        args.text_col,
        args.sent_col,
        args.tox_col,
        args.clean_pool_out,
        args.review_out,
    )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1.5 (Part B): Merge Clean Deduplicated Pool with Human-Reviewed Conflicts/Suspicious Rows
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


def merge_datasets(
    clean_pool_path: str = "data/train_non_conflicting_deduped.csv",
    review_path: str = "data/conflicts_for_review.csv",
    raw_reference_path: str = "train_data/train.csv.bak",
    out_paths: list = ["train_data/train.csv", "data/train.csv"],
) -> None:
    df_clean = pd.read_csv(clean_pool_path, encoding="utf-8-sig")
    df_review = pd.read_csv(review_path, encoding="utf-8-sig")

    # Lọc action == 'DROP' nếu có
    if "action" in df_review.columns:
        df_review = df_review[df_review["action"] != "DROP"]

    # Tra cứu câu gốc ban đầu
    raw_ref = (
        raw_reference_path
        if os.path.exists(raw_reference_path)
        else "train_data/train.csv"
    )
    df_raw = pd.read_csv(raw_ref, encoding="utf-8-sig")
    df_raw["Comment"] = (
        df_raw["Comment"].astype(str).apply(lambda x: unicodedata.normalize("NFC", x))
    )
    df_raw["_clean_text"] = df_raw["Comment"].apply(minimal_clean)
    raw_lookup = (
        df_raw.drop_duplicates(subset=["_clean_text"])
        .set_index("_clean_text")["Comment"]
        .to_dict()
    )

    reviewed_rows = []
    for _, r in df_review.iterrows():
        c_text = r["clean_text"]
        raw_com = raw_lookup.get(c_text, c_text)
        reviewed_rows.append(
            {
                "Comment": raw_com,
                "Sentiment": int(r["final_sentiment"]),
                "Toxicity": int(r["final_toxicity"]),
            }
        )

    df_reviewed_prepared = pd.DataFrame(reviewed_rows)

    # Gộp 2 tập
    df_merged = pd.concat(
        [df_clean[["Comment", "Sentiment", "Toxicity"]], df_reviewed_prepared],
        ignore_index=True,
    )
    df_merged["Comment"] = (
        df_merged["Comment"]
        .astype(str)
        .apply(lambda x: unicodedata.normalize("NFC", x))
    )

    for p in out_paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        df_merged.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"Đã lưu tập train sạch hoàn chỉnh tại: {p} ({len(df_merged)} dòng)")


if __name__ == "__main__":
    merge_datasets()

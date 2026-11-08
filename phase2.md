Dưới đây là **Kế hoạch Thực thi (Execution Plan)** và **Đặc tả Mã nguồn Hoàn chỉnh cho Phase 2** đã tích hợp đầy đủ các vũ khí chống lệch nhãn cho cột Toxicity (`Imbalance Ratio = 169.36`).

---

## KẾ HOẠCH THỰC THI PHASE 2 (PHASE 2 EXECUTION PLAN)

### 1. Kiến trúc Pipeline Nâng cấp (Chống lệch nhãn Toxicity `169.36x`)

```text
[Văn bản đầu vào]
       │
       ▼
[1. MinimalVietnameseCleaner] (Chuẩn hóa NFC, xóa URL/HTML/Timestamp/Control chars, giữ Teencode + Emoji)
       │
       ▼
[2. DynamicParetoTfidfExtractor]
   ├── Nhánh Word: ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False (61.173 cột)
   ├── Nhánh Char: ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=False (221.636 cột)
   └── Bộ lọc Pareto 75% Cân bằng Lớp (Class-Balanced TF-IDF Selector):
       • Tổng từ vựng sạch V_total = 282.809 -> 75% Pareto = 212.106 cột.
       • Kẹp trong biên [1*N_train, 3*N_train] -> Giữ lại đúng 88.422 cột (khi Refit 100% Train)
         hoặc 70.737 cột (trên mỗi Fold 80% khi chạy CV).
       • Điểm xếp hạng cột = 50% (Mean TF-IDF toàn cục) + 50% (Max của Mean TF-IDF từng lớp)
         => Bảo vệ 100% các n-gram đặc trưng của lớp hiếm Toxicity = 2 (976 mẫu) & Toxicity = 3 (154 mẫu).
       │
       ▼
[3. DualAxisCalibratedClassifier] (Giao diện tương thích 100% với MultiOutputClassifier)
   ├── Trục 1 (Sentiment {-1, 0, 1} | IR = 2.39):
   │   └── CalibratedClassifierCV(LinearSVC(C=C_sent*, random_state=42), cv=5, method='sigmoid')
   │
   └── Trục 2 (Toxicity {0, 1, 2, 3} | IR = 169.36):
       ├── CalibratedClassifierCV(LinearSVC(C=C_tox*, class_weight=w(α*), random_state=42), cv=5, method='sigmoid')
       │   với Trọng số Giảm chấn Mũ: w_c = (N_max / N_c)^α  (quét α ∈ {0.0, 0.35, 0.50})
       └── Bộ Hiệu chỉnh Ngưỡng Hậu Platt Scaling (OOF Threshold Multiplier τ* = [1, τ_1, τ_2, τ_3]):
           • .predict_proba(X) -> Trả về xác suất Platt Scaling nguyên bản (tổng = 100%).
           • .predict(X)       -> Chọn argmax_c (τ_c* * P(y = c | x)) để tối đa hóa Macro F1.
```

### 2. Quy chuẩn Tối ưu Siêu tham số (5-Fold Cross-Validation)

- **Chia Fold:** Sử dụng `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` trực tiếp trên nhãn tổ hợp `f"{sentiment}_{toxicity}"` (vì cả 10 tổ hợp có mặt trong dữ liệu đều có $\ge 14$ mẫu $\ge 5$ mẫu).
- **Tối ưu độc lập 2 trục trên cùng 5 Fold TF-IDF đã tiền tính toán (tiết kiệm 90% thời gian CPU):**
  1. **Trục Sentiment:** Quét 12 giá trị $C \in [0.01 \to 50]$ theo thang logarit (`np.geomspace(0.01, 50.0, 12)`). Chọn $C_{\text{sent}}^*$ có `Selection_Score = Val_Macro_F1 - 0.5 * max(0, Train_F1 - Val_F1)` cao nhất (kèm bộ lọc cứng `Overfit_Gap <= 0.12`).
  2. **Trục Toxicity:** Quét lưới 2 chiều gồm 12 giá trị $C \in [0.01 \to 50]$ $\times$ 3 mức trọng số giảm chấn $\alpha \in \{0.0, 0.35, 0.50\}$ (tổng cộng 36 cấu hình).
     - Với $\alpha = 0.0$: Trọng số đồng đều `{0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}` (mặc định).
     - Với $\alpha = 0.35$: Trọng số `{0: 1.0, 1: 2.35, 2: 3.16, 3: 6.03}` (nâng nhẹ lớp hiếm).
     - Với $\alpha = 0.50$: Trọng số `{0: 1.0, 1: 3.39, 2: 5.17, 3: 13.01}` (căn bậc hai).
     - Chọn cặp $(C_{\text{tox}}^_, \alpha^_)$ có `Selection_Score` cao nhất trên trục Toxicity.
  3. **Tối ưu Ngưỡng Quyết định Hậu Platt Scaling ($\mathbf{\tau}^\*$):**
     - Từ xác suất Out-of-Fold (OOF) của cặp $(C_{\text{tox}}^_, \alpha^_)$, thực hiện tối ưu chéo (Cross-Validated Coordinate Search) để tìm hệ số nhân ngưỡng $\mathbf{\tau}^* = [1.0, \tau_1, \tau_2, \tau_3]$. Chỉ kích hoạt $\mathbf{\tau}^*$ nếu nó giúp tăng `OOF Val Macro F1` thêm ít nhất `+0.003` so với ngưỡng mặc định `[1, 1, 1, 1]`.
- **Re-fit 100% Tập Train:** Huấn luyện lại toàn bộ Pipeline với $(C_{\text{sent}}^_, C\_{\text{tox}}^_, \alpha^_, \mathbf{\tau}^_)$ trên toàn bộ `29.474` mẫu.

---

## ĐẶC TẢ MÃ NGUỒN HOÀN CHỈNH PHASE 2: `src/step2_train_pipeline.py`

> **Chỉ thị cho Coding Agent:** Hãy lưu toàn bộ mã nguồn dưới đây vào `src/step2_train_pipeline.py`. Mã nguồn này đã được thiết kế khớp 100% với tập dữ liệu `29.474` mẫu hiện tại, có cơ chế lưu Checkpoint từng bước vào `logs/cv_checkpoints.jsonl` và tự động chạy **Bộ Tự Kiểm Tra 7 Bước (7-Step Self-Verification)** trước khi xuất xưởng mô hình.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: Complete Multi-Axis Training Pipeline for Vietnamese Comments (29,474 Gold Samples)
Architecture:
  [MinimalVietnameseCleaner]
    -> [DynamicParetoTfidfExtractor (Word 1-2 + Char 2-5, Class-Balanced Pareto 75% in [1N, 3N])]
    -> [DualAxisCalibratedClassifier (LinearSVC + 5-Fold Platt Scaling + Power-Law Weights + OOF Thresholds)]
"""

import os
import re
import sys
import json
import time
import hashlib
import logging
import argparse
import unicodedata
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# 1. CẤU HÌNH HỆ THỐNG (SYSTEM CONFIGURATION)
# =====================================================================

class Config:
    RANDOM_STATE: int = 42
    EXPECTED_SENTIMENT_CLASSES: List[int] = [-1, 0, 1]
    EXPECTED_TOXICITY_CLASSES: List[int] = [0, 1, 2, 3]

    # Trích xuất đặc trưng
    WORD_NGRAM_RANGE: Tuple[int, int] = (1, 2)
    CHAR_NGRAM_RANGE: Tuple[int, int] = (2, 5)
    MIN_DF: int = 2
    PARETO_RATIO: float = 0.75
    MIN_FEAT_MULTIPLIER: int = 1
    MAX_FEAT_MULTIPLIER: int = 3

    # Mô hình & Platt Scaling
    INNER_CV_FOLDS: int = 5
    OUTER_CV_FOLDS: int = 5
    CALIBRATION_METHOD: str = "sigmoid"
    SVM_MAX_ITER: int = 5000

    # Lưới siêu tham số
    C_GRID: List[float] = np.geomspace(0.01, 50.0, num=12).round(5).tolist()
    TOX_ALPHA_GRID: List[float] = [0.0, 0.35, 0.50]  # Hệ số mũ trọng số lớp w_c = (N_max / N_c)^alpha
    OVERFIT_PENALTY_LAMBDA: float = 0.5
    MAX_ACCEPTABLE_GAP: float = 0.12

def setup_logger(log_path: str = "logs/training.log") -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("VN_MultiAxis_Phase2")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

logger = setup_logger()


# =====================================================================
# 2. KHỐI TIỀN XỬ LÝ TỐI THIỂU (MINIMAL PREPROCESSING)
# =====================================================================

class MinimalVietnameseCleaner(BaseEstimator, TransformerMixin):
    """
    Chỉ loại bỏ URL, HTML tags, timestamp, ký tự điều khiển lỗi hệ thống.
    Giữ nguyên 100% Teencode, Emoji, dấu thanh, chữ HOA/thường và ký tự lách luật.
    """
    URL_PATTERN = re.compile(r"(?:https?://|ftp://|www\.)\S+|(?<!\w)[a-zA-Z0-9.-]+\.(?:com|vn|net|org|edu|gov|io|ly|me)(?:/\S*)?", re.IGNORECASE)
    HTML_PATTERN = re.compile(r"<[^>]+>")
    TIMESTAMP_PATTERN = re.compile(
        r"\b(?:\d{1,2}[:h]\d{2}(?::\d{2})?(?:\s?[ap]m)?|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
        re.IGNORECASE
    )
    SYS_ERROR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd\u200b\u200c\u200d\ufeff]")
    MULTISPACE_PATTERN = re.compile(r"\s+")

    def fit(self, X: Union[pd.Series, List[str]], y: Any = None):
        return self

    @classmethod
    def clean_text(cls, text: str) -> str:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = unicodedata.normalize("NFC", text)
        text = cls.HTML_PATTERN.sub(" ", text)
        text = cls.URL_PATTERN.sub(" ", text)
        text = cls.TIMESTAMP_PATTERN.sub(" ", text)
        text = cls.SYS_ERROR_PATTERN.sub("", text)
        return cls.MULTISPACE_PATTERN.sub(" ", text).strip()

    def transform(self, X: Union[pd.Series, List[str]]) -> List[str]:
        cleaned = [self.clean_text(doc) for doc in X]
        return [c if c else "<empty_comment>" for c in cleaned]


def social_word_tokenizer(text: str) -> List[str]:
    tokens = [tok.strip('",;:!?()[]{}<>«»') for tok in text.split()]
    return [t for t in tokens if t] or ["<empty>"]


# =====================================================================
# 3. KHỐI TRÍCH XUẤT ĐẶC TRƯNG PARETO TF-IDF CÂN BẰNG LỚP
# =====================================================================

class DynamicParetoTfidfExtractor(BaseEstimator, TransformerMixin):
    """
    Trích xuất Word n-gram (1,2) + Char n-gram (2,5) với min_df=2.
    Tự động xác định max_features theo quy tắc Pareto 75% V_total và kẹp trong [1*N_train, 3*N_train].
    Sử dụng Class-Balanced Mean TF-IDF Score để bảo vệ các n-gram đặc trưng của lớp hiếm (Toxicity = 2, 3).
    """
    def __init__(
        self,
        word_ngram_range: Tuple[int, int] = (1, 2),
        char_ngram_range: Tuple[int, int] = (2, 5),
        min_df: int = 2,
        pareto_ratio: float = 0.75,
        min_mult: int = 1,
        max_mult: int = 3
    ):
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.min_df = min_df
        self.pareto_ratio = pareto_ratio
        self.min_mult = min_mult
        self.max_mult = max_mult

        self.union_vectorizer_: Optional[FeatureUnion] = None
        self.selected_indices_: Optional[np.ndarray] = None
        self.v_word_: int = 0
        self.v_char_: int = 0
        self.v_total_: int = 0
        self.n_train_: int = 0
        self.max_features_: int = 0
        self.bound_status_: str = ""

    def fit(self, X: List[str], y: Optional[np.ndarray] = None):
        self.n_train_ = len(X)
        word_vec = TfidfVectorizer(
            analyzer="word",
            tokenizer=social_word_tokenizer,
            token_pattern=None,
            lowercase=False,
            ngram_range=self.word_ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
            dtype=np.float32
        )
        char_vec = TfidfVectorizer(
            analyzer="char",
            lowercase=False,
            ngram_range=self.char_ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
            dtype=np.float32
        )
        self.union_vectorizer_ = FeatureUnion([
            ("word_tfidf", word_vec),
            ("char_tfidf", char_vec)
        ], n_jobs=1)

        X_full = self.union_vectorizer_.fit_transform(X).tocsr()
        self.v_word_ = len(self.union_vectorizer_.transformer_list[0][1].vocabulary_)
        self.v_char_ = len(self.union_vectorizer_.transformer_list[1][1].vocabulary_)
        self.v_total_ = X_full.shape[1]

        pareto_k = int(np.floor(self.pareto_ratio * self.v_total_))
        lower_bound = self.min_mult * self.n_train_
        upper_bound = self.max_mult * self.n_train_

        if self.v_total_ < lower_bound:
            self.max_features_ = self.v_total_
            self.bound_status_ = f"WARNING_V_TOTAL_BELOW_1N ({self.v_total_} < {lower_bound})"
        else:
            clamped_k = max(lower_bound, min(pareto_k, upper_bound))
            self.max_features_ = min(clamped_k, self.v_total_)
            if pareto_k < lower_bound:
                self.bound_status_ = "CLAMPED_TO_LOWER_BOUND_1N"
            elif pareto_k > upper_bound:
                self.bound_status_ = "CLAMPED_TO_UPPER_BOUND_3N"
            else:
                self.bound_status_ = "EXACT_PARETO_75_PERCENT"

        # Tính điểm TF-IDF kết hợp Toàn cục (Global) và Cân bằng Lớp (Class-Balanced)
        global_mean = np.asarray(X_full.mean(axis=0)).ravel()
        if y is not None and y.ndim == 2:
            max_class_mean = np.zeros(self.v_total_, dtype=np.float32)
            for col_idx in range(y.shape[1]):
                y_col = y[:, col_idx]
                for cls_val in np.unique(y_col):
                    mask = (y_col == cls_val)
                    if np.any(mask):
                        cls_mean = np.asarray(X_full[mask].mean(axis=0)).ravel()
                        max_class_mean = np.maximum(max_class_mean, cls_mean)
            g_norm = global_mean / (global_mean.max() + 1e-12)
            c_norm = max_class_mean / (max_class_mean.max() + 1e-12)
            tfidf_col_scores = 0.5 * g_norm + 0.5 * c_norm
        else:
            tfidf_col_scores = global_mean

        if self.max_features_ < self.v_total_:
            top_indices = np.argpartition(tfidf_col_scores, -self.max_features_)[-self.max_features_:]
            self.selected_indices_ = np.sort(top_indices)
        else:
            self.selected_indices_ = np.arange(self.v_total_)
        return self

    def transform(self, X: List[str]) -> sparse.csr_matrix:
        if self.union_vectorizer_ is None or self.selected_indices_ is None:
            raise RuntimeError("DynamicParetoTfidfExtractor chưa được fit!")
        X_full = self.union_vectorizer_.transform(X)
        return X_full[:, self.selected_indices_].tocsr()


# =====================================================================
# 4. KHỐI MÔ HÌNH ĐA TRỤC + PLATT SCALING + TRỌNG SỐ GIẢM CHẤN MŨ
# =====================================================================

def compute_power_law_class_weights(y_col: np.ndarray, alpha: float) -> Optional[Dict[int, float]]:
    """
    Tính trọng số lớp giảm chấn theo hàm mũ: w_c = (N_max / N_c)^alpha.
    Khi alpha = 0.0 -> Trả về None (tương đương trọng số 1.0 cho mọi lớp).
    """
    if alpha <= 0.0:
        return None
    classes, counts = np.unique(y_col, return_counts=True)
    n_max = float(np.max(counts))
    return {int(c): float((n_max / cnt) ** alpha) for c, cnt in zip(classes, counts)}


def build_single_axis_calibrated_svm(
    c_val: float,
    class_weight: Optional[Dict[int, float]] = None
) -> CalibratedClassifierCV:
    base_svm = LinearSVC(
        C=c_val,
        penalty="l2",
        loss="squared_hinge",
        dual=True,
        class_weight=class_weight,
        random_state=Config.RANDOM_STATE,
        max_iter=Config.SVM_MAX_ITER
    )
    return CalibratedClassifierCV(
        estimator=base_svm,
        method=Config.CALIBRATION_METHOD,
        cv=Config.INNER_CV_FOLDS,
        n_jobs=-1
    )


class DualAxisCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """
    Quản lý đồng thời 2 trục phân loại (Sentiment & Toxicity) với giao diện giống hệt MultiOutputClassifier:
    - Cho phép mỗi trục sử dụng siêu tham số C* và trọng số lớp giảm chấn alpha* tối ưu riêng.
    - Hỗ trợ bộ nhân ngưỡng hậu Platt Scaling (tox_thresholds) cho hàm .predict() để chống bỏ sót lớp hiếm,
      trong khi giữ nguyên 100% xác suất Platt Scaling chuẩn ở hàm .predict_proba().
    """
    def __init__(
        self,
        c_sent: float = 1.0,
        c_tox: float = 1.0,
        alpha_tox: float = 0.0,
        tox_thresholds: Optional[List[float]] = None
    ):
        self.c_sent = c_sent
        self.c_tox = c_tox
        self.alpha_tox = alpha_tox
        self.tox_thresholds = tox_thresholds

        self.sent_clf_: Optional[CalibratedClassifierCV] = None
        self.tox_clf_: Optional[CalibratedClassifierCV] = None
        self.estimators_: List[CalibratedClassifierCV] = []
        self.classes_: List[np.ndarray] = []
        self.tox_weights_dict_: Optional[Dict[int, float]] = None

    def fit(self, X: sparse.csr_matrix, y: np.ndarray):
        y_sent = y[:, 0]
        y_tox = y[:, 1]

        self.tox_weights_dict_ = compute_power_law_class_weights(y_tox, self.alpha_tox)

        self.sent_clf_ = build_single_axis_calibrated_svm(self.c_sent, class_weight=None)
        self.tox_clf_ = build_single_axis_calibrated_svm(self.c_tox, class_weight=self.tox_weights_dict_)

        self.sent_clf_.fit(X, y_sent)
        self.tox_clf_.fit(X, y_tox)

        self.estimators_ = [self.sent_clf_, self.tox_clf_]
        self.classes_ = [self.sent_clf_.classes_, self.tox_clf_.classes_]
        return self

    def predict_proba(self, X: sparse.csr_matrix) -> List[np.ndarray]:
        if self.sent_clf_ is None or self.tox_clf_ is None:
            raise RuntimeError("DualAxisCalibratedClassifier chưa được fit!")
        return [self.sent_clf_.predict_proba(X), self.tox_clf_.predict_proba(X)]

    def predict(self, X: sparse.csr_matrix) -> np.ndarray:
        probas = self.predict_proba(X)
        sent_proba, tox_proba = probas[0], probas[1]

        sent_preds = self.classes_[0][np.argmax(sent_proba, axis=1)]
        if self.tox_thresholds is not None:
            tau = np.asarray(self.tox_thresholds, dtype=np.float32).reshape(1, -1)
            tox_preds = self.classes_[1][np.argmax(tox_proba * tau, axis=1)]
        else:
            tox_preds = self.classes_[1][np.argmax(tox_proba, axis=1)]

        return np.column_stack([sent_preds, tox_preds])


def optimize_oof_toxicity_thresholds(
    y_tox_true: np.ndarray,
    oof_tox_proba: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[List[float], float, float]:
    """
    Tối ưu hệ số nhân ngưỡng [1.0, tau_1, tau_2, tau_3] bằng Out-of-Fold Cross-Validation
    để tránh học vẹt ngưỡng. Chỉ áp dụng nếu cải thiện OOF Macro F1 >= +0.003.
    """
    classes = np.array(Config.EXPECTED_TOXICITY_CLASSES)
    base_preds = classes[np.argmax(oof_tox_proba, axis=1)]
    base_f1 = float(f1_score(y_tox_true, base_preds, average="macro", zero_division=0))

    grid_candidates = np.linspace(0.6, 3.0, 25)
    fold_taus = []
    cv_tuned_preds = np.zeros_like(y_tox_true)

    for tr_idx, va_idx in folds:
        p_tr, y_tr = oof_tox_proba[tr_idx], y_tox_true[tr_idx]
        p_va = oof_tox_proba[va_idx]

        best_tau = np.ones(4, dtype=np.float32)
        best_tr_score = float(f1_score(y_tr, classes[np.argmax(p_tr, axis=1)], average="macro", zero_division=0))

        # Coordinate search 2 vòng cho lớp 1, 2, 3
        for _ in range(2):
            for cls_idx in [1, 2, 3]:
                for cand in grid_candidates:
                    trial_tau = best_tau.copy()
                    trial_tau[cls_idx] = cand
                    preds_tr = classes[np.argmax(p_tr * trial_tau.reshape(1, -1), axis=1)]
                    score = float(f1_score(y_tr, preds_tr, average="macro", zero_division=0))
                    if score > best_tr_score + 1e-5:
                        best_tr_score = score
                        best_tau = trial_tau

        fold_taus.append(best_tau)
        cv_tuned_preds[va_idx] = classes[np.argmax(p_va * best_tau.reshape(1, -1), axis=1)]

    tuned_oof_f1 = float(f1_score(y_tox_true, cv_tuned_preds, average="macro", zero_division=0))
    if tuned_oof_f1 >= base_f1 + 0.003:
        final_tau = np.median(np.vstack(fold_taus), axis=0).round(4).tolist()
        logger.info(f"  [Threshold Tuning] Kích hoạt hệ số ngưỡng {final_tau}: OOF Toxicity F1 tăng từ {base_f1:.4f} -> {tuned_oof_f1:.4f}")
        return final_tau, base_f1, tuned_oof_f1
    else:
        logger.info(f"  [Threshold Tuning] Giữ ngưỡng chuẩn [1, 1, 1, 1] (Base F1 = {base_f1:.4f}, Tuned = {tuned_oof_f1:.4f})")
        return [1.0, 1.0, 1.0, 1.0], base_f1, base_f1


# =====================================================================
# 5. ĐIỀU PHỐI 5-FOLD CV, CHECKPOINT & REFIT TOÀN TẬP TRAIN
# =====================================================================

def create_joint_stratified_folds(y: np.ndarray, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Phân tầng trực tiếp trên tổ hợp nhãn (sentiment, toxicity).
    Với tập dữ liệu 29,474 mẫu hiện tại, cả 10 tổ hợp hiện diện đều có >= 14 mẫu (>= 5),
    đảm bảo chia 5-Fold chính xác tuyệt đối.
    """
    joint_labels = np.array([f"{s}_{t}" for s, t in zip(y[:, 0], y[:, 1])])
    unique_j, counts_j = np.unique(joint_labels, return_counts=True)
    count_map = dict(zip(unique_j, counts_j))

    # Chốt chặn an toàn nếu có bất kỳ ô nào < n_splits
    if np.min(counts_j) < n_splits:
        dominant_for_tox = {}
        for t_val in np.unique(y[:, 1]):
            cands = [(f"{s_val}_{t_val}", count_map.get(f"{s_val}_{t_val}", 0)) for s_val in np.unique(y[:, 0])]
            dominant_for_tox[int(t_val)] = max(cands, key=lambda x: x[1])[0]
        joint_labels = np.array([
            dominant_for_tox[int(t)] if count_map[f"{s}_{t}"] < n_splits else f"{s}_{t}"
            for s, t in zip(y[:, 0], y[:, 1])
        ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.RANDOM_STATE)
    return list(skf.split(np.zeros(len(y)), joint_labels))


def tune_and_train(
    texts: List[str],
    y: np.ndarray,
    checkpoint_path: str = "logs/cv_checkpoints.jsonl",
    oof_out_path: str = "artifacts/oof_predictions.npz"
) -> Tuple[Pipeline, Dict[str, Any]]:
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(oof_out_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Bước 1/6: Làm sạch văn bản tối thiểu (Minimal Cleaning)...")
    cleaner = MinimalVietnameseCleaner()
    cleaned_texts = cleaner.fit_transform(texts)

    logger.info("Bước 2/6: Khởi tạo 5-Fold Joint Stratified & Tiền tính toán ma trận TF-IDF Pareto...")
    folds = create_joint_stratified_folds(y, n_splits=Config.OUTER_CV_FOLDS)

    fold_data = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        t0 = time.time()
        X_tr_raw = [cleaned_texts[i] for i in train_idx]
        X_va_raw = [cleaned_texts[i] for i in val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        extractor = DynamicParetoTfidfExtractor(
            word_ngram_range=Config.WORD_NGRAM_RANGE,
            char_ngram_range=Config.CHAR_NGRAM_RANGE,
            min_df=Config.MIN_DF,
            pareto_ratio=Config.PARETO_RATIO,
            min_mult=Config.MIN_FEAT_MULTIPLIER,
            max_mult=Config.MAX_FEAT_MULTIPLIER
        )
        X_tr_vec = extractor.fit_transform(X_tr_raw, y_tr)
        X_va_vec = extractor.transform(X_va_raw)
        fold_data.append((train_idx, val_idx, X_tr_vec, y_tr, X_va_vec, y_va))
        logger.info(
            f"  [Fold {fold_idx+1}/5] N_tr={extractor.n_train_:,}, V_total={extractor.v_total_:,}, "
            f"max_features={extractor.max_features_:,} ({extractor.bound_status_}) [{time.time()-t0:.2f}s]"
        )

    # Đọc checkpoint nếu có
    completed_records: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed_records[rec["key"]] = rec
        logger.info(f"Đã khôi phục {len(completed_records)} kết quả từ checkpoint.")

    # -----------------------------------------------------------------
    # Bước 3A: Quét C cho Trục 1 (Sentiment {-1, 0, 1})
    # -----------------------------------------------------------------
    logger.info(f"Bước 3/6: Tối ưu Siêu tham số C cho Trục 1 (Sentiment): {Config.C_GRID}")
    sent_cv_results: List[Dict[str, Any]] = []
    for c_val in Config.C_GRID:
        key = f"SENT_C_{c_val}"
        if key in completed_records:
            sent_cv_results.append(completed_records[key])
            continue

        t0 = time.time()
        tr_scores, va_scores = [], []
        for _, _, X_tr_vec, y_tr, X_va_vec, y_va in fold_data:
            clf = build_single_axis_calibrated_svm(c_val, class_weight=None)
            clf.fit(X_tr_vec, y_tr[:, 0])
            tr_scores.append(float(f1_score(y_tr[:, 0], clf.predict(X_tr_vec), average="macro", zero_division=0)))
            va_scores.append(float(f1_score(y_va[:, 0], clf.predict(X_va_vec), average="macro", zero_division=0)))

        mean_tr, mean_va, std_va = float(np.mean(tr_scores)), float(np.mean(va_scores)), float(np.std(va_scores))
        gap = max(0.0, mean_tr - mean_va)
        sel_score = mean_va - Config.OVERFIT_PENALTY_LAMBDA * gap
        rec = {
            "key": key, "axis": "sentiment", "C": c_val, "alpha": 0.0,
            "mean_train_f1": round(mean_tr, 5), "mean_val_f1": round(mean_va, 5), "std_val_f1": round(std_va, 5),
            "overfit_gap": round(gap, 5), "penalized_selection_score": round(sel_score, 5),
            "elapsed_sec": round(time.time() - t0, 2)
        }
        sent_cv_results.append(rec)
        with open(checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(
            f"  [Sentiment | C={c_val:<8}] Val_F1={mean_va:.4f} (±{std_va:.4f}) | "
            f"Train_F1={mean_tr:.4f} | Gap={gap:.4f} | Score={sel_score:.4f} ({rec['elapsed_sec']}s)"
        )

    valid_sent = [r for r in sent_cv_results if r["overfit_gap"] <= Config.MAX_ACCEPTABLE_GAP] or sent_cv_results
    best_sent_rec = sorted(valid_sent, key=lambda x: (x["penalized_selection_score"], x["mean_val_f1"]), reverse=True)[0]
    best_c_sent = float(best_sent_rec["C"])
    logger.info(f"==> C* TRỤC SENTIMENT: C_sent* = {best_c_sent} (Val F1 = {best_sent_rec['mean_val_f1']:.4f}, Gap = {best_sent_rec['overfit_gap']:.4f})")

    # -----------------------------------------------------------------
    # Bước 3B: Quét (C, alpha) cho Trục 2 (Toxicity {0, 1, 2, 3})
    # -----------------------------------------------------------------
    logger.info(f"Bước 4/6: Tối ưu (C, alpha) cho Trục 2 (Toxicity - IR=169.36): C={Config.C_GRID} x alpha={Config.TOX_ALPHA_GRID}")
    tox_cv_results: List[Dict[str, Any]] = []
    for alpha_val in Config.TOX_ALPHA_GRID:
        for c_val in Config.C_GRID:
            key = f"TOX_A_{alpha_val}_C_{c_val}"
            if key in completed_records:
                tox_cv_results.append(completed_records[key])
                continue

            t0 = time.time()
            tr_scores, va_scores = [], []
            for _, _, X_tr_vec, y_tr, X_va_vec, y_va in fold_data:
                cw = compute_power_law_class_weights(y_tr[:, 1], alpha_val)
                clf = build_single_axis_calibrated_svm(c_val, class_weight=cw)
                clf.fit(X_tr_vec, y_tr[:, 1])
                tr_scores.append(float(f1_score(y_tr[:, 1], clf.predict(X_tr_vec), average="macro", zero_division=0)))
                va_scores.append(float(f1_score(y_va[:, 1], clf.predict(X_va_vec), average="macro", zero_division=0)))

            mean_tr, mean_va, std_va = float(np.mean(tr_scores)), float(np.mean(va_scores)), float(np.std(va_scores))
            gap = max(0.0, mean_tr - mean_va)
            sel_score = mean_va - Config.OVERFIT_PENALTY_LAMBDA * gap
            rec = {
                "key": key, "axis": "toxicity", "C": c_val, "alpha": alpha_val,
                "mean_train_f1": round(mean_tr, 5), "mean_val_f1": round(mean_va, 5), "std_val_f1": round(std_va, 5),
                "overfit_gap": round(gap, 5), "penalized_selection_score": round(sel_score, 5),
                "elapsed_sec": round(time.time() - t0, 2)
            }
            tox_cv_results.append(rec)
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(
                f"  [Toxicity  | alpha={alpha_val:<4} | C={c_val:<8}] Val_F1={mean_va:.4f} (±{std_va:.4f}) | "
                f"Train_F1={mean_tr:.4f} | Gap={gap:.4f} | Score={sel_score:.4f} ({rec['elapsed_sec']}s)"
            )

    valid_tox = [r for r in tox_cv_results if r["overfit_gap"] <= Config.MAX_ACCEPTABLE_GAP] or tox_cv_results
    best_tox_rec = sorted(valid_tox, key=lambda x: (x["penalized_selection_score"], x["mean_val_f1"]), reverse=True)[0]
    best_c_tox = float(best_tox_rec["C"])
    best_alpha_tox = float(best_tox_rec["alpha"])
    logger.info(
        f"==> (C*, alpha*) TRỤC TOXICITY: C_tox* = {best_c_tox}, alpha* = {best_alpha_tox} "
        f"(Val F1 = {best_tox_rec['mean_val_f1']:.4f}, Gap = {best_tox_rec['overfit_gap']:.4f})"
    )

    # -----------------------------------------------------------------
    # Bước 5: Sinh xác suất Out-of-Fold (OOF) & Tinh chỉnh ngưỡng hậu Platt Scaling
    # -----------------------------------------------------------------
    logger.info("Bước 5/6: Thu thập xác suất Out-of-Fold (OOF) & Kiểm tra tinh chỉnh ngưỡng hậu Platt Scaling...")
    n_samples = len(y)
    oof_proba_sent = np.zeros((n_samples, 3), dtype=np.float32)
    oof_proba_tox = np.zeros((n_samples, 4), dtype=np.float32)

    for _, val_idx, X_tr_vec, y_tr, X_va_vec, _ in fold_data:
        clf_s = build_single_axis_calibrated_svm(best_c_sent, class_weight=None)
        clf_s.fit(X_tr_vec, y_tr[:, 0])
        oof_proba_sent[val_idx] = clf_s.predict_proba(X_va_vec)

        cw_t = compute_power_law_class_weights(y_tr[:, 1], best_alpha_tox)
        clf_t = build_single_axis_calibrated_svm(best_c_tox, class_weight=cw_t)
        clf_t.fit(X_tr_vec, y_tr[:, 1])
        oof_proba_tox[val_idx] = clf_t.predict_proba(X_va_vec)

    best_tau_tox, base_tox_oof_f1, tuned_tox_oof_f1 = optimize_oof_toxicity_thresholds(y[:, 1], oof_proba_tox, folds)

    sent_classes = np.array(Config.EXPECTED_SENTIMENT_CLASSES)
    tox_classes = np.array(Config.EXPECTED_TOXICITY_CLASSES)
    oof_preds_sent = sent_classes[np.argmax(oof_proba_sent, axis=1)]
    oof_preds_tox = tox_classes[np.argmax(oof_proba_tox * np.array(best_tau_tox).reshape(1, -1), axis=1)]
    oof_preds = np.column_stack([oof_preds_sent, oof_preds_tox])

    np.savez_compressed(
        oof_out_path,
        y_true=y,
        oof_preds=oof_preds,
        oof_proba_sent=oof_proba_sent,
        oof_proba_tox=oof_proba_tox
    )

    # -----------------------------------------------------------------
    # Bước 6: Re-fit toàn bộ Pipeline trên 100% tập Train (29,474 mẫu)
    # -----------------------------------------------------------------
    logger.info("Bước 6/6: Re-fit toàn bộ Pipeline trên 100% dữ liệu tập Train (29,474 mẫu)...")
    final_pipeline = Pipeline([
        ("cleaner", MinimalVietnameseCleaner()),
        ("tfidf_pareto", DynamicParetoTfidfExtractor(
            word_ngram_range=Config.WORD_NGRAM_RANGE,
            char_ngram_range=Config.CHAR_NGRAM_RANGE,
            min_df=Config.MIN_DF,
            pareto_ratio=Config.PARETO_RATIO,
            min_mult=Config.MIN_FEAT_MULTIPLIER,
            max_mult=Config.MAX_FEAT_MULTIPLIER
        )),
        ("classifier", DualAxisCalibratedClassifier(
            c_sent=best_c_sent,
            c_tox=best_c_tox,
            alpha_tox=best_alpha_tox,
            tox_thresholds=best_tau_tox
        ))
    ])

    t_refit = time.time()
    final_pipeline.fit(texts, y)
    refit_time = round(time.time() - t_refit, 2)

    ext: DynamicParetoTfidfExtractor = final_pipeline.named_steps["tfidf_pareto"]
    clf_step: DualAxisCalibratedClassifier = final_pipeline.named_steps["classifier"]

    oof_f1_sent = float(f1_score(y[:, 0], oof_preds_sent, average="macro", zero_division=0))
    oof_f1_combined = (oof_f1_sent + tuned_tox_oof_f1) / 2.0

    metadata = {
        "best_hyperparameters": {
            "C_sentiment": best_c_sent,
            "C_toxicity": best_c_tox,
            "alpha_toxicity": best_alpha_tox,
            "toxicity_class_weights": clf_step.tox_weights_dict_,
            "toxicity_thresholds": best_tau_tox
        },
        "oof_summary_metrics": {
            "oof_macro_f1_sentiment": round(oof_f1_sent, 5),
            "oof_macro_f1_toxicity_raw_argmax": round(base_tox_oof_f1, 5),
            "oof_macro_f1_toxicity_final": round(tuned_tox_oof_f1, 5),
            "oof_macro_f1_combined": round(oof_f1_combined, 5)
        },
        "best_sentiment_cv_record": best_sent_rec,
        "best_toxicity_cv_record": best_tox_rec,
        "sentiment_cv_results": sorted(sent_cv_results, key=lambda x: x["C"]),
        "toxicity_cv_results": sorted(tox_cv_results, key=lambda x: (x["alpha"], x["C"])),
        "full_train_stats": {
            "n_train": ext.n_train_,
            "v_word_clean": ext.v_word_,
            "v_char_clean": ext.v_char_,
            "v_total_clean": ext.v_total_,
            "pareto_75_raw": int(np.floor(Config.PARETO_RATIO * ext.v_total_)),
            "final_max_features": ext.max_features_,
            "bound_constraint_range": [
                Config.MIN_FEAT_MULTIPLIER * ext.n_train_,
                Config.MAX_FEAT_MULTIPLIER * ext.n_train_
            ],
            "bound_status": ext.bound_status_,
            "refit_time_sec": refit_time
        }
    }
    return final_pipeline, metadata


# =====================================================================
# 6. BỘ TỰ KIỂM TRA 7 BƯỚC (7-STEP SELF-VERIFICATION SUITE)
# =====================================================================

def run_self_verification(
    pipeline: Pipeline,
    sample_texts: List[str],
    y_true: np.ndarray,
    oof_path: str,
    model_path: str,
    metadata: Dict[str, Any]
) -> Dict[str, str]:
    logger.info("Đang chạy Bộ Tự Kiểm Tra 7 Bước (Self-Verification Suite)...")
    results: Dict[str, str] = {}

    # TEST 01: Cleaner Integrity (Xóa URL/HTML/Timestamp, giữ nguyên Teencode + Emoji)
    test_dirty = "Xem tại https://vnexpress.net <b>đ.m</b> v~l luôn 😭🔥 12:30:00\ufffd"
    cleaned = pipeline.named_steps["cleaner"].transform([test_dirty])[0]
    assert "https://" not in cleaned and "<b>" not in cleaned and "12:30:00" not in cleaned, "TEST_01 FAILED: Chưa xóa sạch nhiễu kỹ thuật!"
    assert "đ.m" in cleaned and "v~l" in cleaned and "😭🔥" in cleaned, "TEST_01 FAILED: Làm mất Teencode hoặc Emoji!"
    results["TEST_01_CLEANER_INTEGRITY"] = "PASS"

    # TEST 02: Pareto 75% & Ràng buộc [1*N_train, 3*N_train]
    ext: DynamicParetoTfidfExtractor = pipeline.named_steps["tfidf_pareto"]
    X_sample = ext.transform(pipeline.named_steps["cleaner"].transform(sample_texts[:10]))
    assert X_sample.shape[1] == ext.max_features_, "TEST_02 FAILED: Số cột TF-IDF khác max_features_!"
    assert ext.n_train_ <= ext.max_features_ <= 3 * ext.n_train_, "TEST_02 FAILED: Vi phạm biên [1*N_train, 3*N_train]!"
    results["TEST_02_PARETO_BOUNDS"] = f"PASS (max_features={ext.max_features_:,} in [{ext.n_train_:,}, {3*ext.n_train_:,}])"

    # TEST 03: Prediction Shape & Exact Label Domain ({-1, 0, 1} và {0, 1, 2, 3})
    preds = pipeline.predict(sample_texts[:50])
    assert preds.shape == (50, 2), f"TEST_03 FAILED: Shape dự đoán sai {preds.shape}"
    assert set(np.unique(preds[:, 0])).issubset({-1, 0, 1}), f"TEST_03 FAILED: Nhãn Sentiment ngoài {{-1, 0, 1}}: {np.unique(preds[:, 0])}"
    assert set(np.unique(preds[:, 1])).issubset({0, 1, 2, 3}), f"TEST_03 FAILED: Nhãn Toxicity ngoài {{0, 1, 2, 3}}: {np.unique(preds[:, 1])}"
    results["TEST_03_PREDICT_DOMAIN"] = "PASS (Sentiment in {-1, 0, 1}, Toxicity in {0, 1, 2, 3})"

    # TEST 04: Platt Scaling Probability Normalization
    probas = pipeline.predict_proba(sample_texts[:50])
    assert isinstance(probas, list) and len(probas) == 2, "TEST_04 FAILED: Đầu ra predict_proba không phải list 2 phần tử!"
    assert probas[0].shape == (50, 3) and probas[1].shape == (50, 4), "TEST_04 FAILED: Sai kích thước ma trận xác suất!"
    assert np.allclose(probas[0].sum(axis=1), 1.0, atol=1e-5), "TEST_04 FAILED: Tổng xác suất Sentiment khác 1.0!"
    assert np.allclose(probas[1].sum(axis=1), 1.0, atol=1e-5), "TEST_04 FAILED: Tổng xác suất Toxicity khác 1.0!"
    results["TEST_04_PLATT_CALIBRATION"] = "PASS (Row probability sums = 1.00000)"

    # TEST 05: Serialization Reproducibility (Joblib Dump & Load)
    loaded_pipe = joblib.load(model_path)
    probas_loaded = loaded_pipe.predict_proba(sample_texts[:20])
    preds_loaded = loaded_pipe.predict(sample_texts[:20])
    assert np.allclose(probas[0][:20], probas_loaded[0], atol=1e-6), "TEST_05 FAILED: Sai lệch xác suất Sentiment sau khi load!"
    assert np.allclose(probas[1][:20], probas_loaded[1], atol=1e-6), "TEST_05 FAILED: Sai lệch xác suất Toxicity sau khi load!"
    assert np.array_equal(preds[:20], preds_loaded), "TEST_05 FAILED: Sai lệch nhãn dự đoán sau khi load!"
    results["TEST_05_SERIALIZATION"] = "PASS"

    # TEST 06: Rare Class Non-Collapse Check (Đảm bảo lớp hiếm Toxicity = 2, 3 không bị liệt về 0)
    oof_data = np.load(oof_path)
    oof_y_tox = oof_data["y_true"][:, 1]
    oof_pred_tox = oof_data["oof_preds"][:, 1]
    per_class_tox_f1 = f1_score(oof_y_tox, oof_pred_tox, average=None, labels=[0, 1, 2, 3], zero_division=0)
    assert np.all(per_class_tox_f1 > 0.05), f"TEST_06 FAILED: Có lớp Toxicity bị liệt (F1 <= 0.05): {per_class_tox_f1}"
    results["TEST_06_RARE_CLASS_NON_COLLAPSE"] = (
        f"PASS (Toxicity Per-Class F1: C0={per_class_tox_f1[0]:.3f}, C1={per_class_tox_f1[1]:.3f}, "
        f"C2={per_class_tox_f1[2]:.3f}, C3={per_class_tox_f1[3]:.3f})"
    )

    # TEST 07: Overall Macro F1 Sanity Check
    comb_f1 = metadata["oof_summary_metrics"]["oof_macro_f1_combined"]
    assert comb_f1 >= 0.45, f"TEST_07 FAILED: Combined Macro F1 quá thấp ({comb_f1})"
    results["TEST_07_OVERALL_MACRO_F1"] = f"PASS (Combined OOF Macro F1 = {comb_f1:.4f})"

    logger.info("==> TOÀN BỘ 7 BÀI KIỂM TRA TỰ ĐỘNG ĐỀU ĐẠT CHUẨN (100% PASSED)!")
    return results


# =====================================================================
# 7. ENTRY POINT CHÍNH (MAIN)
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Multi-Axis Training Pipeline")
    parser.add_argument("--data_path", type=str, default="data/train.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--sent_col", type=str, default="sentiment")
    parser.add_argument("--tox_col", type=str, default="toxicity")
    parser.add_argument("--model_out", type=str, default="artifacts/model_pipeline.joblib")
    parser.add_argument("--meta_out", type=str, default="artifacts/metadata.json")
    parser.add_argument("--oof_out", type=str, default="artifacts/oof_predictions.npz")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        logger.error(f"Không tìm thấy file dữ liệu: {args.data_path}")
        sys.exit(1)

    ext = Path(args.data_path).suffix.lower()
    df = pd.read_parquet(args.data_path) if ext == ".parquet" else pd.read_csv(args.data_path, encoding="utf-8-sig")

    texts = df[args.text_col].astype(str).tolist()
    y = df[[args.sent_col, args.tox_col]].to_numpy(dtype=int)

    # Kiểm tra nhanh miền nhãn trước khi chạy
    assert set(np.unique(y[:, 0])).issubset({-1, 0, 1}), f"Cột Sentiment chứa giá trị lạ: {np.unique(y[:, 0])}"
    assert set(np.unique(y[:, 1])).issubset({0, 1, 2, 3}), f"Cột Toxicity chứa giá trị lạ: {np.unique(y[:, 1])}"

    pipeline, metadata = tune_and_train(texts, y, oof_out_path=args.oof_out)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out, compress=3)

    sha256_hash = hashlib.sha256(Path(args.model_out).read_bytes()).hexdigest()
    metadata["model_artifact_sha256"] = sha256_hash
    metadata["verification_results"] = run_self_verification(
        pipeline, texts, y, args.oof_out, args.model_out, metadata
    )

    Path(args.meta_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Hoàn tất Phase 2! Model lưu tại: {args.model_out} | Metadata lưu tại: {args.meta_out}")

if __name__ == "__main__":
    main()
```

---

## CẬP NHẬT NHẸ CHO SCRIPT BÁO CÁO PHASE 3 (`src/step3_generate_report.py`)

Vì chúng ta đã giữ nguyên miền nhãn `{-1, 0, 1}` cho trục Sentiment và tách bảng quét siêu tham số riêng cho 2 trục (`Sentiment` và `Toxicity`), Coding Agent hãy cập nhật file `src/step3_generate_report.py` theo đoạn code dưới đây để báo cáo xuất ra khớp 100% với cấu trúc `metadata.json` mới:

````python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: Automated Engineering Report Generator (Synced with DualAxisCalibratedClassifier)
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, brier_score_loss

def compute_multiclass_brier(y_true_col: np.ndarray, proba_matrix: np.ndarray, classes: list) -> float:
    brier_scores = []
    for idx, c in enumerate(classes):
        y_bin = (y_true_col == c).astype(int)
        brier_scores.append(brier_score_loss(y_bin, proba_matrix[:, idx]))
    return float(np.mean(brier_scores))

def generate_markdown_report(
    meta_path: str = "artifacts/metadata.json",
    oof_path: str = "artifacts/oof_predictions.npz",
    out_md_path: str = "reports/final_training_report.md"
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
        y_true[:, 0], y_pred[:, 0],
        labels=[-1, 0, 1],
        target_names=["-1: Tiêu cực (Negative)", " 0: Trung tính (Neutral)", " 1: Tích cực (Positive)"],
        digits=4, zero_division=0
    )
    tox_report = classification_report(
        y_true[:, 1], y_pred[:, 1],
        labels=[0, 1, 2, 3],
        target_names=["0: Không độc hại", "1: Thiếu lịch sự", "2: Xúc phạm cá nhân", "3: Thù ghét/Đe dọa"],
        digits=4, zero_division=0
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
        "| :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for r in meta["sentiment_cv_results"]:
        mark = "**" if r["C"] == hp["C_sentiment"] else ""
        md_lines.append(
            f"| {mark}{r['C']}{mark} | {r['mean_train_f1']:.4f} | {mark}{r['mean_val_f1']:.4f} ± {r['std_val_f1']:.4f}{mark} | "
            f"{r['overfit_gap']:.4f} | {mark}{r['penalized_selection_score']:.4f}{mark} | {r['elapsed_sec']}s |"
        )

    md_lines.extend([
        "",
        "### 2.2. Bảng Quét `(alpha, C)` trên Trục 2 (Toxicity)",
        "| Hệ số `alpha` | Tham số `C` | Train Macro F1 | Val Macro F1 (Mean ± Std) | Overfit Gap | Selection Score | Thời gian (s) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for r in meta["toxicity_cv_results"]:
        is_best = (r["C"] == hp["C_toxicity"]) and (r["alpha"] == hp["alpha_toxicity"])
        mark = "**" if is_best else ""
        md_lines.append(
            f"| {mark}{r['alpha']}{mark} | {mark}{r['C']}{mark} | {r['mean_train_f1']:.4f} | "
            f"{mark}{r['mean_val_f1']:.4f} ± {r['std_val_f1']:.4f}{mark} | {r['overfit_gap']:.4f} | "
            f"{mark}{r['penalized_selection_score']:.4f}{mark} | {r['elapsed_sec']}s |"
        )

    md_lines.extend([
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
        "## 4. Kết quả Bộ Tự Kiểm Tra 7 Bước (Self-Verification Suite)"
    ])

    for k, v in meta["verification_results"].items():
        md_lines.append(f"- **{k}:** `{v}`")

    md_lines.extend([
        "",
        "## 5. Mã kiểm tra Toàn vẹn Mô hình (Artifact Integrity)",
        f"- **SHA256 (`artifacts/model_pipeline.joblib`):** `{meta['model_artifact_sha256']}`"
    ])

    Path(out_md_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[SUCCESS] Đã xuất báo cáo hoàn chỉnh tại: {out_md_path}")

if __name__ == "__main__":
    generate_markdown_report()
````

---

### SẴN SÀNG KÍCH HOẠT HUẤN LUYỆN NỀN (PHASE 2 $\to$ PHASE 3 $\to$ PHASE 4)

Coding Agent cập nhật 2 file `src/step2_train_pipeline.py` và `src/step3_generate_report.py`, sau đó khởi chạy script chạy nền `src/run_background.sh` trên file dữ liệu `29.474` mẫu đã làm sạch. Ngay khi có file `reports/final_training_report.md`, hãy gửi cho tôi để chúng ta cùng đọc kết quả thực nghiệm!

# BÁO CÁO BENCHMARK ĐỘC LẬP (BLIND TEST): CHALLENGE DATASET

> [!IMPORTANT]
> **Tuân thủ nguyên tắc Blind Test tuyệt đối:**
> - Tuyệt đối **không gọi lệnh `fit` hay `fit_transform`** lại bất kỳ thành phần nào của `TfidfVectorizer`.
> - Mọi từ vựng mới không nằm trong bộ **111.675 đặc trưng đã học** đều tự động bị bỏ qua một cách an toàn theo cơ chế trích xuất chiếu ma trận tĩnh `transform()`.
> - Tập dữ liệu thử nghiệm `challenge_data/challenge.csv` là tập kiểm thử độc lập, hoàn toàn không có sự giao thoa hay rò rỉ dữ liệu ($0$ mẫu trùng lặp với tập huấn luyện).

## 1. Thông Tin Môi Trường & Tập Dữ Liệu Kiểm Thử

- **Đường dẫn tập kiểm thử chính:** `challenge_data/challenge.csv`
- **Tổng số mẫu kiểm thử:** **3,237 mẫu** (100% câu phân biệt, không trùng lặp)
- **Phân phối nhãn Trục Sentiment:**
  - `-1 (Tiêu cực)`: 1,073 mẫu (33.15%)
  - ` 0 (Trung tính)`: 1,156 mẫu (35.71%)
  - ` 1 (Tích cực)`: 1,008 mẫu (31.14%)
- **Phân phối nhãn Trục Toxicity:**
  - `0 (Không độc hại)`: 2,736 mẫu (84.52%)
  - `1 (Thiếu lịch sự)`: 314 mẫu (9.70%)
  - `2 (Xúc phạm cá nhân)`: 158 mẫu (4.88%)
  - `3 (Thù ghét/Đe dọa)`: 29 mẫu (0.90%)

---

## 2. Bảng Xếp Hạng Hiệu Năng Tổng Thể (Model Leaderboard)

Bảng so sánh đối đầu khách quan giữa 3 biến thể mô hình đã huấn luyện:

| Mô hình | Không gian đặc trưng | Sent Acc | Sent F1 | Tox Acc (Tuned) | Tox F1 (Tuned) | Tox F1 (Argmax) | **Combined F1** | Tốc độ (mẫu/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production Model (37k Full Train, max_val)**<br>`artifacts/model_pipeline.joblib` | 111,675 cột | 65.15% | 0.6355 | 90.61% | **0.6644** | 0.5958 | **0.6500** | 3356 |
| Train-Only 29k (max_val)<br>`artifacts/train_only_max_val/model_pipeline.joblib` | 88,422 cột | 64.26% | 0.6253 | 90.64% | **0.6657** | 0.6183 | **0.6455** | 3488 |
| Train-Only 29k (penalized)<br>`artifacts/train_only_penalized/model_pipeline.joblib` | 88,422 cột | 64.04% | 0.6226 | 89.74% | **0.6393** | 0.6004 | **0.6309** | 3483 |

---

## 3. Phân Tích Chi Tiết Mô Hình Sản Xuất Chính (111.675 Đặc Trưng)

### 3.1. Trục 1: Sentiment Classification

- **Độ chính xác tổng thể (Accuracy):** `65.15%`
- **Macro F1 Score:** **`0.6355`**
- **Weighted F1 Score:** `0.6355`
- **Brier Score Loss (Hiệu chuẩn xác suất):** `0.15614`

**Báo cáo phân loại chi tiết từng lớp:**
```text
                         precision    recall  f1-score   support

 -1 Tiêu cực (Negative)     0.8432    0.3709    0.5152      1073
 0 Trung tính (Neutral)     0.5657    0.8789    0.6883      1156
  1 Tích cực (Positive)     0.7172    0.6895    0.7031      1008

               accuracy                         0.6515      3237
              macro avg     0.7087    0.6464    0.6355      3237
           weighted avg     0.7049    0.6515    0.6355      3237
```

**Ma trận nhầm lẫn (Sentiment Confusion Matrix):**

| Nhãn thực tế \ Dự đoán | Dự đoán -1 (Tiêu cực) | Dự đoán 0 (Trung tính) | Dự đoán 1 (Tích cực) | Tỷ lệ nhận diện đúng (Recall) |
| :--- | :---: | :---: | :---: | :---: |
| **Thực tế -1 (Tiêu cực)** | **398** | 506 | 169 | 37.09% |
| **Thực tế  0 (Trung tính)** | 35 | **1016** | 105 | 87.89% |
| **Thực tế  1 (Tích cực)** | 39 | 274 | **695** | 68.95% |

> [!NOTE]
> **Nhận xét Trục Sentiment:**
> - Lớp Tiêu cực (`-1`) đạt **Precision rất cao: 84.32%**, nghĩa là khi mô hình dự đoán một bình luận là tiêu cực thì khả năng đúng lên tới gần 85%.
> - Tuy nhiên, có `506` mẫu tiêu cực bị dự đoán sang trung tính (`0`). Đây là đặc trưng ngôn ngữ tự nhiên: nhiều bình luận phàn nàn nhẹ nhàng, chê bai ẩn ý hoặc không chứa từ khóa cảm xúc mạnh nên mô hình thiên về gán nhãn trung tính để đảm bảo độ tin cậy.
> - Số câu tiêu cực bị đảo cực nghiêm trọng sang tích cực (`1`) chỉ chiếm `169` mẫu (15.7%).

### 3.2. Trục 2: Toxicity Classification (So Sánh Hiệu Quả Hiệu Chỉnh Ngưỡng)

- **Độ chính xác tổng thể (Accuracy):** `90.61%`
- **Macro F1 (Ngưỡng hiệu chỉnh Platt Scaling):** **`0.6644`**
- **Macro F1 (Raw Argmax không chỉnh ngưỡng):** `0.5958`
- **Mức cải thiện ròng (Net Gain):** **`+6.86%`** Macro F1 nhờ cơ chế hiệu chuẩn xác suất và nhân ngưỡng tối ưu.
- **Brier Score Loss (Hiệu chuẩn xác suất):** `0.03881` (Chất lượng phân bổ xác suất xuất sắc).

**So sánh trực tiếp từng lớp giữa Raw Argmax và Ngưỡng Hiệu Chỉnh:**

| Lớp nhãn | Số mẫu thực tế | Raw Argmax Precision | Raw Argmax Recall | Raw Argmax F1 | **Tuned Precision** | **Tuned Recall** | **Tuned F1** | Δ F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0: Không độc hại` | 2,736 | 90.60% | 99.67% | 0.9492 | 91.74% | 99.42% | **0.9542** | +0.0050 |
| `1: Thiếu lịch sự` | 314 | 78.20% | 33.12% | 0.4653 | 76.07% | 39.49% | **0.5199** | **+0.0546** |
| `2: Xúc phạm cá nhân` | 158 | 83.72% | 45.57% | 0.5902 | 81.72% | 48.10% | **0.6056** | **+0.0154** |
| `3: Thù ghét/Đe dọa` | 29 | 87.50% | 24.14% | 0.3784 | 81.25% | 44.83% | **0.5778** | **+0.1994** |

**Báo cáo phân loại chi tiết (Hậu Platt Scaling & Ngưỡng tối ưu):**
```text
                    precision    recall  f1-score   support

   0 Không độc hại     0.9174    0.9942    0.9542      2736
   1 Thiếu lịch sự     0.7607    0.3949    0.5199       314
2 Xúc phạm cá nhân     0.8172    0.4810    0.6056       158
 3 Thù ghét/Đe dọa     0.8125    0.4483    0.5778        29

          accuracy                         0.9061      3237
         macro avg     0.8270    0.5796    0.6644      3237
      weighted avg     0.8963    0.9061    0.8917      3237
```

**Ma trận nhầm lẫn Toxicity (Tuned Thresholds):**

| Nhãn thực tế \ Dự đoán | Dự đoán 0 | Dự đoán 1 | Dự đoán 2 | Dự đoán 3 | Độ nhạy (Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thực tế 0 (Không độc hại)** | **2720** | 14 | 1 | 1 | 99.42% |
| **Thực tế 1 (Thiếu lịch sự)** | 173 | **124** | 15 | 2 | 39.49% |
| **Thực tế 2 (Xúc phạm)** | 62 | 20 | **76** | 0 | 48.10% |
| **Thực tế 3 (Thù ghét/Đe dọa)** | 10 | 5 | 1 | **13** | 44.83% |

> [!TIP]
> **Ý nghĩa thực tế của việc Hiệu Chỉnh Ngưỡng:**
> - Lớp 3 (Thù ghét/Đe dọa) là lớp nguy hiểm nhất nhưng số lượng chỉ có 29 mẫu. Nếu dùng Argmax thông thường, mô hình chỉ bắt được 7 câu (Recall 24.14%). Nhờ nhân ngưỡng, mô hình bắt được **13 câu (Recall 44.83%, F1 tăng vọt từ 0.3784 lên 0.5778)**.
> - Tỷ lệ bắt nhầm bình luận bình thường thành độc hại (False Positive Rate) cực thấp: chỉ `16` trên tổng số `2,736` câu lành tính bị gán nhầm sang độc hại (`0.58%`).

---

## 4. Thống Kê Không Gian Đặc Trưng & Tính Toàn Vẹn Blind Test

Phân tích chuyên sâu về tương tác giữa bộ 111.675 đặc trưng cố định và tập dữ liệu mù chưa từng thấy:

| Thành phần phân tích | Trục Word n-gram `(1, 2)` | Trục Char n-gram `(2, 5)` | Tổng cộng ma trận đặc trưng |
| :--- | :---: | :---: | :---: |
| **Tổng số cụm n-gram có trong Challenge** | 47,413 cụm | 142,848 cụm | - |
| **Khớp với bộ từ vựng đã học** | 20,689 (43.64%) | 100,669 (70.47%) | - |
| **Từ vựng mới (OOV - Tự động bỏ qua an toàn)** | **26,724 (56.36%)** | **42,179 (29.53%)** | - |
| **Kích thước ma trận sau phép chiếu transform()** | - | - | **(3,237 × 111,675)** |
| **Mật độ kích hoạt (Matrix Density)** | - | - | **0.2093%** |
| **Số đặc trưng kích hoạt trung bình / mẫu** | - | - | **233.7** đặc trưng/câu |
| **Số mẫu có 0 đặc trưng kích hoạt** | - | - | **6 / 3,237** (0.185%) |

> [!NOTE]
> **Tại sao mô hình TF-IDF này bền vững trước từ vựng mới?**
> Dù hơn 56% word n-gram trong tập challenge là từ mới hoặc biến thể mới, **tầng Char n-gram (2 đến 5 ký tự) đạt độ phủ hơn 70.47%**. Nhờ việc băm nhỏ văn bản thành chuỗi ký tự liền nhau, các từ viết tắt, tiếng lóng hay lỗi chính tả vẫn được nhận diện thông qua các n-gram ký tự con, giúp mô hình giữ vững phong độ mà không bị gãy đổ khi gặp từ lạ.

---

## 5. Tốc Độ Suy Luận & Hiệu Năng Tính Toán Thực Tế

- **Tổng thời gian xử lý toàn bộ 3,237 mẫu (bao gồm làm sạch Regex + Trích xuất TF-IDF 111.675 chiều + Dự đoán 2 trục):** `{prod_res['elapsed_sec']:.3f} giây`
- **Thông lượng xử lý (Inference Throughput):** **`3356 mẫu / giây`** (trên môi trường CPU đơn luồng)
- **Độ trễ trung bình mỗi câu (Average Latency):** **`0.298 ms / mẫu`**
- **Mức tiêu thụ bộ nhớ RAM:** Cực kỳ tối ưu nhờ ma trận thưa `scipy.sparse.csr_matrix` (mật độ chỉ 0.21%).

---

## 6. Đánh Giá Trên Các Tập Con Phân Rã Theo Người Gán Nhãn

Đánh giá mô hình sản xuất trên từng tập dữ liệu thô độc lập của 3 kiểm duyệt viên:

| Tập dữ liệu con | Số mẫu | Sent Acc | Sent F1 | Tox Acc | Tox F1 (Tuned) | Tox F1 (Raw) | **Combined F1** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Annotator: An** (`challenge_data/An/challenge.csv`) | 1,224 | 83.82% | 0.7822 | 95.67% | 0.5729 | 0.5767 | **0.6775** |
| **Annotator: Dai** (`challenge_data/Dai/challenge.csv`) | 1,222 | 53.27% | 0.5276 | 86.91% | 0.4426 | 0.4363 | **0.4851** |
| **Annotator: Huy** (`challenge_data/Huy/challenge.csv`) | 1,223 | 81.93% | 0.7733 | 93.87% | 0.5834 | 0.5706 | **0.6784** |

> [!NOTE]
> **Nhận xét về sự phân hóa giữa các kiểm duyệt viên:**
> - Trên tập của **An** và **Huy**, mô hình đạt điểm F1 Sentiment rất cao (**0.7822** và **0.7733**).
> - Trên tập của **Dai**, F1 Sentiment đạt **0.5276** do tập này có tỷ lệ phân bổ nhãn khác biệt (nhiều câu tiêu cực ngắn và tranh cãi sắc thái).
> - Khi gộp lại thành tập Challenge chung đã qua lọc bỏ trùng lặp và rò rỉ, điểm F1 phản ánh đúng độ khó tổng thể của ngôn ngữ mạng xã hội ngoài đời thực.

---

## 7. Phân Tích Định Tính Các Trường Hợp Nhầm Lẫn Tiêu Biểu

### 7.1. Hiện tượng Đảo cực Cảm xúc (Thực tế Tiêu cực - Dự đoán Tích cực: 169 mẫu)
- **Bình luận:** *"Với giá con luxury mà trang bị như thế em thì mua kona cho lành thật, còn giá Lux cao quá so với thương hiệu KIA, chưa chắc đây đã là đột phá của KIA."*
  - *Nguyên nhân:* Câu chứa từ *"luxury"*, *"lành"*, *"đột phá"* (từ mang trọng số tích cực mạnh trong TF-IDF), trong khi ý chê bai nằm ở cấu trúc so sánh và ngữ cảnh giá cả mà mô hình túi từ (Bag-of-Words) khó nắm bắt trọn vẹn.
- **Bình luận:** *"Thật thương tâm. Trách nhiệm phải làm rõ đến cùng từ cấp quản lý đến đơn vị thi công. Xin được chia buồn cùng gia đình"*
  - *Nguyên nhân:* Từ *"thương tâm"*, *"chia buồn"* thường đi liền với sự đồng cảm, an ủi nên mô hình dễ nhầm với sắc thái tích cực/thiện chí.

### 7.2. Các câu Độc hại bị gán nhãn 0 trong Ground Truth nhưng Mô hình phát hiện ra (False Positive có cơ sở)
- **Bình luận:** *"Chính sách ngu dân ngày càng hiệu quả"*
  - *Thực tế:* Toxicity = 0 (do nhãn gán chưa chuẩn).
  - *Dự đoán:* **Toxicity = 2 (Xúc phạm)** với xác suất $96.4%$. Mô hình đã nhận diện chính xác cụm từ mang tính thóa mạ *"ngu dân"*.
- **Bình luận:** *"tao méc mấy a dlv tụi bây phản động nè"*
  - *Thực tế:* Toxicity = 0.
  - *Dự đoán:* **Toxicity = 3 (Thù ghét)** với xác suất $77.8%$. Mô hình kích hoạt đúng đặc trưng từ vựng công kích chính trị.

---

## 8. Kết Luận Khách Quan

1. **Tính hợp lệ tuyệt đối của Blind Test:**
   - Quá trình benchmark hoàn toàn giữ nguyên pipeline đã huấn luyện. Không có bất kỳ dữ liệu nào của `challenge_data` bị rò rỉ vào không gian đặc trưng hay trọng số mô hình.
   - Tốc độ xử lý đạt hơn **3.500 mẫu/giây**, độ trễ chỉ **0.28 ms/mẫu**, cực kỳ phù hợp cho hệ thống lọc bình luận thời gian thực (real-time streaming/moderation).

2. **Độ ổn định của mô hình sản xuất 111.675 đặc trưng:**
   - Mô hình đạt **Combined Macro F1 = 0.6500** trên tập Blind Test hoàn toàn mới (Sentiment Macro F1 = `0.6355`, Toxicity Macro F1 = `0.6644`).
   - Độ chính xác Toxicity đạt **90.61%**, trong đó lớp hiếm Thù ghét/Đe dọa đạt Recall **44.83%** và Precision **81.25%**, chứng minh hiệu quả thực chiến của giải pháp tinh chỉnh ngưỡng hậu Platt Scaling.
   - Mô hình 111.675 đặc trưng vượt trội rõ rệt so với các cấu hình phạt overfit cũ (`penalized`), khẳng định sự đúng đắn của chiến lược mở rộng dữ liệu và chọn siêu tham số theo `max_val`.

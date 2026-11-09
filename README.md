# NT120-TFIDF

Dự án này xây dựng một mô hình đọc bình luận tiếng Việt trên mạng xã hội và gắn cho mỗi bình luận hai nhãn cùng lúc. Nhãn thứ nhất cho biết thái độ của người viết. Nhãn thứ hai cho biết mức độ độc hại của câu chữ.

| Trục | Giá trị | Ý nghĩa |
| :--- | :---: | :--- |
| Sentiment | -1 | Tiêu cực |
| Sentiment | 0 | Trung tính |
| Sentiment | 1 | Tích cực |
| Toxicity | 0 | Không độc hại |
| Toxicity | 1 | Thiếu lịch sự |
| Toxicity | 2 | Xúc phạm cá nhân |
| Toxicity | 3 | Thù ghét hoặc đe dọa |

Mô hình dùng TF-IDF để biến chữ thành số và LinearSVC để phân loại. Cách làm này chạy trên CPU, không cần GPU. File mô hình nặng khoảng 30 MB.

## Những gì đã thực hiện

### 1. Kiểm tra và rà soát dữ liệu

Script `src/step1_data_audit.py` kiểm tra tập train để tìm ô trống, lỗi Unicode, câu trùng và câu trùng nhưng mang nhãn khác nhau. Nhóm xuất các câu có nhãn mâu thuẫn ra file CSV bằng `src/step1b_export_for_review.py`, rà soát thủ công, rồi gộp lại bằng `src/step1c_merge_data.py`. Sau bước này, tập train còn 29.474 bình luận và không còn câu nào mang nhãn mâu thuẫn.

Kết quả kiểm tra cho thấy nhãn Toxicity lệch rất mạnh. Lớp 0 có 26.082 bình luận, còn lớp 3 chỉ có 154 bình luận. Lớp lớn nhất gấp khoảng 169 lần lớp nhỏ nhất. Phần lớn các bước sau nhằm xử lý độ lệch này.

### 2. Làm sạch văn bản ở mức tối thiểu

Mô hình chỉ xóa đường link, thẻ HTML, mốc ngày giờ và ký tự lỗi, rồi chuẩn hóa Unicode về dạng NFC. Mô hình giữ nguyên teencode, emoji, chữ hoa và lỗi chính tả cố ý. Người viết bình luận độc hại thường viết lách như "dcm" hay "óc ch0", nên các dấu hiệu này có ích cho việc phân loại.

### 3. Biến văn bản thành số

Máy tính không đọc chữ trực tiếp. Mô hình cắt mỗi bình luận thành nhiều mảnh nhỏ rồi đếm mức độ quan trọng của từng mảnh bằng TF-IDF. Có hai cách cắt.

- Word n-gram cắt theo từ đơn và cặp hai từ liền nhau. Câu "dịch vụ không tốt" cho ra "dịch", "vụ", "không", "tốt", "dịch vụ", "không tốt". Cặp "không tốt" giúp mô hình phân biệt với chữ "tốt" đứng riêng.
- Char n-gram cắt theo chuỗi 2 đến 5 ký tự liên tiếp. Cách này giúp mô hình nhận ra từ viết tắt hoặc viết sai mà nó chưa gặp nguyên dạng.

### 4. Chọn lọc đặc trưng

Với 37.225 bình luận, hai cách cắt trên tạo ra 333.624 mảnh khác nhau. Giữ hết sẽ tốn bộ nhớ và kéo theo nhiều mảnh chỉ xuất hiện ngẫu nhiên. Mô hình chấm điểm từng mảnh rồi giữ lại các mảnh có điểm cao nhất. Số mảnh giữ lại nằm trong khoảng từ N đến 3N, với N là số bình luận huấn luyện. Mô hình chính giữ 111.675 mảnh.

Điểm của mỗi mảnh gồm một nửa là điểm trung bình trên toàn bộ dữ liệu và một nửa là điểm cao nhất trong từng lớp. Cách tính này giữ lại những từ chỉ xuất hiện trong lớp hiếm như lớp đe dọa, dù chúng ít gặp trên toàn bộ dữ liệu.

### 5. Huấn luyện bộ phân loại

Mỗi trục dùng một LinearSVC riêng. LinearSVC chỉ trả về điểm thô, nên mô hình dùng Platt Scaling để đổi điểm này thành xác suất. Tổng xác suất các lớp của một bình luận luôn bằng 1.

### 6. Dò siêu tham số

C là tham số quyết định mô hình học sát dữ liệu đến đâu. C nhỏ làm mô hình học lỏng và dễ bỏ sót. C lớn làm mô hình học thuộc dữ liệu huấn luyện và kém đi trên dữ liệu mới. Nhóm thử 12 giá trị C từ 0,01 đến 50 cho mỗi trục. Với trục Toxicity, nhóm thử thêm 3 mức tăng trọng số cho lớp hiếm. Mức không tăng trọng số cho kết quả tốt nhất ở cả ba lần chạy.

Mỗi cấu hình chạy qua 5-fold cross-validation. Dữ liệu chia thành 5 phần, mô hình học trên 4 phần và kiểm tra trên phần còn lại, lặp lại 5 lần. Nhờ vậy mỗi bình luận đều có một dự đoán từ mô hình chưa từng học nó. Báo cáo gọi các dự đoán này là OOF.

Nhóm so sánh hai luật chọn C.

- `penalized` lấy điểm kiểm tra trừ đi một phần chênh lệch giữa điểm học và điểm kiểm tra. Luật này ưu tiên mô hình ổn định và chọn C nhỏ.
- `max_val` chọn C có điểm kiểm tra cao nhất.

### 7. Chỉnh ngưỡng cho trục Toxicity

Lớp đe dọa chiếm dưới 1% dữ liệu, nên xác suất của lớp này hiếm khi cao nhất. Nếu chỉ chọn lớp có xác suất cao nhất, mô hình sẽ bỏ sót phần lớn câu đe dọa. Mô hình nhân xác suất của mỗi lớp với một hệ số rồi mới chọn lớp có điểm cao nhất. Nhóm tìm bộ hệ số trên dự đoán OOF. Ở mô hình chính, hệ số của lớp 0, 1, 2, 3 lần lượt là 1; 1,8; 1,8 và 2,8.

### 8. Tự kiểm tra và đánh giá

Sau khi huấn luyện, script chạy 7 bài kiểm tra tự động. Các bài này kiểm tra bộ làm sạch, số đặc trưng, miền giá trị nhãn, tổng xác suất, việc lưu và nạp lại file, điểm của từng lớp hiếm và điểm tổng. Cả ba mô hình đều qua đủ 7 bài.

Nhóm huấn luyện ba phiên bản mô hình rồi đánh giá trên hai tập dữ liệu mà mô hình chưa học.

| Tên | Dữ liệu huấn luyện | Luật chọn C | Số đặc trưng | Vị trí |
| :--- | :--- | :---: | :---: | :--- |
| Mô hình chính | tập train và tập valid, 37.225 bình luận | `max_val` | 111.675 | `artifacts/model_pipeline.joblib` |
| train_only_max_val | tập train, 29.474 bình luận | `max_val` | 88.422 | `artifacts/train_only_max_val/` |
| train_only_penalized | tập train, 29.474 bình luận | `penalized` | 88.422 | `artifacts/train_only_penalized/` |

Mô hình chính gộp tập train với tập valid. Khi gộp, script bỏ dòng rỗng, bỏ câu trùng và bỏ cả nhóm câu trùng có nhãn mâu thuẫn.

## Kết quả đạt được

Các bảng dưới dùng Macro F1. Chỉ số này đi từ 0 đến 1. Nó tính F1 riêng cho từng lớp rồi lấy trung bình, nên lớp hiếm có trọng lượng ngang lớp phổ biến. Một mô hình đoán mọi câu là "không độc hại" vẫn đạt độ chính xác khoảng 85% trên tập challenge, nhưng Macro F1 của trục Toxicity sẽ rất thấp. Cột "Kết hợp" là trung bình Macro F1 của hai trục.

### Trên tập challenge

Tập `challenge_data/challenge.csv` có 3.237 bình luận do ba người gán nhãn An, Dai và Huy chuẩn bị. Theo báo cáo benchmark, không câu nào trong tập này trùng với dữ liệu huấn luyện. Script đánh giá chỉ gọi `transform` và `predict`, không học lại trên tập này.

| Mô hình | Sentiment Macro F1 | Toxicity Macro F1 | Kết hợp |
| :--- | :---: | :---: | :---: |
| Mô hình chính | 0,6355 | 0,6644 | 0,6500 |
| train_only_max_val | 0,6253 | 0,6657 | 0,6455 |
| train_only_penalized | 0,6226 | 0,6393 | 0,6309 |

Kết quả chi tiết của mô hình chính trên tập này như sau.

- Mô hình đoán đúng Sentiment cho 65,15% bình luận và đúng Toxicity cho 90,61% bình luận.
- Chỉnh ngưỡng nâng Toxicity Macro F1 từ 0,5958 lên 0,6644. Với lớp đe dọa, mô hình bắt được 13 trong 29 câu, so với 7 câu khi không chỉnh ngưỡng.
- Mô hình gắn nhầm 16 trong 2.736 bình luận không độc hại thành độc hại, tức 0,58%.
- Trên máy thử nghiệm, mô hình xử lý 3.356 bình luận mỗi giây trên CPU đơn luồng, khoảng 0,3 ms cho mỗi bình luận.

### Trên tập valid

Tập `valid_data/valid.csv` có 7.756 bình luận. Bảng này chỉ có hai mô hình train_only vì mô hình chính đã học tập valid.

| Mô hình | Sentiment Macro F1 | Toxicity Macro F1 | Kết hợp |
| :--- | :---: | :---: | :---: |
| train_only_max_val | 0,6725 | 0,6503 | 0,6614 |
| train_only_penalized | 0,6632 | 0,5976 | 0,6304 |

Luật `max_val` cho điểm kết hợp cao hơn `penalized` 0,031 trên tập valid và 0,015 trên tập challenge. Đây là lý do mô hình chính dùng luật `max_val`. Thêm tập valid vào dữ liệu huấn luyện giúp Sentiment Macro F1 trên tập challenge tăng 0,010, còn Toxicity Macro F1 gần như không đổi.

### Điểm OOF của mô hình chính

Trên 37.225 bình luận huấn luyện, điểm OOF của mô hình chính là 0,6848 cho Sentiment, 0,6680 cho Toxicity và 0,6764 khi kết hợp. Điểm này cao hơn điểm trên tập challenge khoảng 0,026, nên khi dùng thực tế nên tham khảo con số của tập challenge.

### Hạn chế đã biết

- Mô hình bỏ sót nhiều bình luận tiêu cực. Trên tập challenge, mô hình nhận đúng 398 trong 1.073 bình luận tiêu cực, tức 37%. Mô hình xếp 506 bình luận tiêu cực vào nhóm trung tính.
- Mô hình bắt chưa tới một nửa số câu độc hại ở lớp 1, 2 và 3, với tỉ lệ lần lượt là 39%, 48% và 45%. Khi mô hình đã gắn nhãn độc hại thì nhãn đó đúng từ 76% đến 82% số lần.
- Lớp đe dọa chỉ có 29 câu trong tập challenge. Con số của lớp này có thể thay đổi nhiều nếu đổi sang tập khác.
- Điểm kết hợp trên phần dữ liệu của An là 0,6775 và của Huy là 0,6784, nhưng của Dai chỉ là 0,4851. Chênh lệch này có thể đến từ cách gán nhãn khác nhau giữa ba người. Repo chưa kiểm chứng nguyên nhân.
- Mô hình dựa trên từ và chuỗi ký tự, không hiểu ngữ cảnh cả câu. Câu so sánh, mỉa mai hay câu chia buồn có chứa từ mang sắc thái tích cực dễ bị đảo nghĩa.

## Repo đang có gì

```text
NT120-TFIDF/
├── src/
│   ├── step1_data_audit.py          Kiểm tra chất lượng tập train
│   ├── step1b_export_for_review.py  Xuất câu có nhãn mâu thuẫn ra CSV để rà soát
│   ├── step1c_merge_data.py         Gộp dữ liệu sau rà soát vào tập train
│   ├── step2_train_pipeline.py      Định nghĩa pipeline, dò siêu tham số, huấn luyện, tự kiểm tra
│   ├── step3_generate_report.py     Viết báo cáo huấn luyện từ metadata
│   ├── step4_holdout_eval.py        Đánh giá mô hình trên tập valid
│   ├── step5_blind_benchmark.py     Đánh giá mô hình trên tập challenge
│   └── run_background.sh            Chạy step 2 rồi step 3 để ra mô hình chính
├── data/train.csv                   Tập train, 29.474 bình luận
├── valid_data/valid.csv             Tập valid, 7.756 bình luận
├── challenge_data/
│   ├── challenge.csv                Tập challenge đã gộp, 3.237 bình luận
│   └── An/, Dai/, Huy/              Phần dữ liệu riêng của từng người gán nhãn
├── artifacts/
│   ├── model_pipeline.joblib        Mô hình chính
│   ├── metadata.json                Siêu tham số, điểm số, mã SHA256 của mô hình chính
│   ├── oof_predictions.npz          Dự đoán OOF của mô hình chính
│   ├── train_only_max_val/          Mô hình train_only_max_val và metadata
│   └── train_only_penalized/        Mô hình train_only_penalized và metadata
├── reports/
│   ├── data_audit_report.json       Kết quả kiểm tra dữ liệu
│   ├── final_training_report.md     Báo cáo huấn luyện mô hình chính
│   ├── train_only_max_val.md        Báo cáo huấn luyện train_only_max_val
│   ├── train_only_penalized.md      Báo cáo huấn luyện train_only_penalized
│   ├── holdout_eval_valid.md        Kết quả trên tập valid
│   ├── blind_test_benchmark_report.md  Kết quả trên tập challenge
│   └── phase2_training_v2_review.md So sánh lần huấn luyện 1 và 2, đề xuất bước tiếp theo
├── logs/                            Nhật ký huấn luyện và điểm lưu tạm khi dò siêu tham số
└── phase2.md                        Bản thiết kế pipeline huấn luyện
```

Mọi file CSV dùng mã hóa UTF-8 và có ba cột `Comment`, `Sentiment`, `Toxicity`.

## Hướng dẫn dùng mô hình

### Cài đặt

Nhóm huấn luyện và lưu mô hình với Python 3.14 và scikit-learn 1.9.1. Nên dùng đúng phiên bản scikit-learn này. Phiên bản khác có thể không nạp được file mô hình hoặc cho kết quả lệch.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy joblib scikit-learn==1.9.1
```

### Dự đoán cho một vài câu

Chạy đoạn mã sau từ thư mục gốc của repo. File mô hình cần các lớp định nghĩa trong `src/step2_train_pipeline.py`, nên Python phải tìm thấy thư mục `src`.

```python
import __main__
import joblib
import src.step2_train_pipeline as s2

# Một số file mô hình lưu tên lớp dưới __main__, nên cần gắn lại trước khi nạp
for name in ("MinimalVietnameseCleaner", "DynamicParetoTfidfExtractor",
             "DualAxisCalibratedClassifier", "social_word_tokenizer"):
    setattr(__main__, name, getattr(s2, name))

model = joblib.load("artifacts/model_pipeline.joblib")

texts = [
    "Sản phẩm dùng rất tốt, giao hàng nhanh",
    "Đồ ngu, cút đi",
    "Hôm nay trời mưa",
]
print(model.predict(texts))
# [[ 1  0]
#  [-1  2]
#  [ 0  0]]
```

Mỗi hàng ứng với một bình luận. Cột đầu là Sentiment, cột sau là Toxicity. Trong ví dụ trên, câu thứ hai nhận Sentiment -1 và Toxicity 2, tức tiêu cực và xúc phạm cá nhân.

Muốn xem mô hình chắc chắn đến đâu, dùng `predict_proba`. Hàm này trả về hai bảng xác suất.

```python
sent_proba, tox_proba = model.predict_proba(texts)
# sent_proba có 3 cột theo thứ tự nhãn -1, 0, 1
# tox_proba có 4 cột theo thứ tự nhãn 0, 1, 2, 3
```

`predict_proba` trả về xác suất gốc, chưa nhân hệ số ngưỡng. `predict` đã nhân hệ số cho trục Toxicity. Vì vậy, với một số câu, lớp có xác suất cao nhất trong `tox_proba` có thể khác nhãn Toxicity mà `predict` trả về.

### Dự đoán cho cả file CSV

```python
import pandas as pd

df = pd.read_csv("input.csv", encoding="utf-8-sig")
pred = model.predict(df["Comment"].astype(str).tolist())
df["pred_sentiment"] = pred[:, 0]
df["pred_toxicity"] = pred[:, 1]
df.to_csv("output.csv", index=False, encoding="utf-8-sig")
```

### Chạy lại đánh giá

Đánh giá cả ba mô hình trên tập challenge và ghi kết quả vào `reports/blind_test_benchmark_report.md`.

```bash
python -m src.step5_blind_benchmark
```

Đánh giá hai mô hình train_only trên tập valid.

```bash
python -m src.step4_holdout_eval \
    --models artifacts/train_only_max_val/model_pipeline.joblib \
             artifacts/train_only_penalized/model_pipeline.joblib \
    --holdout valid_data/valid.csv \
    --out reports/holdout_eval_valid.md
```

Chỉ nên đánh giá trên tập valid với mô hình chưa học tập valid. Mô hình chính đã học tập này nên điểm sẽ cao hơn thực tế.

### Huấn luyện lại

Lệnh sau huấn luyện lại mô hình chính từ tập train và tập valid, rồi viết báo cáo vào `reports/final_training_report.md`. Lệnh này ghi đè các file trong `artifacts/`.

```bash
bash src/run_background.sh
```

Lần huấn luyện cuối chỉ mất khoảng 17 giây. Bước dò siêu tham số chạy lâu hơn nhiều, riêng giá trị C = 50 trên trục Sentiment mất khoảng 470 giây. Script lưu kết quả từng bước vào `logs/cv_checkpoints_*.jsonl`, nên lần chạy sau có thể dùng lại kết quả cũ nếu dữ liệu không đổi.

Muốn huấn luyện một mô hình chỉ từ tập train, gọi trực tiếp step 2.

```bash
python -m src.step2_train_pipeline \
    --data_path data/train.csv \
    --text_col Comment --sent_col Sentiment --tox_col Toxicity \
    --selection max_val \
    --model_out artifacts/train_only_max_val/model_pipeline.joblib \
    --meta_out artifacts/train_only_max_val/metadata.json \
    --oof_out artifacts/train_only_max_val/oof_predictions.npz
```

Đổi `--selection max_val` thành `--selection penalized` để dùng luật chọn C còn lại.

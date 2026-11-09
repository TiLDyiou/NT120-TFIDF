# BÁO CÁO ĐÁNH GIÁ KẾT QUẢ HUẤN LUYỆN LẦN 2 & ĐỀ XUẤT HOÀN THIỆN HỆ THỐNG

---

## 1. Báo cáo Kết quả Sau khi Train Lần 2 (Đối chiếu với Train Lần 1)

Ở lần train thứ hai, dữ liệu huấn luyện đã được mở rộng bằng cách gộp tập `data/train.csv` ban đầu với tập `valid_data/valid.csv` (loại bỏ các dòng rỗng và các câu trùng lặp có nhãn mâu thuẫn), nâng tổng số mẫu huấn luyện sạch từ **29,474 mẫu** lên **37,225 mẫu**. Đồng thời, luật chọn siêu tham số được chuyển từ cơ chế phạt overfit khắt khe (`penalized`) sang cơ chế tối đa hóa điểm kiểm định (`max_val`).

Dưới đây là bảng so sánh chi tiết hiệu năng giữa hai lần huấn luyện (đo bằng phương pháp 5-Fold Out-of-Fold - OOF):

| Chỉ số / Thành phần | Train Lần 1 (29,474 mẫu) | Train Lần 2 (37,225 mẫu) | Mức độ thay đổi |
| :--- | :---: | :---: | :---: |
| **Quy tắc chọn siêu tham số** | `penalized` (phạt khoảng cách overfit) | `max_val` (ưu tiên điểm kiểm định cao nhất) | Thay đổi chiến lược |
| **Số chiều đặc trưng TF-IDF Pareto** | 88,422 cột (kẹp ở $3 \times N$) | 111,675 cột (kẹp ở $3 \times N$) | +23,253 đặc trưng |
| **Siêu tham số tối ưu Trục Sentiment** | $C = 0.04705$ | $C = 0.22135$ | Tăng độ nhạy với dữ liệu |
| **Siêu tham số tối ưu Trục Toxicity** | $C = 0.01$ (hoặc $0.02169$), $\alpha = 0.0$ | $C = 0.48012$, $\alpha = 0.0$ | Tăng độ phạt sai số |
| **Hệ số ngưỡng hậu Platt Scaling** | `[1.0, 0.8, 1.8, 2.8]` | `[1.0, 1.8, 1.8, 2.8]` | Nâng ưu tiên lớp độc hại |
| **Macro F1 Trục Sentiment** | **0.6660** | **0.6848** | **+0.0188 (+1.88%)** |
| **Macro F1 Trục Toxicity** | **0.5440** (argmax gốc: 0.5013) | **0.6680** (argmax gốc: 0.6397) | **+0.1240 (+12.40%)** |
| **Combined Macro F1 (Trung bình 2 trục)** | **0.6050** | **0.6764** | **+0.0714 (+7.14%)** |
| **Brier Score Loss (Độ lệch xác suất)** | Sent: 0.13646 \| Tox: 0.03240 | Sent: 0.13159 \| Tox: 0.02784 | Giảm (dự đoán chuẩn xác hơn) |
| **Thời gian Re-fit toàn bộ pipeline** | 10.42 giây | 17.04 giây | +6.62 giây |
| **Trạng thái Self-Verification (7 bài test)** | 7/7 PASS | 7/7 PASS | Đạt chuẩn toàn diện |

### Chi tiết hiệu năng từng lớp trên Trục 2 - Toxicity (Nơi dữ liệu bị lệch nhãn nghiêm trọng)
- **Lớp 0 (Không độc hại):** F1-score tăng nhẹ từ `0.9597` lên **`0.9663`** (độ chính xác precision đạt 0.9470, độ bao phủ recall đạt 0.9865).
- **Lớp 1 (Thiếu lịch sự):** F1-score tăng vọt từ `0.5042` lên **`0.6105`** (+10.63%), trong đó recall tăng từ 0.3550 lên 0.5067 (giảm hẳn việc bỏ lọt các câu thô lỗ).
- **Lớp 2 (Xúc phạm cá nhân):** F1-score tăng từ `0.5102` lên **`0.6677`** (+15.75%), precision đạt 0.7591 và recall đạt 0.5960.
- **Lớp 3 (Thù ghét/Đe dọa - lớp cực hiếm):** F1-score tăng gấp hơn hai lần từ `0.2018` lên **`0.4274`** (+22.56%), recall tăng từ 0.1494 lên 0.3171.

---

## 2. Bản chất Những gì Đã Được Thực Hiện

Để hiểu tại sao kết quả lần 2 lại có bước nhảy vọt như trên, quy trình xử lý và huấn luyện gồm 4 mắt xích cốt lõi sau:

### 2.1. Biến văn bản đời thường thành số: N-gram kết hợp Word và Char
- **Bản chất:** Máy tính và các thuật toán phân loại không thể đọc hiểu chữ viết trực tiếp; chúng chỉ có thể tính toán trên các ma trận số.
- **Cách làm:** 
  - **Word n-gram (1, 2):** Cắt câu thành từng từ đơn và từng cặp hai từ đi liền nhau. Ví dụ câu *"dịch vụ không tốt"* sẽ được tách thành các cụm: *"dịch", "vụ", "không", "tốt", "dịch vụ", "không tốt"*. Việc ghép 2 từ liền nhau giúp mô hình phân biệt được ngữ cảnh phủ định (khác hoàn toàn với chữ *"tốt"* đứng một mình).
  - **Char n-gram (2 đến 5 ký tự):** Cắt văn bản thành các chuỗi ký tự liên tiếp có độ dài từ 2 đến 5 chữ cái.
- **Tại sao phải dùng n-gram ký tự?** Trên mạng xã hội Việt Nam, người dùng viết tắt, cố tình sai chính tả hoặc dùng teencode rất nhiều nhằm lách các bộ lọc tự động (ví dụ: *"đm"*, *"dcm"*, *"chóoo"*, *"óc ch0"*). Word n-gram sẽ hoàn toàn bất lực vì coi đó là từ lạ. Ngược lại, Char n-gram cắt nhỏ từ đó thành các mảnh như *"óc"*, *"c c"*, *"ch0"*, giúp mô hình nhận ra ngay các mẫu ký tự độc hại quen thuộc mà không cần người kiểm duyệt phải tự tay lập danh sách từ cấm.

### 2.2. Bộ lọc Pareto 75% Cân bằng Lớp (Class-Balanced Pareto Filtering)
- **Vấn đề thực tế:** Khi băm toàn bộ văn bản thành n-gram từ và ký tự, ta thu được tới **333,624 đặc trưng khác nhau**. Nếu đưa toàn bộ số này vào huấn luyện, bộ nhớ RAM sẽ bị quá tải, việc tính toán rất chậm, và mô hình dễ bị nhiễu bởi các từ rác chỉ xuất hiện ngẫu nhiên 1–2 lần.
- **Nguyên lý Pareto:** Trong thực tế, 20% lượng từ vựng cốt lõi thường mang tới 80% giá trị phân biệt ngữ nghĩa. Do đó, hệ thống chỉ giữ lại các cột mang nhiều thông tin nhất cho tới khi tích lũy đủ 75% tổng giá trị đặc trưng.
- **Ý tưởng cân bằng lớp:** Dữ liệu có sự chênh lệch khủng khiếp (lớp bình thường có gần 33,000 mẫu, trong khi lớp đe dọa chỉ có 246 mẫu). Nếu chỉ lọc theo độ phổ biến toàn cục, các từ ngữ đe dọa hay xúc phạm đặc thù sẽ bị xóa sổ hoàn toàn vì chúng xuất hiện quá ít. Bộ lọc giải quyết việc này bằng cách tính điểm đặc trưng kết hợp: **50% điểm trung bình trên toàn tập** + **50% điểm cao nhất của đặc trưng đó trong từng lớp riêng biệt**. Nhờ đó, một từ dù chỉ xuất hiện trong lớp hiếm nhưng có tính nhận diện cao cho lớp đó vẫn được bảo vệ 100% và giữ lại trong không gian 111,675 đặc trưng.

### 2.3. Thay đổi chiến lược chọn Siêu tham số từ `penalized` sang `max_val`
- **Bản chất của siêu tham số $C$ trong LinearSVC:**
  - $C$ là hệ số kiểm soát mức độ phạt khi đoán sai trên tập huấn luyện.
  - Khi $C$ rất nhỏ: Mô hình "dễ tính", chấp nhận bỏ qua sai sót để tìm đường phân chia ranh giới đơn giản, phẳng lặng. Nguy cơ: mô hình bị nhút nhát (underfitting), không nắm bắt được các sắc thái câu nói phức tạp.
  - Khi $C$ lớn: Mô hình "khắt khe", cố gắng uốn nắn ranh giới để phân loại đúng tối đa các mẫu trong tập học. Nguy cơ: học vẹt (overfitting).
- **Nguyên nhân lần 1 cho kết quả thấp:** Ở lần 1, hệ thống dùng luật `penalized` — lấy điểm kiểm định trừ đi một phần độ lệch giữa điểm tập học và điểm kiểm định. Vì n-gram tuyến tính trên tập học luôn đạt điểm rất cao (~0.95 - 1.0), khoảng cách chênh lệch này lớn. Hệ thống sợ bị phạt nên đã ép chọn $C$ cực nhỏ ($C = 0.01$ đến $0.047$). Kết quả là mô hình quá nhút nhát, bỏ lọt rất nhiều câu xúc phạm.
- **Sự điều chỉnh ở lần 2:** Khi đánh giá thực tế trên tập kiểm định độc lập, cấu hình chọn theo điểm kiểm nghiệm cao nhất (`max_val`) cho thấy khả năng thực chiến tốt hơn hẳn. Ở lần 2, hệ thống cho phép nâng $C$ lên mức $0.22135$ (Sentiment) và $0.48012$ (Toxicity), giúp mô hình đủ độ nhạy để phân biệt rõ các ranh giới phức tạp mà vẫn giữ vững độ ổn định trên 5 fold kiểm tra chéo.

### 2.4. Platt Scaling và Tinh chỉnh Ngưỡng Quyết định Hậu Xác Suất (Threshold Tuning)
- **Platt Scaling là gì?** Thuật toán phân loại LinearSVC vốn chỉ trả về một khoảng cách hình học thô (điểm số âm/dương so với mặt phẳng phân chia), không cho biết xác suất phần trăm đáng tin cậy. Platt Scaling sử dụng hàm hồi quy logistic để nắn các điểm số này thành xác suất phần trăm thực sự (tổng xác suất các lớp cộng lại đúng bằng 100%).
- **Tại sao cần Tinh chỉnh Ngưỡng?** 
  - Bình thường, máy tính sẽ chọn nhãn có xác suất cao nhất (nguyên tắc chọn giá trị lớn nhất - argmax, tức > 50%).
  - Tuy nhiên, do lớp đe dọa (lớp 3) chỉ chiếm dưới 1% dữ liệu, mô hình rất hiếm khi tự tin cho xác suất của lớp này vượt quá 50%. Hệ quả là người dùng bị chửi bới, đe dọa nhưng mô hình vẫn chọn nhãn 0 (bình thường) vì xác suất lớp 0 vẫn chiếm ưu thế.
  - Bộ tinh chỉnh áp dụng hệ số nhân ưu tiên `[1.0, 1.8, 1.8, 2.8]`. Khi xét đoán, xác suất của lớp xúc phạm và lớp đe dọa được nhân hệ số trợ lực. Chỉ cần mô hình phát hiện dấu hiệu đe dọa với xác suất khoảng 30% - 35%, hệ số 2.8 sẽ đẩy điểm của lớp này lên đủ để kích hoạt cảnh báo. Đây chính là yếu tố thực tiễn giúp F1 của lớp thù ghét/đe dọa tăng từ 0.20 lên gần 0.43.

---

## 3. Đề xuất Những gì Cần Thực Hiện Tiếp Theo để Hoàn Thiện

1. **Bổ sung tập dữ liệu kiểm tra độc lập ngoài phân phối (Out-of-Distribution Test Set)**
   - *Hiện trạng:* Do đã gộp cả `train.csv` và `valid.csv` vào để huấn luyện lần 2, hệ thống hiện chỉ còn đo lường qua 5-Fold Cross-Validation (OOF).
   - *Hành động:* Cần thu thập hoặc trích xuất một tập test độc lập (khoảng 2,000 - 3,000 mẫu hoàn toàn mới, chưa từng xuất hiện trong quá trình tối ưu) để đánh giá mù (blind test), đảm bảo độ khách quan trước khi đưa vào ứng dụng thực tế.

2. **Phân tích lỗi định lượng (Error Analysis)**
   - *Hiện trạng:* Lớp 1 (Thiếu lịch sự) và Lớp 2 (Xúc phạm cá nhân) vẫn có sự nhầm lẫn với nhau (F1 dao động khoảng 0.61 - 0.66).
   - *Hành động:* Viết script lọc ra toàn bộ các câu đoán sai giữa Lớp 1 và Lớp 2, xuất ra file để rà soát. Đánh giá xem nguyên nhân do nhãn người gán ban đầu thiếu nhất quán, do câu có tính châm biếm/mỉa mai, hay do thiếu từ khóa đặc trưng.

3. **Bổ sung từ điển chuẩn hóa Teencode / Từ lóng trước khi vector hóa**
   - *Hiện trạng:* Hiện tại bước `MinimalVietnameseCleaner` mới chỉ làm sạch cơ bản (chuẩn hóa Unicode NFC, xóa URL/HTML, giữ nguyên ký tự teencode).
   - *Hành động:* Tích hợp thêm một bảng ánh xạ các từ teencode/từ lóng phổ biến về dạng chuẩn (ví dụ: *"ko", "k", "hổng"* $\to$ *"không"*; *"đc", "dc"* $\to$ *"được"*; *"wa"* $\to$ *"quá"*). Việc này giúp thu gom các biến thể từ vựng về cùng một gốc, làm cô đọng không gian n-gram hơn.

4. **Xây dựng module phục vụ suy luận (Inference Serving Service)**
   - *Hiện trạng:* Mô hình hiện được lưu thành file `artifacts/model_pipeline.joblib` (~30.3 MB), chỉ có thể gọi thông qua script nội bộ.
   - *Hành động:* Xây dựng một script nhẹ hoặc API endpoint (sử dụng FastAPI) nhận chuỗi văn bản đầu vào và trả về JSON chứa nhãn dự đoán cùng xác suất tin cậy của cả 2 trục trong thời gian dưới 5 mili-giây.

5. **Thực hiện benchmark đối đầu với mô hình ngôn ngữ dạng Transformer (PhoBERT)**
   - *Hiện trạng:* Pipeline hiện tại (TF-IDF + LinearSVC) có ưu điểm chạy cực nhanh, tốn rất ít RAM và không cần GPU, nhưng bản chất vẫn là mô hình tuyến tính dựa trên n-gram (không hiểu sâu ngữ cảnh ngữ pháp đảo ngữ hay ẩn dụ).
   - *Hành động:* Huấn luyện một model PhoBERT baseline trên cùng tập dữ liệu này để làm chuẩn so sánh trần hiệu năng, từ đó cân nhắc bài toán đánh đổi giữa độ chính xác và chi phí tài nguyên máy chủ.

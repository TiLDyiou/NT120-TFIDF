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

1. Kiểm tra tập train, rà soát thủ công các câu trùng mang nhãn mâu thuẫn. Sau rà soát, tập train còn 29.474 bình luận.
2. Xây pipeline gồm làm sạch văn bản, trích xuất TF-IDF, chọn lọc đặc trưng và phân loại hai trục.
3. Dò siêu tham số bằng 5-fold cross-validation và so sánh hai luật chọn tham số.
4. Chỉnh ngưỡng quyết định cho trục Toxicity để giảm bỏ sót lớp hiếm.
5. Huấn luyện ba phiên bản mô hình. Mỗi bản qua đủ 7 bài tự kiểm tra về bộ làm sạch, số đặc trưng, miền nhãn, tổng xác suất, lưu và nạp file, điểm lớp hiếm và điểm tổng.
6. Đánh giá trên tập valid và tập challenge.

| Tên | Dữ liệu huấn luyện | Luật chọn C | Số đặc trưng | Vị trí |
| :--- | :--- | :---: | :---: | :--- |
| Mô hình chính | tập train và tập valid, 37.225 bình luận | `max_val` | 111.675 | `artifacts/model_pipeline.joblib` |
| train_only_max_val | tập train, 29.474 bình luận | `max_val` | 88.422 | `artifacts/train_only_max_val/` |
| train_only_penalized | tập train, 29.474 bình luận | `penalized` | 88.422 | `artifacts/train_only_penalized/` |

Mô hình chính gộp tập train với tập valid. Khi gộp, script bỏ dòng rỗng, bỏ câu trùng và bỏ cả nhóm câu trùng có nhãn mâu thuẫn.

## Thuật toán và lý do lựa chọn

Đề tài yêu cầu dùng TF-IDF, nên phần này không bàn lý do chọn TF-IDF. Phần này giải thích cách từng khâu còn lại hoạt động và vì sao nhóm chọn nó. Với mỗi quyết định, README ghi rõ quyết định nào có số đo trong repo và quyết định nào chỉ dựa trên lập luận.

```text
Bình luận
  -> MinimalVietnameseCleaner        làm sạch tối thiểu
  -> DynamicParetoTfidfExtractor     word n-gram và char n-gram, giữ 111.675 cột
  -> DualAxisCalibratedClassifier
       Sentiment   LinearSVC + Platt Scaling
       Toxicity    LinearSVC + Platt Scaling + nhân ngưỡng
```

### Đặc điểm dữ liệu chi phối các lựa chọn

Trục Toxicity lệch rất mạnh. Trong tập train, lớp 0 có 26.082 bình luận, lớp 3 chỉ có 154. Lớp lớn nhất gấp khoảng 169 lần lớp nhỏ nhất. Trục Sentiment lệch nhẹ hơn, khoảng 2,4 lần.

Độ lệch này dẫn đến hai hệ quả. Thứ nhất, độ chính xác không phản ánh đúng chất lượng. Một mô hình luôn đoán "không độc hại" vẫn đúng khoảng 88% trên tập train. Vì vậy nhóm dùng Macro F1 làm thước đo ở mọi bước, từ dò tham số đến đánh giá cuối. Thứ hai, lớp hiếm dễ bị bỏ qua ở bất kỳ khâu nào. Các khâu chọn đặc trưng, chia fold và chỉnh ngưỡng bên dưới đều có cơ chế riêng để giữ lớp hiếm.

### 1. Làm sạch tối thiểu

`MinimalVietnameseCleaner` chuẩn hóa Unicode về dạng NFC rồi xóa thẻ HTML, đường link, mốc ngày giờ và ký tự điều khiển. Bình luận rỗng sau khi làm sạch nhận ký hiệu `<empty_comment>` để pipeline không lỗi.

Bộ làm sạch cố ý không làm bốn việc sau.

- Không chuyển về chữ thường. "NGU" viết hoa toàn bộ thể hiện mức gay gắt khác "ngu".
- Không chuẩn hóa teencode. Kiểu viết lách như "dcm" hay "óc ch0" là dấu hiệu của câu độc hại. Đưa chúng về dạng chuẩn sẽ xóa dấu hiệu đó.
- Không bỏ dấu và không xóa emoji.
- Không dùng công cụ tách từ tiếng Việt. Các công cụ này thường học trên văn bản báo chí và dễ tách sai teencode. Khâu word n-gram bên dưới dùng cặp hai từ để bù cho việc này.

Đường link và mốc thời gian gần như khác nhau ở mỗi bình luận và không mang cảm xúc. Giữ chúng chỉ tạo ra nhiều n-gram xuất hiện một lần.

Repo chưa đo xem chuẩn hóa teencode giúp hay hại. Báo cáo `reports/phase2_training_v2_review.md` đề xuất thử bước này ở lần sau.

### 2. Cấu hình TF-IDF

Pipeline chạy hai bộ TF-IDF song song rồi nối kết quả lại.

Bộ word n-gram tách câu theo khoảng trắng và bỏ dấu câu ở hai đầu mỗi từ. Bộ này lấy từ đơn và cặp hai từ liền nhau. Tiếng Việt có nhiều từ gồm hai âm tiết cách nhau bởi khoảng trắng, như "dịch vụ" hay "chính sách". Cặp hai từ giữ được các từ này mà không cần công cụ tách từ. Cặp hai từ cũng giữ được phủ định, như "không tốt".

Bộ char n-gram lấy mọi chuỗi 2 đến 5 ký tự liên tiếp, kể cả khoảng trắng giữa hai từ. Bộ này xử lý từ viết sai hoặc viết tắt mà mô hình chưa gặp nguyên dạng. Số liệu trên tập challenge cho thấy rõ vai trò của nó. 56,36% word n-gram trong tập challenge chưa từng xuất hiện khi huấn luyện, nhưng 70,47% char n-gram vẫn khớp với từ vựng đã học. Chỉ 6 trong 3.237 bình luận không kích hoạt đặc trưng nào.

Hai tùy chọn khác áp dụng cho cả hai bộ.

- `sublinear_tf=True` thay số lần xuất hiện `tf` bằng `1 + log(tf)`. Một bình luận lặp "ngu" 4 lần nhận trọng số khoảng 2,39 thay vì 4. Nhờ đó, một bình luận spam lặp từ không chi phối mô hình.
- `min_df=2` bỏ n-gram chỉ xuất hiện trong một bình luận. Những n-gram này phần lớn là lỗi gõ, tên riêng hoặc con số. Mô hình không thể học quy luật từ một lần xuất hiện.

### 3. Chọn lọc đặc trưng cân bằng lớp

Với 37.225 bình luận, hai bộ TF-IDF tạo ra 333.624 cột. `DynamicParetoTfidfExtractor` giữ lại một phần theo các bước sau.

1. Tính số cột cần giữ `k = 0,75 × V`, với V là tổng số cột. Sau đó ép k vào khoảng từ N đến 3N, với N là số bình luận huấn luyện.
2. Với mỗi cột j, tính `g_j` là giá trị TF-IDF trung bình trên toàn bộ bình luận.
3. Tính `c_j` là giá trị TF-IDF trung bình cao nhất của cột j trong 7 lớp, gồm 3 lớp Sentiment và 4 lớp Toxicity.
4. Chấm điểm `s_j = 0,5 × g_j / max(g) + 0,5 × c_j / max(c)`.
5. Giữ k cột có điểm cao nhất.

Lý do cần thành phần `c_j` nằm ở lớp hiếm. Giả sử một n-gram chỉ xuất hiện trong lớp 3. Lớp này chiếm 246 trên 37.225 bình luận, tức 0,66%. Điểm trung bình toàn cục của n-gram đó chỉ bằng khoảng 0,66% điểm trung bình trong lớp 3. Nếu chỉ xếp hạng theo `g_j`, bộ chọn gần như chắc chắn loại n-gram này. Thành phần `c_j` cho nó điểm cao vì trong lớp 3 nó xuất hiện dày. Phép chia cho giá trị lớn nhất đưa hai thành phần về cùng thang trước khi cộng.

Điểm `s_j` dùng nhãn, nên nếu chọn đặc trưng trên toàn bộ dữ liệu trước khi chia fold, nhãn của phần kiểm tra sẽ rò vào bước chọn. Code tránh lỗi này bằng cách chạy lại bộ chọn đặc trưng trong từng fold, chỉ trên 4 phần dùng để học.

Có ba điểm cần nói thẳng về khâu này.

- Ở cả ba lần chạy, mốc 75% đều vượt 3N. Với mô hình chính, mốc 75% là 250.218 cột, còn 3N là 111.675. Trên thực tế, số cột do mốc 3N quyết định, quy tắc 75% không có tác dụng.
- Repo không thử mức khác 3N.
- Repo không so sánh cách chấm điểm này với cách chọn theo tần suất hay theo kiểm định chi bình phương. Lợi ích của thành phần `c_j` là lập luận, chưa có số đo riêng.

### 4. LinearSVC cho từng trục

LinearSVC tìm một mặt phẳng tách hai nhóm dữ liệu sao cho khoảng cách từ mặt phẳng đến các điểm gần nhất là lớn nhất. Khoảng cách này gọi là lề. Với nhiều lớp, LinearSVC dùng cách một lớp đối với phần còn lại. Trục Sentiment có 3 mặt phẳng, trục Toxicity có 4. Mỗi mặt phẳng cho một điểm số, lớp có điểm cao nhất thắng.

Mô hình tối thiểu hóa hàm `½‖w‖² + C × Σ max(0, 1 − y·(w·x))²`. Vế đầu giữ trọng số nhỏ để mô hình đơn giản. Vế sau phạt các điểm nằm sai phía hoặc nằm trong lề. C cân bằng hai vế. C lớn ưu tiên phân loại đúng dữ liệu huấn luyện, C nhỏ ưu tiên mô hình đơn giản.

Nhóm chọn mô hình tuyến tính vì hình dạng dữ liệu.

- Mô hình chính có 111.675 cột nhưng chỉ 37.225 bình luận. Mỗi bình luận kích hoạt trung bình khoảng 234 cột, mật độ khoảng 0,21%. Khi số chiều lớn hơn số mẫu, các lớp thường đã tách được bằng mặt phẳng, nên kernel phi tuyến khó mang lại thêm nhiều.
- SVM dùng kernel cần ma trận kernel kích thước N × N. Với 37.225 bình luận, ma trận này có khoảng 1,39 tỷ phần tử, tương đương khoảng 11 GB ở dạng số thực 64 bit. LinearSVC làm việc trực tiếp trên ma trận thưa và có thời gian chạy tăng gần tuyến tính theo số phần tử khác 0.
- Thời gian huấn luyện cho phép dò tham số rộng. Lần huấn luyện cuối của mô hình chính mất 17 giây.

Mỗi trục dùng một mô hình riêng thay vì gộp thành một bài toán 12 lớp. Lý do thứ nhất là mỗi trục cần mức C khác nhau. Mô hình chính dùng C khoảng 0,22 cho Sentiment và 0,48 cho Toxicity. Lý do thứ hai là một số tổ hợp nhãn quá hiếm. Trong tập train, không có bình luận nào vừa trung tính hoặc tích cực vừa thuộc Toxicity 3, và chỉ 14 bình luận vừa trung tính vừa thuộc Toxicity 2. Một mô hình 12 lớp sẽ có những lớp gần như không có dữ liệu.

Repo không so sánh LinearSVC với Logistic Regression, Naive Bayes hay mô hình khác trên cùng bộ đặc trưng. Báo cáo `reports/phase2_training_v2_review.md` đề xuất so với PhoBERT ở bước sau.

### 5. Platt Scaling

LinearSVC chỉ trả về điểm số thô, là khoảng cách có dấu đến mặt phẳng. Điểm này không phải xác suất và không so sánh được giữa các lớp. Platt Scaling học hàm `P = 1 / (1 + exp(A·f + B))`, với f là điểm thô, A và B là hai tham số cần học. Hàm này biến điểm thô thành xác suất từ 0 đến 1.

Code dùng `CalibratedClassifierCV` với `cv=5`. Dữ liệu chia 5 phần. Mỗi lượt, LinearSVC học trên 4 phần và hàm sigmoid học trên phần còn lại. Dự đoán cuối là trung bình của 5 cặp. Với nhiều lớp, mỗi lớp có hàm sigmoid riêng, sau đó code chia lại xác suất các lớp để tổng bằng 1.

Có hai lý do cần xác suất. Khâu nhân ngưỡng ở mục 9 nhân trực tiếp lên xác suất, nên xác suất phải có ý nghĩa. Người dùng cũng cần biết mô hình chắc chắn đến đâu với từng câu.

Nhóm chọn sigmoid thay vì isotonic regression. Isotonic regression là hàm bậc thang linh hoạt, cần nhiều mẫu để không học thuộc. Trong mỗi phần hiệu chuẩn, lớp 3 chỉ có khoảng 40 đến 50 bình luận. Sigmoid chỉ có 2 tham số nên ổn định hơn với lượng dữ liệu này. Repo không chạy thử isotonic để so sánh.

Brier score đo trung bình bình phương sai lệch giữa xác suất dự đoán và kết quả thật, càng gần 0 càng tốt. Trên dự đoán OOF, mô hình chính đạt 0,1316 cho Sentiment và 0,0278 cho Toxicity.

Platt Scaling có chi phí. Mỗi lần huấn luyện thực chất chạy 5 LinearSVC. Khi dò tham số với 5 fold, mỗi cấu hình cần 25 lần huấn luyện LinearSVC. Toàn bộ lưới 48 cấu hình cần 1.200 lần.

### 6. Trọng số lớp cho Toxicity

Một cách phổ biến để xử lý lệch lớp là tăng trọng số lỗi của lớp hiếm. Code dùng công thức `w_c = (N_max / N_c)^α`, với `N_c` là số mẫu của lớp c. α = 0 nghĩa là mọi lớp có trọng số 1. α = 1 là nghịch đảo tần suất đầy đủ, khi đó lớp 3 nặng gấp khoảng 169 lần lớp 0. Trọng số lớn như vậy đẩy mô hình đoán lớp hiếm quá nhiều và làm giảm precision. Số mũ α nhỏ hơn 1 giảm bớt mức tăng này. Nhóm thử α bằng 0; 0,35 và 0,5. Với α = 0,5, trọng số bốn lớp trên tập train là 1; 3,39; 5,17 và 13,01.

Kết quả là α = 0 thắng ở cả ba lần chạy. Với mô hình chính, Macro F1 kiểm tra tốt nhất ứng với α = 0; 0,35 và 0,5 lần lượt là 0,6396; 0,6337 và 0,6341. Chênh lệch dưới 0,006, nhỏ hơn độ lệch chuẩn giữa các fold, khoảng 0,01. Số liệu này không cho thấy trọng số lớp giúp ích trong pipeline hiện tại. Một cách giải thích có thể là bước Platt Scaling học lại xác suất theo phân bố lớp thật của phần hiệu chuẩn, nên phần lớn tác dụng của trọng số bị triệt tiêu. Repo chưa kiểm chứng giải thích này.

### 7. Chia fold phân tầng theo tổ hợp nhãn

Khâu dò tham số dùng 5-fold cross-validation. Dữ liệu chia thành 5 phần. Mô hình học trên 4 phần và kiểm tra trên phần còn lại, lặp lại 5 lần. Nhờ vậy mỗi bình luận đều có một dự đoán từ mô hình chưa từng học nó. Báo cáo gọi các dự đoán này là OOF.

Code chia fold theo tổ hợp hai nhãn, ví dụ `-1_3` là tiêu cực và thù ghét. Mỗi fold giữ đúng tỉ lệ của từng tổ hợp. Lớp 3 có 246 bình luận, nên mỗi fold có khoảng 49 bình luận lớp 3. Nếu chia ngẫu nhiên, một fold có thể nhận ít hơn nhiều và điểm F1 của lớp 3 trên fold đó sẽ dao động mạnh. Code gộp các tổ hợp có ít hơn 5 mẫu vào tổ hợp đông nhất cùng mức Toxicity, nên phân tầng theo Toxicity vẫn giữ được.

Code tính sẵn ma trận TF-IDF của 5 fold một lần rồi dùng lại cho mọi giá trị C. Cách này tiết kiệm thời gian. Mọi cấu hình cũng chạy trên cùng 5 fold, nên chênh lệch điểm chỉ đến từ tham số. Code cố định `random_state=42` để kết quả lặp lại được.

### 8. Luật chọn siêu tham số

Lưới C gồm 12 giá trị từ 0,01 đến 50, mỗi giá trị gấp khoảng 2,17 lần giá trị trước. Nhóm chia theo cấp số nhân vì tác động của C thay đổi theo bậc độ lớn, không theo khoảng cộng.

Code hỗ trợ hai luật chọn.

- `penalized` loại các cấu hình có chênh lệch giữa điểm học và điểm kiểm tra lớn hơn 0,12, rồi xếp theo `điểm kiểm tra − 0,5 × chênh lệch`. Nếu không cấu hình nào qua ngưỡng 0,12, luật này xếp hạng trên toàn bộ.
- `max_val` chọn cấu hình có Macro F1 kiểm tra trung bình cao nhất. Nếu bằng điểm, luật này chọn cấu hình có độ lệch chuẩn nhỏ hơn.

Lần huấn luyện đầu dùng `penalized`. Luật này giả định chênh lệch lớn nghĩa là mô hình sẽ kém trên dữ liệu mới. Với mô hình tuyến tính trên n-gram, giả định này không đúng. Khi C tăng, điểm học tiến gần 1 rất nhanh, nên chênh lệch luôn lớn. Trên trục Sentiment của mô hình chính, chênh lệch là 0,0824 ở C = 0,047 và 0,1213 ở C = 0,102. Chỉ các giá trị C từ 0,047 trở xuống qua được ngưỡng 0,12. Trên trục Toxicity, không cấu hình nào qua ngưỡng. Kết quả là `penalized` luôn chọn C nhỏ, trong khi điểm kiểm tra cao nhất nằm ở C lớn hơn.

Điểm kiểm tra đã đo trên dữ liệu mô hình chưa học, nên nó là thước đo trực tiếp hơn chênh lệch. Số liệu xác nhận điều này. Trên tập valid, `max_val` hơn `penalized` 0,031 điểm kết hợp, phần lớn ở trục Toxicity với 0,6503 so với 0,5976. Trên tập challenge, mức hơn là 0,015.

`max_val` cũng có nhược điểm. Chọn điểm cao nhất trong 12 hoặc 36 cấu hình trên cùng 5 fold làm điểm OOF hơi lạc quan. Các giá trị C gần đỉnh có điểm chênh nhau nhỏ hơn độ lệch chuẩn giữa các fold. Ví dụ trên trục Sentiment, C = 0,102 đạt 0,6826 còn C = 0,221 đạt 0,6847. Vì vậy tập challenge là thước đo cuối cùng.

### 9. Nhân ngưỡng cho Toxicity

Quy tắc mặc định chọn lớp có xác suất cao nhất. Với lớp chiếm dưới 1% dữ liệu, xác suất của lớp đó hiếm khi đứng đầu, kể cả khi mô hình đã nhận ra dấu hiệu. Code đổi quy tắc thành chọn lớp có `τ_c × p_c` cao nhất, với `p_c` là xác suất lớp c và `τ_c` là hệ số của lớp đó. Hệ số của lớp 0 cố định bằng 1.

Code tìm hệ số theo các bước sau.

1. Thử 25 giá trị từ 0,6 đến 3,0, bước 0,1.
2. Dò lần lượt từng hệ số của lớp 1, 2, 3, giữ hai hệ số kia cố định. Chỉ nhận giá trị mới khi Macro F1 tăng. Lặp lại hai vòng.
3. Chạy bước 1 và 2 theo kiểu cross-validation. Với mỗi fold, code tìm hệ số trên xác suất OOF của 4 fold còn lại rồi áp lên fold đang xét. Điểm sau chỉnh ngưỡng vì thế đo trên phần dữ liệu không dùng để chọn hệ số.
4. Lấy trung vị của 5 bộ hệ số làm kết quả cuối. Một fold có giá trị bất thường ít làm lệch trung vị.
5. Chỉ dùng bộ hệ số mới nếu Macro F1 tăng ít nhất 0,003. Nếu không, giữ hệ số bằng 1 cho mọi lớp.

Nhóm chọn nhân hệ số thay vì đặt ngưỡng cứng như "p_3 lớn hơn 0,3 thì chọn lớp 3". Với 4 lớp, ngưỡng cứng gây xung đột khi hai lớp cùng vượt ngưỡng. Nhân hệ số giữ nguyên quy tắc chọn lớp cao nhất, nên mỗi câu luôn nhận đúng một nhãn. Về mặt xác suất, nhân `p_c` với `τ_c` tương đương với giả định lớp c phổ biến hơn `τ_c` lần so với dữ liệu huấn luyện.

Kết quả của mô hình chính là bộ hệ số 1; 1,8; 1,8 và 2,8. Toxicity Macro F1 tăng từ 0,6397 lên 0,6680 trên OOF, và từ 0,5958 lên 0,6644 trên tập challenge. Đổi lại, precision của lớp 3 trên tập challenge giảm từ 87,50% xuống 81,25%. Tỉ lệ nhận đúng bình luận không độc hại giảm từ 99,67% xuống 99,42%.

`predict_proba` vẫn trả về xác suất gốc chưa nhân hệ số. Người dùng muốn đặt quy tắc riêng có thể dùng xác suất này.

Code chỉ chỉnh ngưỡng cho trục Toxicity. Trục Sentiment chưa có bước này, dù trên tập challenge mô hình chỉ nhận đúng 37% bình luận tiêu cực. Áp dụng cách tương tự cho Sentiment là một hướng chưa thử.

### 10. Huấn luyện lại trên toàn bộ dữ liệu

Sau khi chọn xong C, α và hệ số ngưỡng, code huấn luyện lại toàn bộ pipeline trên 100% dữ liệu. Các mô hình trong lúc dò tham số chỉ học trên 80%, nên bước này tận dụng thêm 20% còn lại. Mô hình chính đi xa hơn, gộp cả tập valid vào dữ liệu huấn luyện. Trên tập challenge, việc gộp này tăng Sentiment Macro F1 thêm 0,010. Đổi lại, mô hình chính không còn tập valid để đánh giá độc lập, nên tập challenge là thước đo độc lập duy nhất của nó.

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
- Mô hình dựa trên từ và chuỗi ký tự, không hiểu ngữ cảnh cả câu. Với câu so sánh, câu mỉa mai hay câu chia buồn có chứa từ mang sắc thái tích cực, mô hình dễ hiểu ngược nghĩa.

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

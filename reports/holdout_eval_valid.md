# Đánh giá Holdout: `valid_data/valid.csv` (7,756 mẫu)

| Model | Sent F1 | Tox F1 (thresholded) | Tox F1 (argmax) | Combined | Tox per-class F1 [0,1,2,3] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `artifacts/train_only_penalized/model_pipeline.joblib`<br>C_sent=0.04705, C_tox=0.02169, alpha=0.0 | 0.6632 | 0.5976 | 0.5706 | **0.6304** | [0.9612, 0.4993, 0.5885, 0.3415] |
| `artifacts/train_only_max_val/model_pipeline.joblib`<br>C_sent=0.22135, C_tox=0.22135, alpha=0.0 | 0.6725 | 0.6503 | 0.6210 | **0.6614** | [0.9657, 0.5829, 0.6575, 0.3951] |

## `artifacts/train_only_penalized/model_pipeline.joblib`
```text
              precision    recall  f1-score   support

 -1 Negative     0.6626    0.4017    0.5002      1638
   0 Neutral     0.6989    0.8644    0.7729      3717
  1 Positive     0.7553    0.6814    0.7164      2401

    accuracy                         0.7100      7756
   macro avg     0.7056    0.6492    0.6632      7756
weighted avg     0.7087    0.7100    0.6978      7756

                   precision    recall  f1-score   support

  0 Không độc hại     0.9342    0.9898    0.9612      6866
  1 Thiếu lịch sự     0.8930    0.3466    0.4993       554
       2 Xúc phạm     0.6429    0.5426    0.5885       282
3 Thù ghét/Đe dọa     0.5000    0.2593    0.3415        54

         accuracy                         0.9225      7756
        macro avg     0.7425    0.5345    0.5976      7756
     weighted avg     0.9176    0.9225    0.9103      7756
```

## `artifacts/train_only_max_val/model_pipeline.joblib`
```text
              precision    recall  f1-score   support

 -1 Negative     0.6878    0.4182    0.5201      1638
   0 Neutral     0.7026    0.8714    0.7780      3717
  1 Positive     0.7614    0.6818    0.7194      2401

    accuracy                         0.7170      7756
   macro avg     0.7172    0.6571    0.6725      7756
weighted avg     0.7177    0.7170    0.7054      7756

                   precision    recall  f1-score   support

  0 Không độc hại     0.9444    0.9879    0.9657      6866
  1 Thiếu lịch sự     0.7944    0.4603    0.5829       554
       2 Xúc phạm     0.7389    0.5922    0.6575       282
3 Thù ghét/Đe dọa     0.5926    0.2963    0.3951        54

         accuracy                         0.9310      7756
        macro avg     0.7676    0.5842    0.6503      7756
     weighted avg     0.9238    0.9310    0.9232      7756
```

Chạy lệnh trong thư mục Thực hành 3:

```powershell
python -m pip install -r requirements.txt
python -m data_utils.prepare_data
python main.py --assignment 1
python main.py --assignment 2
python main.py --assignment 3
```

Hoặc mở assignments.ipynb để xem kết quả đã chạy và chạy lại từng bài.

Cấu trúc thư mục:

```text
Thực hành 3/
    assignments.ipynb
    main.py
    requirements.txt
    data_utils/
        prepare_data.py
        uit_vsfc.py
        phoner.py
    models/
        classifiers/
            lstm.py
            gru.py
        taggers/
            bilstm.py
    training/
        experiments.py
        trainer.py
        metrics.py
    utils/
        visualization.py
    datasets/
    outputs/
```

data_utils đọc, tải dữ liệu và tạo batch. models chứa kiến trúc mạng. training phụ trách chạy thí nghiệm, huấn luyện và tính F1. utils chứa hàm vẽ đồ thị. main.py là nơi chọn bài và truyền các tham số chạy.

Bài 1 dùng LSTM 5 lớp, bài 2 dùng GRU 5 lớp, cùng hidden size 256. Hai bài phân loại topic của UIT-VSFC theo code mẫu. Đổi sang sentiment bằng --label-key sentiment nếu cần.

Bài 3 dùng BiLSTM 5 lớp, hidden size 256 cho mỗi chiều. Đầu ra 512 chiều của mỗi token được đưa qua Linear để dự đoán nhãn BIO. Dữ liệu dùng bản word của PhoNER_COVID19, giữ nguyên chữ hoa và dấu câu.

Các bài dùng Adam, tối đa 30 epoch và dừng sớm nếu F1 dev không tăng trong 5 epoch. Model tốt nhất trên dev được nạp lại trước khi đánh giá test. Đặt --patience 0 nếu muốn chạy đủ số epoch.

Kết quả đã chạy trên toàn bộ dữ liệu:

| Bài | Mô hình | Số epoch | Epoch tốt nhất | F1 test |
|---|---|---:|---:|---:|
| 1 | LSTM | 11 | 6 | 0.7551 |
| 2 | GRU | 8 | 3 | 0.7468 |
| 3 | BiLSTM | 16 | 11 | 0.8739 |

Bài 1 và 2 dùng macro-F1. Bài 3 dùng micro-F1 theo thực thể: phải đúng cả vị trí và loại thực thể (strict IOB2), không tính padding. Notebook đã lưu log huấn luyện, đồ thị loss/F1 và báo cáo trên test.

Mỗi bài lưu best_model.pt, vocab.json, history.json và results.json trong outputs/bai1, outputs/bai2 hoặc outputs/bai3.

Nguồn dữ liệu:

- UIT-VSFC: https://drive.google.com/drive/folders/1rdcXNGt_3-QUvV8EtSvVsLMVeHmk6Yqk
- PhoNER: https://github.com/VinAIResearch/PhoNER_COVID19

Script tải dữ liệu vào datasets/. datasets/ và outputs/ chỉ lưu ở máy, không đưa lên GitHub. PhoNER dùng cho học tập và không được phân phối lại.

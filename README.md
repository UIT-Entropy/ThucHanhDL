# Thực hành Deep Learning

## Thực hành 1

[Notebook bài làm](<Thực Hành 1/Bai thuc hanh 01.ipynb>) và code module xây dựng MLP trên MNIST.

```bash
cd "Thực Hành 1"
pip install numpy
python main.py
```

## Thực hành 2

[Notebook bài làm](<Thực Hành 2/assignments.ipynb>) gồm phần chạy, đánh giá và vẽ đồ thị cho các mô hình:

| Bài | Mô hình | Dữ liệu |
|---|---|---|
| 1 | LeNet | MNIST |
| 2 | GoogLeNet | MNIST |
| 3 | ResNet-18 | VinaFood21 |
| 4 | Fine-tune ResNet50 pretrained từ Hugging Face | VinaFood21 |

```bash
cd "Thực Hành 2"
pip install -r requirements.txt
```

Mở `assignments.ipynb` và chạy lần lượt các ô. Notebook tự chọn GPU nếu có, hoặc CPU khi chạy local. Code mô hình nằm trong `modeling`, loader nằm trong `Load_data`; `support/trainer.py` phụ trách huấn luyện, `support/visualization.py` phụ trách vẽ đồ thị và `metrics.py` phụ trách đánh giá.

Notebook thực hành 2 đã có output chạy trên Kaggle GPU Tesla T4, đủ 10 epoch cho mỗi mô hình, bảng độ đo và 15 hình. Accuracy test: LeNet 98,27%; GoogLeNet 98,93%; ResNet-18 49,55%; ResNet50 86,23%. ResNet-18 còn chưa hội tụ sau 10 epoch; xem nhận xét cuối notebook.

## Thực hành 3

[Notebook bài làm](<Thực hành 3/assignments.ipynb>) có output huấn luyện và đánh giá trên toàn bộ dữ liệu.

| Bài | Mô hình | Dữ liệu | F1 test |
|---|---|---|---:|
| 1 | LSTM 5 lớp, hidden size 256 | UIT-VSFC (topic) | 0.7551 |
| 2 | GRU 5 lớp, hidden size 256 | UIT-VSFC (topic) | 0.7468 |
| 3 | BiLSTM 5 lớp, hidden size 256 | PhoNER | 0.8739 |

```bash
cd "Thực hành 3"
pip install -r requirements.txt
python -m data_utils.prepare_data
python main.py --assignment 1
python main.py --assignment 2
python main.py --assignment 3
```

Code được chia thành `data_utils` (dữ liệu), `models` (kiến trúc mạng), `training` (huấn luyện và đánh giá), `utils` (đồ thị). Các bài dùng Adam và early stopping theo F1 dev. Bài 1, 2 dùng macro-F1; bài 3 dùng F1 theo thực thể. Xem [hướng dẫn thực hành 3](<Thực hành 3/README.md>) để biết cấu hình và cách chạy.

## Dữ liệu

Dữ liệu và trọng số không được đưa lên GitHub. Đặt 4 file MNIST vào thư mục `data` của mỗi bài thực hành:

```text
data/
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
```

VinaFood21 đặt trong `Thực Hành 2/data/VinaFood21`, gồm hai thư mục `train` và `test`, mỗi thư mục có 21 thư mục lớp. Loader cũng hỗ trợ cấu trúc lồng `VinaFood21/VinaFood21/train` và `test`.

Trên Kaggle, thêm code và dữ liệu vào Input, bật GPU và Internet để tải pretrained ResNet50. Trọng số và kết quả được lưu trong `/kaggle/working/checkpoints`.

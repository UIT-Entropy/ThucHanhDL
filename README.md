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

Mở `assignments.ipynb` và chạy lần lượt các ô. Notebook tự chọn GPU nếu có, hoặc CPU khi chạy local. Code mô hình nằm trong `modeling`, loader nằm trong `Load_data`; phần huấn luyện, đánh giá và vẽ đồ thị dùng các module riêng.

Kết quả huấn luyện GPU của thực hành 2 đang được chạy trên Kaggle và sẽ cập nhật sau.

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

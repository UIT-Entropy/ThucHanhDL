# Thực hành Deep Learning

## Thực hành 1

[Notebook bài làm](<Thực Hành 1/Bai thuc hanh 01.ipynb>)

```bash
cd "Thực Hành 1"
pip install numpy
python main.py
```

## Thực hành 2

[Notebook bài làm](<Thực Hành 2/assignments.ipynb>)

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

## Thực hành 3

[Notebook bài làm](<Thực hành 3/assignments.ipynb>) 

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




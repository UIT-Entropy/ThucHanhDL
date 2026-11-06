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

Nguồn dữ liệu:

- UIT-VSFC: https://drive.google.com/drive/folders/1rdcXNGt_3-QUvV8EtSvVsLMVeHmk6Yqk
- PhoNER: https://github.com/VinAIResearch/PhoNER_COVID19


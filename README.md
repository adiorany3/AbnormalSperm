# Sperm Detector Streamlit App

Aplikasi Streamlit untuk menjalankan model `best_model.pkl` yang berisi `state_dict` PyTorch MobileNetV2 grayscale dengan 3 kelas.

## Struktur file

```text
sperm_detector_streamlit/
├── app.py
├── best_model.pkl
├── requirements.txt
└── README.md
```

## Jalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub baru.
2. Upload `app.py`, `best_model.pkl`, `requirements.txt`, dan `README.md` ke repository tersebut.
3. Buka Streamlit Community Cloud.
4. Pilih repository dan arahkan main file ke `app.py`.
5. Deploy.

## Catatan penting

- Model ini memakai input grayscale ukuran 224 x 224.
- File `best_model.pkl` adalah `state_dict`, bukan objek model penuh.
- Default urutan label di `app.py` adalah:

```python
CLASS_NAMES = ["Abnormal", "Non-Sperm", "Normal"]
```

Jika notebook training Kaggle memiliki `class_to_idx` berbeda, ubah `CLASS_NAMES` agar sesuai.

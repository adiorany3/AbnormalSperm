import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image


# =========================================================
# Konfigurasi dasar
# =========================================================
APP_TITLE = "Sperm Detector"
MODEL_PATH = Path(__file__).parent / "best_model.pkl"
IMAGE_SIZE = 224

# Urutan label sesuai target_names evaluasi notebook:
# 0 = Normal_Sperm, 1 = Abnormal_Sperm, 2 = Non_Sperm
CLASS_NAMES = ["Normal_Sperm", "Abnormal_Sperm", "Non_Sperm"]


# =========================================================
# Arsitektur MobileNetV2 grayscale yang sesuai dengan state_dict best_model.pkl
# Tidak memakai torchvision agar deploy Streamlit Cloud lebih ringan/stabil.
# =========================================================
class Conv2dNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU6,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2

        layers = OrderedDict()
        layers["0"] = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        layers["1"] = norm_layer(out_channels)
        if activation_layer is not None:
            layers["2"] = activation_layer(inplace=True)
        super().__init__(layers)


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError("stride harus 1 atau 2")

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(
                Conv2dNormActivation(
                    inp,
                    hidden_dim,
                    kernel_size=1,
                    activation_layer=nn.ReLU6,
                )
            )

        layers.extend(
            [
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    activation_layer=nn.ReLU6,
                ),
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Gray(nn.Module):
    def __init__(self, num_classes: int = 3, width_mult: float = 1.0, dropout: float = 0.2):
        super().__init__()

        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * max(1.0, width_mult))

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [
            Conv2dNormActivation(
                1,
                input_channel,
                stride=2,
                activation_layer=nn.ReLU6,
            )
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(
            Conv2dNormActivation(
                input_channel,
                self.last_channel,
                kernel_size=1,
                activation_layer=nn.ReLU6,
            )
        )

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@st.cache_resource(show_spinner="Memuat model...")
def load_model() -> nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}. Pastikan best_model.pkl berada satu folder dengan app.py."
        )

    model = MobileNetV2Gray(num_classes=len(CLASS_NAMES))

    # File model ini tersimpan sebagai pickle OrderedDict/state_dict.
    with open(MODEL_PATH, "rb") as f:
        state_dict = pickle.load(f)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def preprocess_image(image: Image.Image, mean: float, std: float) -> torch.Tensor:
    image = image.convert("L")  # grayscale 1 channel
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - mean) / std
    arr = np.expand_dims(arr, axis=0)  # channel: 1 x H x W
    arr = np.expand_dims(arr, axis=0)  # batch: 1 x 1 x H x W

    return torch.from_numpy(arr)


def predict(image: Image.Image, model: nn.Module, mean: float, std: float):
    x = preprocess_image(image, mean, std)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    return top_idx, probs


# =========================================================
# UI Streamlit
# =========================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🔬", layout="centered")

st.title("🔬 Sperm Detector")
st.write(
    "Aplikasi ini menjalankan model MobileNetV2 grayscale dari file `best_model.pkl` "
    "untuk klasifikasi citra sperma ke 3 kelas."
)

with st.sidebar:
    st.header("Pengaturan")
    st.caption("Samakan nilai mean/std dengan preprocessing saat training di notebook Kaggle.")
    mean = st.number_input("Normalize mean", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    std = st.number_input("Normalize std", min_value=0.01, max_value=1.0, value=0.5, step=0.05)

    st.divider()
    st.write("Urutan label aktif:")
    for i, name in enumerate(CLASS_NAMES):
        st.write(f"{i}: {name}")

uploaded_file = st.file_uploader(
    "Unggah gambar sperm / non-sperm",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is None:
    st.info("Silakan unggah gambar terlebih dahulu.")
else:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    try:
        model = load_model()
        pred_idx, probabilities = predict(image, model, mean, std)

        st.subheader("Hasil Prediksi")
        st.success(f"Prediksi: **{CLASS_NAMES[pred_idx]}**")
        st.metric("Confidence", f"{probabilities[pred_idx] * 100:.2f}%")

        st.subheader("Probabilitas Tiap Kelas")
        prob_table = {
            "Kelas": CLASS_NAMES,
            "Probabilitas (%)": [round(float(p) * 100, 2) for p in probabilities],
        }
        st.dataframe(prob_table, use_container_width=True, hide_index=True)
        st.bar_chart({name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)})

    except Exception as e:
        st.error("Model gagal dijalankan.")
        st.exception(e)

st.caption(
    "Catatan: aplikasi ini hanya alat demonstrasi computer vision, bukan pengganti analisis klinis/laboratorium."
)

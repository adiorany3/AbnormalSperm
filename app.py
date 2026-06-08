import streamlit as st
import torch
import torch.nn as nn
import pickle

from PIL import Image
from torchvision import models, transforms

# =====================================
# LOAD CHECKPOINT
# =====================================
@st.cache_resource
def load_model():

    with open("best_model.pkl", "rb") as f:
        checkpoint = pickle.load(f)

    class_names = checkpoint["class_names"]

    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint

model, checkpoint = load_model()

CLASS_NAMES = checkpoint["class_names"]

IMAGE_SIZE = checkpoint["image_size"]

NORMALIZE_MEAN = checkpoint["normalize_mean"]
NORMALIZE_STD = checkpoint["normalize_std"]

# =====================================
# TRANSFORM
# =====================================
transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD
    )
])

# =====================================
# UI
# =====================================
st.title("Sperm Morphology Classification")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    x = transform(image)

    x = x.unsqueeze(0)

    with torch.no_grad():

        outputs = model(x)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            dim=1
        )

    pred_idx = pred.item()

    st.success(
        f"Prediction : "
        f"{CLASS_NAMES[pred_idx]}"
    )

    st.write(
        f"Confidence : "
        f"{confidence.item()*100:.2f}%"
    )

    st.subheader("Class Probabilities")

    for i, name in enumerate(CLASS_NAMES):

        st.write(
            f"{name}: "
            f"{probs[0][i].item()*100:.2f}%"
        )

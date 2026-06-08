import streamlit as st
import torch
import torch.nn as nn
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Sperm Morphology Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# HIDE STREAMLIT EMBLEM
# ==================================================
hide_style = """
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

[data-testid="stToolbar"]{
display:none;
}

.block-container{
padding-top:1rem;
padding-bottom:1rem;
}

</style>
"""

st.markdown(
    hide_style,
    unsafe_allow_html=True
)

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():

    with open("best_model.pkl", "rb") as f:
        checkpoint = pickle.load(f)

    class_names = checkpoint["class_names"]

    model = models.mobilenet_v2(
        weights=None
    )

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

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:

    st.header("Model Information")

    st.write(
        "Architecture:",
        checkpoint.get(
            "architecture",
            "MobileNetV2"
        )
    )

    st.write(
        "Image Size:",
        IMAGE_SIZE
    )

    st.write(
        "Classes:",
        len(CLASS_NAMES)
    )

    st.write(
        "Best Validation F1:",
        checkpoint.get(
            "best_valid_f1",
            "N/A"
        )
    )

# ==================================================
# TRANSFORM
# ==================================================
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

# ==================================================
# HEADER
# ==================================================
st.title(
    "🧬 Sperm Morphology Intelligence System"
)

st.markdown("""
Artificial Intelligence for sperm morphology classification.

Supported Classes:

- Normal_Sperm
- Abnormal_Sperm
- Non_Sperm
""")

# ==================================================
# FILE UPLOADER
# ==================================================
uploaded_files = st.file_uploader(
    "Upload Sperm Image(s)",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True
)

# ==================================================
# PREDICTION
# ==================================================
if uploaded_files:

    for uploaded_file in uploaded_files:

        st.divider()

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:

            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True
            )

        with col2:

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

            predicted_class = (
                CLASS_NAMES[pred_idx]
            )

            confidence_pct = (
                confidence.item() * 100
            )

            st.success(
                f"Prediction: {predicted_class}"
            )

            st.metric(
                "Confidence",
                f"{confidence_pct:.2f}%"
            )

            st.progress(
                float(confidence.item())
            )

            # ==================================
            # INSIGHT
            # ==================================
            if predicted_class == "Normal_Sperm":

                insight = """
                Normal morphology detected.

                Morphological structure appears
                similar to normal sperm samples
                observed during training.
                """

            elif predicted_class == "Abnormal_Sperm":

                insight = """
                Abnormal morphology detected.

                Potential abnormalities may
                involve head, midpiece or tail.

                Manual expert review is recommended.
                """

            else:

                insight = """
                Non-sperm object detected.

                Image may contain debris,
                staining artifacts,
                epithelial cells,
                or non-sperm structures.
                """

            st.info(insight)

            if confidence_pct < 60:

                st.warning(
                    "Low confidence prediction. "
                    "Manual verification recommended."
                )

            elif confidence_pct > 90:

                st.success(
                    "High confidence prediction."
                )

        # ======================================
        # PROBABILITY TABLE
        # ======================================
        st.subheader(
            "Class Probabilities"
        )

        prob_df = pd.DataFrame({

            "Class":
                CLASS_NAMES,

            "Probability (%)":
                [
                    round(
                        p * 100,
                        2
                    )
                    for p in probs[0]
                    .cpu()
                    .numpy()
                ]
        })

        st.dataframe(
            prob_df,
            use_container_width=True
        )

        # ======================================
        # BAR CHART
        # ======================================
        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.bar(
            CLASS_NAMES,
            probs[0]
            .cpu()
            .numpy()
        )

        ax.set_title(
            "Prediction Distribution"
        )

        ax.set_ylabel(
            "Probability"
        )

        st.pyplot(fig)

        # ======================================
        # DOWNLOAD JSON
        # ======================================
        result = {

            "filename":
                uploaded_file.name,

            "prediction":
                predicted_class,

            "confidence":
                round(
                    confidence_pct,
                    2
                ),

            "probabilities": {

                CLASS_NAMES[i]:
                    round(
                        probs[0][i]
                        .item() * 100,
                        2
                    )

                for i in range(
                    len(CLASS_NAMES)
                )
            }
        }

        st.download_button(
            label="Download Result JSON",
            data=json.dumps(
                result,
                indent=4
            ),
            file_name=f"{uploaded_file.name}.json",
            mime="application/json"
        )

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")

st.caption(
    "Research Use Only • "
    "Not intended for clinical diagnosis."
)

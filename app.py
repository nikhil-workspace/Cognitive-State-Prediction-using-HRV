import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, random_split

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="HRV Cognitive State Prediction",
    page_icon="🫀",
    layout="centered"
)

LABELS = {
    0: "No Stress",
    1: "Interruption",
    2: "Time Pressure"
}

INTERVENTIONS = {
    0: "Focus Boosters — short meditation or a deep-work playlist.",
    1: "Mindfulness & Re-focusing — 5-minute breathing or a quick stretch.",
    2: "Stress Reduction & Grounding — relaxation, guided imagery, or a short break."
}


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


@st.cache_resource
def train_model():
    import kagglehub

    path = kagglehub.dataset_download(
        "qiriro/swell-heart-rate-variability-hrv"
    )

    csv_path = os.path.join(
        path,
        "hrv dataset",
        "data",
        "final",
        "train.csv"
    )

    df = pd.read_csv(csv_path).dropna()

    # Same four features and preprocessing used in the notebook
    features = df[["MEAN_RR", "RMSSD", "SDRR", "LF_HF"]].copy()
    features["LF_HF"] = np.log1p(features["LF_HF"])

    mapping = {
        "no stress": 0,
        "interruption": 1,
        "time pressure": 2
    }

    y = df["condition"].map(mapping).values

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_data, test_data = random_split(
        dataset,
        [train_size, test_size]
    )

    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True
    )

    model = SimpleMLP()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for _ in range(10):
        model.train()

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        test_loader = DataLoader(
            test_data,
            batch_size=64,
            shuffle=False
        )

        for inputs, labels in test_loader:
            predictions = model(inputs).argmax(1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    # Save model and scaler for faster future startup
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "accuracy": accuracy
        },
        "hrv_model.pth"
    )

    return model, scaler, accuracy


@st.cache_resource
def load_saved_model():
    checkpoint = torch.load(
        "hrv_model.pth",
        weights_only=False
    )

    model = SimpleMLP()

    # Accept either saved-model format.
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        raise KeyError(
            "hrv_model.pth does not contain model weights. "
            "Please save the model again using the code provided."
        )

    model.eval()

    scaler = checkpoint["scaler"]
    accuracy = checkpoint.get("accuracy", None)

    return model, scaler, accuracy


# ---------------------------------------------------------
# Clean UI styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1050px;
        padding-top: 3.5rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 0.35rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }

    .result-card {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        margin-top: 10px;
    }

    .state-card {
        border: 1px solid #dbe7ff;
        border-radius: 16px;
        padding: 20px;
        text-align: left;
        background: #f8fbff;
    }

    .state-name {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 6px;
    }

    .confidence-number {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 6px;
    }

    .intervention-card {
        border-left: 4px solid #4f7cff;
        background: #f8fbff;
        border-radius: 10px;
        padding: 14px 16px;
        line-height: 1.5;
    }

    .probability-label {
        font-weight: 600;
        margin-top: 8px;
    }

    div[data-testid="stButton"] > button {
        border-radius: 10px;
        height: 3rem;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🫀 HRV Cognitive State Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enter four HRV measurements to predict the current cognitive state.</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
if not os.path.exists("hrv_model.pth"):
    st.error(
        "hrv_model.pth was not found. Save the trained model from your notebook "
        "and place hrv_model.pth in the same folder as app_clean.py."
    )
    st.stop()

try:
    model, scaler, test_accuracy = load_saved_model()
except Exception as e:
    st.error(f"Could not load hrv_model.pth: {e}")
    st.stop()


# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    mean_rr = st.number_input(
        "Average Time Between Heartbeats",
        min_value=0.0,
        value=800.0,
        step=1.0,
        help="MEAN_RR — Average time between consecutive heartbeats, in milliseconds."
    )

    rmssd = st.number_input(
        "Short-Term Heartbeat Variation",
        min_value=0.0,
        value=35.0,
        step=1.0,
        help="RMSSD — Short-term variation between heartbeats, in milliseconds."
    )

with col2:
    sdrr = st.number_input(
        "Overall Heartbeat Variation",
        min_value=0.0,
        value=50.0,
        step=1.0,
        help="SDRR — Overall variation in RR intervals, in milliseconds."
    )

    lf_hf = st.number_input(
        "Low-to-High Frequency Ratio",
        min_value=0.0,
        value=1.5,
        step=0.1,
        help="LF/HF — Ratio between low- and high-frequency HRV components."
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button(
    "Predict Cognitive State",
    type="primary",
    use_container_width=True
):
    # IMPORTANT:
    # LF/HF is transformed exactly as during model training.
    x = np.array([
        [mean_rr, rmssd, sdrr, np.log1p(lf_hf)]
    ])

    x = scaler.transform(x)

    with torch.no_grad():
        probabilities = torch.softmax(
            model(
                torch.tensor(
                    x,
                    dtype=torch.float32
                )
            ),
            dim=1
        )[0].numpy()

    state_id = int(np.argmax(probabilities))
    confidence = probabilities[state_id] * 100

    st.divider()

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns([1, 1.25])

    with result_col1:
        st.markdown(
            f"""
            <div class="state-card">
                <div>Predicted Cognitive State</div>
                <div class="state-name">{LABELS[state_id]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with result_col2:
        st.markdown(
            f"""
            <div class="result-card">
                <div>Prediction Confidence</div>
                <div class="confidence-number">{confidence:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Recommended Intervention")

    st.markdown(
        f"""
        <div class="intervention-card">
            {INTERVENTIONS[state_id]}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Class Probabilities")

    for i, label in LABELS.items():
        probability = probabilities[i] * 100

        st.markdown(
            f'<div class="probability-label">{label} — {probability:.1f}%</div>',
            unsafe_allow_html=True
        )

        st.progress(float(probabilities[i]))

    if test_accuracy is not None:
        st.caption(f"Model test accuracy: {test_accuracy:.2f}%")

# HRV Cognitive State Prediction

A machine learning project for predicting **cognitive states using HRV (Heart Rate Variability) biomedical signals**.

The project uses four HRV-derived features to classify observations into three cognitive-state categories:

- **No Stress**
- **Interruption**
- **Time Pressure**

The trained model is an **MLP (Multi-Layer Perceptron) neural network** and is deployed through a **Streamlit** web application.

## Project Overview

The model uses the following HRV features:

| Feature | Description |
|---|---|
| `MEAN_RR` | Average time between heartbeats |
| `RMSSD` | Short-term variation between successive heartbeats |
| `SDRR` | Overall variation in RR intervals |
| `LF_HF` | Ratio of low-frequency to high-frequency HRV components |

The `LF_HF` feature is transformed using:

`log1p(LF_HF) = ln(1 + LF_HF)`

This reduces the effect of very large LF/HF values before standardization.

## Machine Learning Workflow

1. Load the HRV dataset.
2. Remove missing values.
3. Select the four HRV features.
4. Apply `log1p` transformation to `LF_HF`.
5. Standardize the features using `StandardScaler`.
6. Encode the three target classes.
7. Handle class imbalance using balanced class weights.
8. Split the data into training and testing sets using an 80/20 split.
9. Train an MLP neural network.
10. Evaluate the model using accuracy, loss, precision, recall and F1-score.
11. Save the trained model and scaler.
12. Use Streamlit for interactive state prediction.

## Model Architecture

The MLP architecture is:

`4 input features → 64 neurons → 64 neurons → 3 output classes`

- Hidden-layer activation: **ReLU**
- Optimizer: **Adam**
- Learning rate: **0.001**
- Batch size: **64**
- Epochs: **10**
- Loss function: **Weighted Cross-Entropy Loss**

Balanced class weights are used so that the model does not overly favor the majority class.

## Model Performance

The trained model achieved approximately:

- **Test Accuracy: 93.41%**
- **Test Loss: 0.1604**
- **Macro F1-score: ~0.93**

Class-wise F1-scores:

| State | F1-score |
|---|---:|
| No Stress | 0.95 |
| Interruption | 0.94 |
| Time Pressure | 0.89 |

## Streamlit Application

The Streamlit application allows the user to enter the four HRV feature values and receive:

- Predicted cognitive state
- Prediction confidence
- Class probabilities
- Recommended intervention based on the predicted state

The application loads the already-trained model and scaler from `hrv_model.pth`, so the model does not need to be retrained every time the application starts.

## Project Structure

```text
Biomedical Project/
│
├── app.py
├── code.ipynb
├── hrv_model.pth
├── Project Report.docx
├── req.txt
├── README.md
│
└── myvenv/
```

> `myvenv/` should not be uploaded to GitHub. It is included in `.gitignore`.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv myvenv
myvenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r req.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## Files

- **`app.py`** — Streamlit application for interactive prediction.
- **`code.ipynb`** — Model development, preprocessing, training and evaluation.
- **`hrv_model.pth`** — Saved trained model and scaler.
- **`Project Report.docx`** — Detailed project report.
- **`req.txt`** — Python dependencies.
- **`README.md`** — Project documentation.

## Dataset

The project is based on the **SWELL-KW HRV dataset** and focuses on predicting three cognitive states associated with stress-related working conditions.

## Limitations

- The model is based on a limited set of four HRV-derived features.
- Predictions represent learned patterns from the dataset and should not be treated as medical diagnoses.
- Performance on new users or different recording conditions may differ from the reported test performance.

## Future Scope

Possible improvements include:

- Testing additional HRV features.
- Comparing the MLP with other machine learning and deep learning models.
- Hyperparameter tuning.
- Testing the model on independent datasets.
- Improving real-time HRV signal processing and prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Streamlit
- Jupyter Notebook
- Git & GitHub

## Author

**Nikhil**

Final Year B.Sc. Data Science Project

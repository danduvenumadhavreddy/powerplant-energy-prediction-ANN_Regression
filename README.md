# ⚡ Power Plant Energy Output Prediction using ANN (PyTorch)

An Artificial Neural Network (ANN) regression model built with **PyTorch** to predict the **net hourly electrical energy output (PE)** of a Combined Cycle Power Plant (CCPP) from ambient environmental conditions.

## 📌 Overview

| Item | Detail |
|---|---|
| Task | Supervised regression |
| Framework | PyTorch, scikit-learn |
| Dataset size | 9,568 samples, 4 features, 1 target |
| Test R² score | **0.934** |
| Test MSE | **18.91** |

## 📊 Dataset

File: `powerplant_data.csv`

| Feature | Description | Unit |
|---|---|---|
| `AT` | Ambient Temperature | °C |
| `V` | Exhaust Vacuum | cm Hg |
| `AP` | Ambient Pressure | mbar |
| `RH` | Relative Humidity | % |
| `PE` | **Target**: Net hourly electrical energy output | MW |

## 🧠 Model Architecture

```
Input (4) → Linear(6) → ReLU → Linear(6) → ReLU → Linear(1)
```

- **Loss:** Mean Squared Error (MSELoss)
- **Optimizer:** Adam
- **Batch size:** 32
- **Epochs:** 100
- **Checkpointing:** best model (lowest validation loss) saved to `best_model.pt`

## 🔄 Workflow

1. Load data with pandas
2. Split features/target (`PE` is the target)
3. Train/test split (80/20, `random_state=42`)
4. Standardize features with `StandardScaler` (fit on train only, to avoid data leakage)
5. Convert to PyTorch tensors, wrap in `TensorDataset` and `DataLoader`
6. Train the ANN with per-epoch validation tracking
7. Save the best checkpoint, then plot training vs. validation loss
8. Evaluate with MSE and R², and compare predicted vs. actual values

## 📈 Results

| Metric | Train | Test |
|---|---|---|
| MSE | 20.66 | 18.91 |
| R² | n/a | 0.934 |

Predictions closely track actual values, e.g. predicted 435.07 vs actual 433.27 MW.

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn torch matplotlib jupyter
```

### Run
```bash
git clone https://github.com/danduvenumadhavreddy/powerplant-energy-prediction-ANN_Regression.git
cd powerplant-energy-prediction-ann
jupyter notebook ANN_Regression.ipynb
```

## 📁 Project Structure

```
├── ANN_Regression.ipynb   # Model training & evaluation notebook
├── powerplant_data.csv    # Dataset
└── README.md
```

## 🔮 Future Improvements

- Increase network width/depth and tune hyperparameters
- Add learning-rate scheduling and early stopping
- Compare against baselines (Linear Regression, Random Forest, XGBoost)
- Add RMSE/MAE metrics and an actual-vs-predicted scatter plot
- Wrap the model in a simple Streamlit/Flask app for inference

## 📚 Dataset Source

Combined Cycle Power Plant dataset.


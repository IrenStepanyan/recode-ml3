# Recode ml3

## 🔍 Project Overview
A machine learning solution for automated pattern recognition in text-based communications. This project implements a robust classification pipeline to distinguish between two distinct categories in messaging data.

**Keywords:** `filtering` · `message` · `authenticity` 

## 📊 Dataset Structure

### Input Files
- **Training Dataset**: 640 samples
- **Validation Dataset**: 160 samples

### Features
The dataset contains 4 numerical features and 1 binary target label:

| Column | Type | Description |
|--------|------|-------------|
| `feature_1` | float64 | Normalized continuous variable |
| `feature_2` | float64 | Normalized continuous variable |
| `feature_3` | float64 | Normalized continuous variable (0-1 range) |
| `feature_4` | int64 | Discrete categorical variable (0-9) |
| `label` | int64 | Binary target (0 or 1) |

**Class Distribution:**
- Class 0: ~70.6% (majority class)
- Class 1: ~29.4% (minority class)

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.7+
```

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## 🚀 How to Run

### Option 1: Google Colab 
1. Upload `Recode_ml3.ipynb` to Google Colab
2. Run all cells sequentially (Runtime → Run all)
3. The notebook will automatically download datasets from Google Drive

### Option 2: Local Jupyter Notebook
1. Install Jupyter: `pip install jupyter`
2. Launch notebook: `jupyter notebook Recode_ml3.ipynb`
3. Ensure dataset URLs are accessible or modify to use local files


## 📈 Modeling Approach

### Pipeline Architecture
```
Data Loading → Preprocessing → Model Training → Evaluation → Model Selection → Serialization
```

### 1. **Data Preprocessing**
- Missing value handling via median imputation
- Feature scaling using StandardScaler
- Outlier detection (IQR method, ~0.78% outliers detected)

### 2. **Models Implemented**
Two classification algorithms were trained and compared:

#### **Model 1: Logistic Regression**
- Solver: L-BFGS
- Regularization: C=1.0
- Max iterations: 1000

#### **Model 2: Support Vector Machine (Linear)**
- Kernel: Linear
- Regularization: C=1.0
- Probability estimates: Enabled

### 3. **Evaluation Metrics**
Models were evaluated using:
- Accuracy (training & validation)
- Precision
- Recall (Sensitivity)
- F1-Score
- ROC-AUC Score
- Confusion Matrix

### 4. **Model Selection**
**Selected Model:** Logistic Regression

**Justification:**
- Validation Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUCn)

The model was selected based on superior F1-score and precision metrics, which are critical for minimizing false positive predictions in the target application domain.

## 📁 Project Structure
```
├── Recode_ml3.ipynb                 # Main notebook with full pipeline
└── README.md                        # This file
```

## 🔄 Model Deployment

### Saving the Model
The final model is saved as a scikit-learn Pipeline including:
1. **Imputer** (median strategy)
2. **StandardScaler** (fitted on training data)
3. **LogisticRegression** (trained classifier)
```python
import joblib
pipeline = joblib.load('logistic_model_pipeline.pkl')
```

### Making Predictions
```python
# Load model
model = joblib.load('logistic_model_pipeline.pkl')

# Prepare new data (must have 4 features: feature_1 through feature_4)
new_data = pd.DataFrame({
    'feature_1': [value1],
    'feature_2': [value2],
    'feature_3': [value3],
    'feature_4': [value4]
})

# Predict
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)
```

# 🏠 Boston House Price Prediction

A machine learning web application that predicts Boston house prices based on 13 key features. Built with Python, Flask, and Scikit-learn — fully deployable as a web app.

---

## 🚀 Live Demo

Run locally at: `http://127.0.0.1:5000`

---

## 📌 Project Overview

This project takes 13 socio-economic and geographic features of a Boston neighborhood and predicts the **median house price** using a trained Linear Regression model. The model is served through a Flask web application where users can input values and get real-time predictions.

---

## 🛠️ Tech Stack

| Category | Tools Used |
|---|---|
| Language | Python 3 |
| ML Library | Scikit-learn |
| Web Framework | Flask |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Saving | Pickle |
| Frontend | HTML, CSS |
| Version Control | Git & GitHub |
| Deployment | Gunicorn, Heroku/Netlify |

---

## 📂 Project Structure

```
Boston-House-Price-Prediction/
├── app.py                  # Flask web application
├── retrain.py              # Script to regenerate model files
├── housepred.pkl           # Trained Linear Regression model
├── scaler.pkl              # Fitted StandardScaler
├── requirements.txt        # Python dependencies
├── Procfile                # Deployment config
├── netlify.toml            # Netlify deployment config
├── templates/
│   └── home.html           # Frontend HTML form
└── Boston_House_Price_Prediction.ipynb  # Full EDA + training notebook
```

---

## 🧠 Machine Learning Techniques Used

### 1. Data Preprocessing
- Loaded dataset from CSV using **Pandas**
- Checked for null values, data types, and statistics
- Split data into **80% training / 20% testing** using `train_test_split`

### 2. Feature Scaling
- Applied **StandardScaler** (Z-score normalization) to normalize all 13 features
- Formula: `scaled = (value - mean) / std`
- Saved the fitted scaler as `scaler.pkl` for consistent predictions

### 3. Model Training
- Trained a **Linear Regression** model on scaled training data
- Evaluated using **R² Score** on test data

### 4. Model Serialization
- Saved trained model and scaler using **Pickle** (`.pkl` files)
- Loaded at app startup for real-time inference

---

## 📊 Dataset — Boston Housing

The dataset contains **506 samples** with **13 input features**:

| Feature | Description |
|---|---|
| CRIM | Per capita crime rate |
| ZN | Proportion of residential land zoned |
| INDUS | Proportion of non-retail business acres |
| CHAS | Charles River dummy variable (1 if tract bounds river) |
| NOX | Nitric oxide concentration |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of owner-occupied units built before 1940 |
| DIS | Weighted distance to employment centres |
| RAD | Index of accessibility to radial highways |
| TAX | Full-value property tax rate |
| PTRATIO | Pupil-teacher ratio |
| B | Proportion of Black residents |
| LSTAT | % lower status of the population |

**Target Variable:** `MEDV` — Median value of homes in $1000s

---

## 🌐 Flask Web App — How It Works

### Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Renders the home page with input form |
| `/predict` | POST | Accepts form input, returns predicted price |
| `/predict_api` | POST | Accepts JSON input, returns prediction as JSON |

### Prediction Flow

```
User fills form (13 values)
        ↓
Flask reads form data
        ↓
Convert to NumPy array
        ↓
Scale using scaler.pkl
        ↓
Predict using housepred.pkl
        ↓
Multiply by 1000 (convert to dollars)
        ↓
Display result on page
```

---

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/boston-house-price-prediction.git
cd boston-house-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate model files
```bash
python retrain.py
```

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

---

## 📦 Requirements

```
Flask
pandas
numpy
matplotlib
seaborn
scikit-learn
catboost
gunicorn
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Linear Regression |
| R² Score | ~0.67 |
| Test Size | 20% |
| Random State | 42 |

---

## 🔍 Exploratory Data Analysis

The notebook includes:
- Correlation **heatmap** of all features
- **Pairplot** across all variables
- Individual **regression plots** for each feature vs price
- Data info, description, null checks, and dtype analysis

---

## 🐛 Common Issues & Fixes

| Error | Fix |
|---|---|
| `FileNotFoundError: housepred.pkl` | Run `python retrain.py` first |
| `UnpicklingError: invalid load key` | Delete old pkl files and run `retrain.py` again |
| `TemplateNotFound: home.html` | Make sure `home.html` is inside a `templates/` folder |
| Prediction shows dollars not thousands | Multiply output by 1000 in `app.py` |

---

## 👤 Author

**Your Name**
- GitHub: [@YourUsername](https://github.com/YourUsername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

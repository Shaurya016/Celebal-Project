# Celebal-Project
# 🔍 Creditworthiness Prediction Using Random Forest

This project focuses on predicting whether an individual is creditworthy or not using various financial and personal attributes. A Random Forest ensemble model is trained on the Statlog German Credit Data for accurate classification. 

As an additional step, I have deployed the project as an interactive web application using Streamlit, allowing users to test the model in real time.
## 🚀 Live App

🌐[Click here to open the app](https://celebal-project-kj7if9sv4qxjbqgm7ga5oa.streamlit.app/)

*(Deployed using Streamlit Cloud)*


## 📦 Features

- 🧠 Random Forest model trained on the German Credit Data dataset
- 📈 Feature importance visualization
- ✅ Interactive input form for live prediction
- 📊 Confusion matrix to evaluate model performance
- 🔒 Predicts whether an applicant is creditworthy or not


## 📁 Files in This Repo

| File                      | Description                                |
|---------------------------|--------------------------------------------|
| `app.py`                  | Streamlit app source code                  |
| `random_forest_credit_model.joblib` | Trained Random Forest model           |
| `requirements.txt`        | Python dependencies for Streamlit Cloud    |
| `README.md`               | Project overview (this file)               |

---

## 📊 Dataset

- **Source**: [UCI Statlog German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Records**: 1,000 individuals
- **Target**: Creditworthy (1) vs Not Creditworthy (0)

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Joblib

## 📊 Visualizations(in the project)

  1. Target Distribution
  2. Correlation Heatmap
  3. Feature Importance Curve
  4. Confusion Matrix
  5. Classification Report Heatmap
  6. ROC Curve
  7. Prediction Probability Distribution
---

## 📌 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Shaurya016/Celebal-Project.git
cd Celebal-Project

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

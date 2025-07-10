# ChurnGuard: AI-Powered AutoML + SHAP + Visual Analytics

ChurnGuard is a no-code, AutoML-powered dashboard built with Streamlit that allows users to:
- Upload any CSV dataset
- Automatically detect target columns and train ML models (RandomForest, XGBoost, LightGBM, CatBoost)
- View beautiful, interactive charts, pies, histograms, and business analytics
- Explore model explainability via SHAP
- Fully supports binary and multi-class datasets

## 🔧 Features
- Auto-detect categorical/numerical columns
- Visual EDA: heatmaps, distributions, pie charts
- Model training and evaluation with classification reports
- Confusion matrix and SHAP plots
- Streamlit interface with real-time controls

## 🧠 ML Algorithms
- Random Forest
- XGBoost
- LightGBM
- CatBoost

## 📁 Folder Structure
```
ChurnGuard/
├── dashboard/
│   ├── streamlit_app.py       # Main Streamlit dashboard app
│   ├── check_model_path.py    # Model path checker (optional)
│   └── churn_model.pkl        # Trained model (optional)
├── data/
│   ├── telco_customer_churn.csv
│   └── cleaned_telco.csv
├── models/
│   └── churn_model.pkl        # Pickled trained model
├── notebooks/
│   ├── 01_EDA_and_Cleaning.ipynb
│   ├── 02_EDA_and_Visualization.ipynb
│   ├── 03_SHAP_Explainability.ipynb
│   └── 04_Sentiment_Feature.ipynb
└── requirements.txt
```

## 🚀 Running the App

```bash
pip install -r requirements.txt
streamlit run dashboard/streamlit_app.py
```

## 📝 License
MIT License
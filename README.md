# InsightIQ: AI-Powered AutoML + SHAP + Visual Analytics

InsightIQ is a no-code, AutoML-powered dashboard built with Streamlit that allows users to:
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
InsightIQ/
│
├── dashboard/                 # Streamlit app
│   └── streamlit_app.py
│
├── data/                     # Raw and cleaned datasets
│   ├── telco_customer_churn.csv
│   └── cleaned_telco.csv
│
├── notebooks/                # Jupyter notebooks (above)
│   ├── 01_data_loading_and_cleaning.ipynb
│   ├── 02_eda_and_visuals.ipynb
│   └── 03_model_training_and_shap.ipynb
│
├── utils/                    # Helper functions
│   └── preprocessing.py
│
├── requirements.txt
└── README.md

```

## 🚀 Running the App

```bash
pip install -r requirements.txt
streamlit run dashboard/streamlit_app.py
```

## 📝 License
MIT License

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Page Config
st.set_page_config(page_title="InsightIQ - AutoML & SHAP Dashboard", layout="wide")
st.title("📊 InsightIQ: Universal AutoML + SHAP + Visual Analytics")

# Upload File
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data loaded successfully!")
    st.write("### 📌 Sample Data")
    st.dataframe(df.head())

    if st.checkbox("🔍 Show Data Overview"):
        st.write(df.describe(include="all"))
        st.write("Shape:", df.shape)
        st.write("Missing values:", df.isnull().sum())

    if st.checkbox("📈 Show Business Visual Insights"):
        st.subheader("Correlation Heatmap")
        fig = plt.figure(figsize=(12, 6))
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig)

        st.subheader("Distribution Plots (Numeric Features)")
        for col in df.select_dtypes(include=[np.number]).columns[:4]:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig)

        st.subheader("Pie Charts (Top Categorical Columns)")
        for col in df.select_dtypes(include='object').columns[:2]:
            pie = df[col].value_counts().reset_index()
            pie.columns = [col, 'count']
            fig = px.pie(pie, names=col, values='count', title=f"{col} Distribution")
            st.plotly_chart(fig)

    target = st.selectbox("🎯 Select Target Column (Label)", [None] + list(df.columns))
    if target:
        X = df.drop(columns=[target])
        y = df[target]

        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        if y.dtypes == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) < 20 else None
        )

        model_name = st.selectbox("🔍 Choose Model", ["RandomForest", "XGBoost", "LightGBM", "CatBoost"])
        stop_training = st.checkbox("🛑 Stop Training")
        train_btn = st.button("🚀 Train Model")

        if train_btn:
            with st.spinner("Training..."):
                if stop_training:
                    st.warning("Training was cancelled by the user.")
                else:
                    if model_name == "RandomForest":
                        model = RandomForestClassifier()
                    elif model_name == "XGBoost":
                        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
                    elif model_name == "LightGBM":
                        model = LGBMClassifier()
                    else:
                        model = CatBoostClassifier(verbose=0)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.success("✅ Training Complete!")

                    # Confusion Matrix
                    st.subheader("🔢 Confusion Matrix")
                    try:
                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                        st.pyplot(fig_cm)
                    except Exception as e:
                        st.warning(f"Confusion matrix error: {e}")

                    # Visual Classification Report
                    st.subheader("📊 Visualized Classification Metrics")
                    try:
                        report_dict = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report_dict).transpose().reset_index()
                        report_df = report_df[report_df['index'].str.isnumeric()]
                        report_df.rename(columns={'index': 'Class'}, inplace=True)

                        fig_report = px.bar(report_df, x='Class', y=['precision', 'recall', 'f1-score'],
                                            barmode='group', title="Precision, Recall, F1-Score per Class")
                        st.plotly_chart(fig_report)
                    except Exception as e:
                        st.warning(f"Classification report error: {e}")

                    # ROC-AUC
                    try:
                        y_proba = model.predict_proba(X_test)
                        auc = roc_auc_score(y_test, y_proba, multi_class="ovr") if y_proba.shape[1] > 2 else roc_auc_score(y_test, y_proba[:, 1])
                        st.metric("🎯 ROC-AUC Score", round(auc, 4))
                    except Exception as e:
                        st.warning(f"ROC-AUC Score could not be calculated: {e}")

                    # SHAP
                    st.subheader("🧠 SHAP Explainability")
                    try:
                        explainer = shap.Explainer(model, X_train)
                        shap_values = explainer(X_test)

                        st.write("📌 Global Feature Importance")
                        fig_bar = shap.plots.bar(shap_values, show=False)
                        st.pyplot(fig_bar)

                        st.write("📌 SHAP Summary Plot")
                        fig_summary = plt.figure()
                        shap.summary_plot(shap_values, X_test, show=False)
                        st.pyplot(fig_summary)
                    except Exception as e:
                        st.warning(f"SHAP could not be computed: {e}")
else:
    st.info("Upload a dataset to get started.")

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Detector", page_icon="🕵️", layout="wide")
st.markdown("<h1 style='text-align: center;'>🕵️ Fraud Transaction Predictor</h1>", unsafe_allow_html=True)
st.markdown("This app uses your trained model to detect fraudulent transactions from the original dataset.")

df = pd.read_csv("E:\Git\Fraud-transaction-detection\Model\pipeline_with_smote.pkl")
df = df.drop(['step', 'nameDest', 'nameOrig', 'isFlaggedFraud'], axis=1)
df['orig_balance_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['dest_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']

pipeline = joblib.load("E:\\Git\\Fraud-transaction-detection\\Model\\best_pipeline_Logistic_Regression.pkl")

predictions = pipeline.predict(df.drop('isFraud', axis=1))
probs = pipeline.predict_proba(df.drop('isFraud', axis=1))[:, 1]

df['Prediction'] = predictions
df['Probability'] = probs
df['Result'] = df['Prediction'].apply(lambda x: 'Fraudulent' if x == 1 else 'Legitimate')

st.subheader("📊 Prediction Summary")
fraud_count = df['Result'].value_counts().get('Fraudulent', 0)
total = len(df)
st.metric(label="🔍 Fraudulent Transactions Detected", value=fraud_count, delta=f"{(fraud_count/total)*100:.2f}% of total")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=(fraud_count / total) * 100,
    title={'text': "Fraud Rate (%)"},
    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red"}}
))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Top 50 Predictions")
st.dataframe(df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'Result', 'Probability']].head(50))

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Full Prediction Results", data=csv, file_name="fraud_predictions.csv", mime="text/csv")
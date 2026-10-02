import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_iris

# กำหนดขนาดหน้าจอให้กว้าง
st.set_page_config(page_title="Iris Flower Classifier", layout="wide")

# โหลดข้อมูล Iris เพื่อใช้หาค่าเฉลี่ยของ Dataset
iris = load_iris()
target_names = ["Setosa", "Versicolor", "Virginica"]
feature_names = iris.feature_names
dataset_means = iris.data.mean(axis=0)


# โหลดโมเดลที่บันทึกไว้
@st.cache_resource
def load_trained_model():
    return joblib.load("iris_model.pkl")


try:
    model = load_trained_model()
except Exception:
    st.error(
        "ไม่พบไฟล์ 'iris_model.pkl' กรุณารัน Model Training.ipynb เพื่อสร้างโมเดลก่อน"
    )
    st.stop()

# --- Sidebar: Input Features ---
st.sidebar.header("📊 Input Features")
st.sidebar.caption("Adjust the sliders to input flower measurements:")

sepal_length = st.sidebar.slider(
    "🌿 Sepal Length (cm)", min_value=4.0, max_value=8.0, value=6.1, step=0.1
)
sepal_width = st.sidebar.slider(
    "🌿 Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.3, step=0.1
)
petal_length = st.sidebar.slider(
    "🌸 Petal Length (cm)", min_value=1.0, max_value=7.0, value=3.9, step=0.1
)
petal_width = st.sidebar.slider(
    "🌸 Petal Width (cm)", min_value=0.1, max_value=2.6, value=1.2, step=0.1
)

predict_btn = st.sidebar.button("🌸 Predict Species", type="primary")

# รวมค่าอินพุตจากผู้ใช้
user_inputs = [sepal_length, sepal_width, petal_length, petal_width]

# --- Main Layout ---
st.title("🌺 Iris Flower Classifier")
st.markdown("#### Predict the species of Iris flowers using Machine Learning")

col1, col2 = st.columns([1.1, 1], gap="large")

# --- การทำนาย ---
input_df = pd.DataFrame([user_inputs], columns=feature_names)
prediction_idx = model.predict(input_df)[0]
predicted_species = target_names[prediction_idx]
probabilities = model.predict_proba(input_df)[0]
confidence = probabilities[prediction_idx] * 100

# --- ฝั่งซ้าย: กราฟแท่งเปรียบเทียบ Input vs Dataset Average ---
with col1:
    st.subheader("📈 Input Visualization")
    st.caption("**Your Input vs Dataset Average**")

    labels = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

    fig_comp = go.Figure()
    fig_comp.add_trace(
        go.Bar(
            x=labels,
            y=user_inputs,
            name="Your Input",
            marker_color="#EA4335",
            text=[f"{v:.1f}" for v in user_inputs],
            textposition="auto",
        )
    )
    fig_comp.add_trace(
        go.Bar(
            x=labels,
            y=dataset_means,
            name="Dataset Average",
            marker_color="#1A73E8",
            text=[f"{v:.2f}" for v in dataset_means],
            textposition="auto",
        )
    )

    fig_comp.update_layout(
        barmode="group",
        yaxis_title="Value (cm)",
        xaxis_title="Features",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# --- ฝั่งขวา: ผลการทำนายและกราฟ Probability ---
with col2:
    st.subheader("🎯 Prediction Result")

    # การ์ดแสดงผลลัพธ์สีม่วง/น้ำเงินตามดีไซน์ในภาพ
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #5C6BC0, #512DA8); 
                    padding: 28px; border-radius: 14px; text-align: center; color: white; margin-bottom: 25px;">
            <div style="font-size: 22px; font-weight: 500; opacity: 0.9;">Predicted Species</div>
            <div style="font-size: 38px; font-weight: 800; margin: 12px 0;">{predicted_species}</div>
            <div style="font-size: 18px; font-weight: 400; opacity: 0.95;">Confidence: {confidence:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("📊 Probability Distribution")

    # กราฟแท่งความน่าจะเป็นของแต่ละคลาส
    prob_df = pd.DataFrame(
        {"Species": target_names, "Probability (%)": probabilities * 100}
    )

    fig_prob = px.bar(
        prob_df,
        x="Species",
        y="Probability (%)",
        color="Probability (%)",
        color_continuous_scale=["#FFCDD2", "#4CAF50"],
        text=prob_df["Probability (%)"].apply(lambda x: f"{x:.1f}%"),
    )
    fig_prob.update_traces(textposition="outside")
    fig_prob.update_layout(
        yaxis_range=[0, 115],
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
        coloraxis_showscale=True,
    )
    st.plotly_chart(fig_prob, use_container_width=True)
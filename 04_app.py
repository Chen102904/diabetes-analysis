"""
第 4 步：交互式数据看板（Streamlit）
这是项目的"门面"，面试时可以现场打开演示。

运行：streamlit run 04_app.py
然后浏览器自动打开 http://localhost:8501
"""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="糖尿病数据分析看板", page_icon="🩺", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("data/diabetes_clean.csv")


df = load_data()

# ---------- 侧边栏筛选 ----------
st.sidebar.header("🔍 筛选条件")
age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("年龄范围", age_min, age_max, (age_min, age_max))

bmi_min, bmi_max = float(df["BMI"].min()), float(df["BMI"].max())
bmi_range = st.sidebar.slider("BMI 范围", bmi_min, bmi_max, (bmi_min, bmi_max))

glucose_min, glucose_max = int(df["Glucose"].min()), int(df["Glucose"].max())
glucose_range = st.sidebar.slider("血糖范围 (mg/dL)", glucose_min, glucose_max,
                                  (glucose_min, glucose_max))

mask = (
    df["Age"].between(*age_range)
    & df["BMI"].between(*bmi_range)
    & df["Glucose"].between(*glucose_range)
)
fdf = df[mask]

# ---------- 标题 ----------
st.title("🩺 糖尿病数据分析看板")
st.caption("数据集：Pima Indians Diabetes (UCI) ｜ 共 768 条记录 ｜ "
           "作者：[你的名字]")

if len(fdf) == 0:
    st.warning("当前筛选条件下没有数据，请放宽条件。")
    st.stop()

# ---------- 核心指标卡 ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("筛选后样本数", f"{len(fdf)} 人")
c2.metric("患病人数", f"{int(fdf['Outcome'].sum())} 人")
c3.metric("患病率", f"{fdf['Outcome'].mean() * 100:.1f}%")
c4.metric("平均年龄", f"{fdf['Age'].mean():.1f} 岁")

st.divider()

# ---------- 图表区 ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("血糖分布对比")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(fdf[fdf["Outcome"] == 0]["Glucose"], bins=20, alpha=0.7,
            label="未患病", color="#4A90D9")
    ax.hist(fdf[fdf["Outcome"] == 1]["Glucose"], bins=20, alpha=0.7,
            label="患病", color="#D9534F")
    ax.set_xlabel("血糖 (mg/dL)")
    ax.set_ylabel("人数")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("BMI 分布对比")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(fdf[fdf["Outcome"] == 0]["BMI"], bins=20, alpha=0.7,
            label="未患病", color="#4A90D9")
    ax.hist(fdf[fdf["Outcome"] == 1]["BMI"], bins=20, alpha=0.7,
            label="患病", color="#D9534F")
    ax.set_xlabel("BMI")
    ax.set_ylabel("人数")
    ax.legend()
    st.pyplot(fig)

st.divider()

# ---------- 分组统计表 ----------
st.subheader("📊 患病组 vs 未患病组 指标均值")
features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "Age", "DiabetesPedigreeFunction"]
grouped = fdf.groupby("Outcome")[features].mean().T
grouped.columns = ["未患病", "患病"]
grouped["差值"] = grouped["患病"] - grouped["未患病"]
st.dataframe(grouped.style.format("{:.2f}"), use_container_width=True)

# ---------- 原始数据 ----------
with st.expander("查看原始数据（前 50 行）"):
    st.dataframe(fdf.head(50), use_container_width=True)

st.divider()
st.caption("⚠️ 说明：本数据集仅包含皮马印第安女性样本，"
           "分析结论不可直接外推至其他人群。")

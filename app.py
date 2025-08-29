import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import PSReadline

# ==========================
# Load dataset
# ==========================
st.title("🌆 Smart City Energy Consumption Dashboard")
st.write("This app shows data analysis and predicts city energy consumption.")

df = pd.read_csv(
    'PJM_Load_hourly.csv',
    parse_dates=['Datetime'],
    index_col='Datetime'
)

df = df[['PJM_Load_MW']]
df.columns = ['energy_consumption']

# Feature engineering
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year
df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

# ==========================
# Data Analysis (EDA)
# ==========================
st.subheader("📊 Raw Data Sample")
st.dataframe(df.head())

st.subheader("📈 Energy Consumption over Time")
st.line_chart(df['energy_consumption'])

st.subheader("⚡ Average Consumption by Hour")
hourly_avg = df.groupby("hour")["energy_consumption"].mean()
st.line_chart(hourly_avg)

st.subheader("📅 Average Consumption by Month")
monthly_avg = df.groupby("month")["energy_consumption"].mean()
st.bar_chart(monthly_avg)

# ==========================
# Model Training
# ==========================
features = ['hour', 'dayofweek', 'month', 'year', 'is_weekend']
target = 'energy_consumption'

X = df[features]
y = df[target]

# Simple train/test split
split_point = int(len(X) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================
# Prediction Section
# ==========================
st.subheader("🔮 Predict Energy Consumption")

# User input
input_date = st.date_input("Select Date")
input_time = st.time_input("Select Time", value=datetime.now().time())
combined_datetime = datetime.combine(input_date, input_time)

# Extract features
new_data = {
    'hour': combined_datetime.hour,
    'dayofweek': combined_datetime.weekday(),
    'month': combined_datetime.month,
    'year': combined_datetime.year,
    'is_weekend': int(combined_datetime.weekday() in [5, 6])
}
new_df = pd.DataFrame([new_data])

if st.button("Predict"):
    prediction = model.predict(new_df[features])[0]
    st.success(f"Predicted Energy Consumption at {combined_datetime} → **{prediction:.2f} MW**")

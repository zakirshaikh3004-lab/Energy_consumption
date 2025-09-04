import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Use caching to load and preprocess the data only once.
# This prevents the app from reloading the CSV file and doing feature engineering on every rerun.
@st.cache_data
def load_data():
    """Loads the dataset and performs feature engineering."""
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
    
    return df

# Use caching for the model training, which is an expensive resource.
# This ensures the model is trained only once.
@st.cache_resource
def train_model(df):
    """Trains and returns the machine learning model."""
    features = ['hour', 'dayofweek', 'month', 'year', 'is_weekend']
    target = 'energy_consumption'
    
    X = df[features]
    y = df[target]
    
    # Simple train/test split
    split_point = int(len(X) * 0.8)
    X_train, y_train = X[:split_point], y[:split_point]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Main app logic
st.set_page_config(layout="wide", page_title="Smart City Dashboard")
st.title("🌆 Smart City Energy Consumption Dashboard")
st.write("This app shows data analysis and predicts city energy consumption.")

# Load the cached data
df = load_data()

# Train the cached model
model = train_model(df)


# --- Data Analysis (EDA) ---
st.subheader("📊 Raw Data Sample")
st.dataframe(df.head())

st.subheader("📈 Energy Consumption over Time")
st.line_chart(df['energy_consumption'])

st.subheader("⚡ Average Consumption by Hour")
# This calculation is fast, but we can still cache it if the dataset was larger
@st.cache_data
def get_hourly_avg(df):
    return df.groupby("hour")["energy_consumption"].mean()
    
st.line_chart(get_hourly_avg(df))

st.subheader("📅 Average Consumption by Month")
# This calculation is also fast, but can be cached for consistency
@st.cache_data
def get_monthly_avg(df):
    return df.groupby("month")["energy_consumption"].mean()

st.bar_chart(get_monthly_avg(df))


# --- Prediction Section ---
st.subheader("🔮 Predict Energy Consumption")

col1, col2 = st.columns(2)
with col1:
    input_date = st.date_input("Select Date")
with col2:
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
    features = ['hour', 'dayofweek', 'month', 'year', 'is_weekend']
    prediction = model.predict(new_df[features])[0]
    st.success(f"Predicted Energy Consumption at {combined_datetime.strftime('%Y-%m-%d %H:%M')} → **{prediction:.2f} MW**")

st.markdown("""
---
_**Note:** The application now runs much faster because the data loading and model training
steps are only performed once, thanks to caching with `@st.cache_data` and
`@st.cache_resource`._
""")
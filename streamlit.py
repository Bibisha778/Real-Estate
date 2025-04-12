import streamlit as st
import pandas as pd
import pickle
from src.data.make_dataset import load_and_preprocess_data
from src.models.train_model import train_RFmodel
from src.models.predict_model import evaluate_model
from src.models.visualization import plot_feature_importance
from src.features.build_features import build_features
from sklearn.preprocessing import MinMaxScaler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))


st.set_page_config(page_title="Real Estate Price Predictor", layout="centered")
st.title("🏠 Real Estate Price Predictor")

scaler = None

st.sidebar.header("📁 Load Data & Train Model")
if st.sidebar.button("Load and Train Model"):
    with st.spinner("Training the model..."):
        df = load_and_preprocess_data("final.csv")
        X = build_features(df.drop("price", axis=1))
        y = df["price"]
        model, scaler, X_test_scaled, y_test = train_RFmodel(X, y)
        rmse, r2 = evaluate_model(model, X_test_scaled, y_test)
        plot_feature_importance(model, X)
        st.success(f"Model trained. RMSE: {rmse:.2f}, R² Score: {r2:.2f}")

try:
    with open("models/RFmodel.pkl", "rb") as f:
        model = pickle.load(f)
    df = load_and_preprocess_data("final.csv")
    X = build_features(df.drop("price", axis=1))
    scaler = MinMaxScaler().fit(X)
except FileNotFoundError:
    model = None
    st.warning("Train the model first using the sidebar.")

st.subheader("📝 Property Details")
with st.form("input_form"):
    year_sold = st.number_input("Year Sold", min_value=2000, max_value=2025)
    property_tax = st.number_input("Property Tax", min_value=0.0)
    insurance = st.number_input("Insurance Cost", min_value=0.0)
    beds = st.number_input("Beds", min_value=0)
    baths = st.number_input("Baths", min_value=0)
    sqft = st.number_input("Square Feet", min_value=100)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025)
    lot_size = st.number_input("Lot Size", min_value=0.0)
    basement = st.selectbox("Basement", ["None", "Finished", "Unfinished"])
    property_type = st.selectbox("Property Type", ["House", "Condo", "Townhouse"])
    submit = st.form_submit_button("Predict Price")

if submit and model and scaler is not None:
    input_df = pd.DataFrame([{
        'year_sold': year_sold,
        'property_tax': property_tax,
        'insurance': insurance,
        'beds': beds,
        'baths': baths,
        'sqft': sqft,
        'year_built': year_built,
        'lot_size': lot_size,
        'basement': basement,
        'property_type': property_type
    }])

    full_df = load_and_preprocess_data("final.csv")
    full_df = pd.concat([full_df.drop(columns=["price"], errors='ignore'), input_df], ignore_index=True)
    encoded = build_features(full_df)
    user_encoded = encoded.tail(1)
    user_scaled = scaler.transform(user_encoded)

    prediction = model.predict(user_scaled)[0]

    st.subheader("💰 Predicted Price")
    st.success(f"Estimated Property Price: ${prediction:,.2f}")

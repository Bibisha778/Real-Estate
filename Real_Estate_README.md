# 🏠 Real Estate Price Predictor

This project predicts housing prices using a Random Forest Regressor. It provides a Streamlit interface where users can input property details and get real-time price predictions.

## 🚀 Features
- Predicts price based on attributes like size, rooms, lot size, basement type, etc.
- Built using Random Forest Regressor
- Interactive Streamlit interface
- Displays RMSE and R² Score
- Visualizes feature importance

## 📂 Project Structure
- `streamlit.py`: Main web app
- `main.py`: Command-line model trainer
- `models/`: Stores trained model
- `src/`: Contains preprocessing, training, and visualization scripts

## ▶️ Run Locally
```bash
streamlit run streamlit.py
```

## 🧰 Requirements
- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn
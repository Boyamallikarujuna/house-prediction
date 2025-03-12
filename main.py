import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit App Title
st.title("🏡 House Price Prediction")

# Sidebar - Upload Dataset
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Display Dataset
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Handle Missing Values
    if df.isnull().sum().sum() > 0:
        st.warning("Dataset contains missing values. They will be handled automatically.")
        df.fillna(df.median(numeric_only=True), inplace=True)

    # Encoding Categorical Features
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        st.write(f"Encoding categorical columns: {list(categorical_columns)}")
        for col in categorical_columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    # Sidebar - Feature Selection
    st.sidebar.subheader("Select Features for Model")
    selected_features = st.sidebar.multiselect("Choose Features", df.columns[:-1])
    target = st.sidebar.selectbox("Select Target Variable", df.columns)

    if selected_features and target:
        X = df[selected_features]
        y = df[target]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Sidebar - Model Selection
        model_choice = st.sidebar.selectbox("Choose Model", ["Linear Regression", "Random Forest"])

        if model_choice == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train Model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display Model Performance
        st.subheader("📊 Model Performance")
        st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

        # Prediction Section
        st.subheader("🏠 Predict House Price")
        user_inputs = {}
        for feature in selected_features:
            user_inputs[feature] = st.number_input(f"Enter value for {feature}", float(df[feature].min()),
                                                   float(df[feature].max()), float(df[feature].mean()))

        if st.button("Predict Price"):
            user_df = pd.DataFrame([user_inputs])
            user_df = scaler.transform(user_df)
            predicted_price = model.predict(user_df)
            st.success(f"Predicted House Price: ${predicted_price[0]:,.2f}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Data collection and pre-processing
df = pd.read_csv("Black_Friday.csv")
# Handle missing values
df.fillna(0, inplace=True)

# Engineering and feature selection
# Select relevant features for the model
selected_features = ['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Purchase']

# Keep only the selected features
df_selected = df[selected_features]

# Encode categorical variables
df_encoded = pd.get_dummies(df_selected)

# Split data into features (X) and target variable (y)
X = df_encoded.drop(columns=['Purchase'])
y = df_encoded['Purchase']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Model Deployment using Streamlit
def predict_sales(user_id, product_id, gender, age, occupation, city_category, stay_in_current_city, marital_status, product_category_1, product_category_2, product_category_3):
    input_data = pd.DataFrame({
        'User_ID': [user_id],
        'Product_ID': [product_id],
        'Gender_F': [1 if gender == 'F' else 0],
        'Gender_M': [1 if gender == 'M' else 0],
        'Age': [age],
        'Occupation': [occupation],
        'City_Category_A': [1 if city_category == 'A' else 0],
        'City_Category_B': [1 if city_category == 'B' else 0],
        'City_Category_C': [1 if city_category == 'C' else 0],
        'Stay_In_Current_City_Years': [stay_in_current_city],
        'Marital_Status': [marital_status],
        'Product_Category_1': [product_category_1],
        'Product_Category_2': [product_category_2],
        'Product_Category_3': [product_category_3]
    })
    return model.predict(input_data)[0]

# Streamlit UI
st.title("Black Friday Sales Prediction")
st.sidebar.title("Input Parameters")

user_id = st.sidebar.text_input("User ID", "")
product_id = st.sidebar.text_input("Product ID", "")
gender = st.sidebar.selectbox("Gender", ["M", "F"])
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
occupation = st.sidebar.number_input("Occupation", min_value=0, max_value=20, value=10)
city_category = st.sidebar.selectbox("City Category", ["A", "B", "C"])
stay_in_current_city = st.sidebar.number_input("Stay in Current City (Years)", min_value=0, max_value=20, value=5)
marital_status = st.sidebar.selectbox("Marital Status", [0, 1])
product_category_1 = st.sidebar.number_input("Product Category 1", min_value=1, max_value=20, value=1)
product_category_2 = st.sidebar.number_input("Product Category 2", min_value=0, max_value=20, value=0)
product_category_3 = st.sidebar.number_input("Product Category 3", min_value=0, max_value=20, value=0)

if st.sidebar.button("Predict Sales"):
    prediction = predict_sales(user_id, product_id, gender, age, occupation, city_category, stay_in_current_city, marital_status, product_category_1, product_category_2, product_category_3)
    st.write(f"Predicted Sales Amount: ${prediction:.2f}")

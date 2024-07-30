import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Load the data (you should replace this with your own dataset)
@st.cache_data
def load_data():
    data = pd.read_csv("housing.csv")
    return data

# Train the model
@st.cache_resource
def train_model(data):
    X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
    y = data['Price']
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Streamlit app
def main():
    st.title('House Price Prediction')
    
    # Load data and train model
    data = load_data()
    model, X_test, y_test = train_model(data)
    
    # Sidebar for user input
    st.sidebar.header('Enter House Details')
    AI = st.sidebar.slider('Average Area Income', float(data['Avg. Area Income'].min()), float(data['Avg. Area Income'].max()), float(data['Avg. Area Income'].mean()))
    AA = st.sidebar.slider('Average Area House Age', float(data['Avg. Area House Age'].min()), float(data['Avg. Area House Age'].max()), float(data['Avg. Area House Age'].mean()))
    AR = st.sidebar.slider('Average Area Number of Rooms', float(data['Avg. Area Number of Rooms'].min()), float(data['Avg. Area Number of Rooms'].max()), float(data['Avg. Area Number of Rooms'].mean()))
    AB = st.sidebar.slider('Average Area Number of Bedrooms', float(data['Avg. Area Number of Bedrooms'].min()), float(data['Avg. Area Number of Bedrooms'].max()), float(data['Avg. Area Number of Bedrooms'].mean()))
    AP = st.sidebar.slider('Area Population', float(data['Area Population'].min()), float(data['Area Population'].max()), float(data['Area Population'].mean()))

    # Make prediction
    if st.sidebar.button('Predict Price'):
        prediction = model.predict([[AI, AA, AR, AB, AP]])
        st.success(f'Predicted House Price: ${prediction[0]:.2f}k')
    
    # Display model performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.header('Model Performance')
    st.write(f'Mean Squared Error: {mse:.2f}')
    st.write(f'R-squared Score: {r2:.2f}')
    
    # Display scatter plot
    st.header('Actual vs Predicted Prices')
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted House Prices')
    st.pyplot(fig)

if __name__ == '__main__':
    main()
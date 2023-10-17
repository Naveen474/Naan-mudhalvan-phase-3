import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
file_path = "MSFT.csv"
data = pd.read_csv(file_path)

# Convert the 'Date' column to a datetime object
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data = data.sort_values(by='Date')

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Handle missing data (you can choose to fill or remove them)
# For simplicity, filling missing values with the previous day's value
data = data.fillna(method='ffill')

# Select features (X) and target variable (y)
X = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
y = data['Adj Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optionally, you can scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Regression Model: Support Vector Regressor (SVR)
svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)
svr_predictions = svr_model.predict(X_test_scaled)

# Evaluate the model
print("SVR RMSE:", mean_squared_error(y_test, svr_predictions, squared=False))

# Plotting for SVR model
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Prices')
plt.plot(svr_predictions, label='SVR Predictions')
plt.legend()
plt.title("SVR Model Predictions")
plt.show()
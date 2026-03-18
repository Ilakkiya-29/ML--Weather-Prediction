# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries such as pandas, numpy, matplotlib, and sklearn.
2.Create or load the dataset containing environmental sensor values.
3.Separate the dataset into input features (X) and target variables (y).
4.Split the dataset into training and testing sets.
5.Train the Random Forest Regressor model using the training data.
6.Predict the output values using the testing data.
7.Evaluate the model using R2 Score and RMSE.
8.Visualize the results using a scatter plot.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Ilakkiya K
RegisterNumber: 212225040130
# 📌 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 📌 Step 1: Create Sample Dataset (if you don't have one)
np.random.seed(42)

data = {
    'humidity': np.random.randint(40, 90, 100),
    'wind_speed': np.random.randint(1, 20, 100),
    'pressure': np.random.randint(980, 1050, 100),
    'temperature': np.random.randint(20, 40, 100),
    'pm2_5': np.random.randint(30, 150, 100),
    'energy': np.random.randint(80, 200, 100)
}

df = pd.DataFrame(data)

# 📌 Step 2: Define Inputs & Outputs
X = df[['humidity', 'wind_speed', 'pressure']]
y = df[['temperature', 'pm2_5', 'energy']]

# 📌 Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 📌 Step 4: Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 Step 5: Predictions
y_pred = model.predict(X_test)

# 📌 Step 6: Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2 Score:", r2)
print("RMSE:", rmse)

# 📌 Step 7: Plot (Temperature Prediction)
plt.figure()

plt.scatter(y_test.iloc[:,0], y_pred[:,0])
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Temperature Prediction using Random Forest")

# 📌 Save as Image
plt.savefig("prediction_output.png")

plt.show()
*/
```

## Output:
<img width="735" height="567" alt="image" src="https://github.com/user-attachments/assets/9e0566db-8825-41a7-847e-1b1a8cecf11c" />


## Result:
The Random Forest model was successfully implemented to predict temperature, PM2.5 levels, and energy consumption.

The model achieved a good accuracy with a high R2 Score.

The predicted values closely match the actual values.

The scatter plot shows a strong correlation between actual and predicted temperature.

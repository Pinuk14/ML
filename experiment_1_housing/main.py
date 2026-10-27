from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


df = pd.read_csv(Path(__file__).with_name("housing.csv"))
X = df[["RM", "LSTAT", "PTRATIO"]]
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Actual values:", list(y_test))
print("Predicted values:", y_pred.round(2))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))

new_house = pd.DataFrame(
    [[
        float(input("Enter average number of rooms (RM): ")),
        float(input("Enter lower status percentage (LSTAT): ")),
        float(input("Enter pupil-teacher ratio (PTRATIO): ")),
    ]],
    columns=["RM", "LSTAT", "PTRATIO"],
)
print("Predicted MEDV:", round(model.predict(new_house)[0], 2))
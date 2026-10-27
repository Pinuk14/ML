from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


df = pd.read_csv(Path(__file__).with_name("titanic.csv"))
sex_options = sorted(df["Sex"].dropna().unique())
embarked_options = sorted(df["Embarked"].dropna().unique())
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True, dtype=int)
df = df.dropna()

X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Actual values:", list(y_test))
print("Predicted values:", list(y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Did not survive", "Survived"],
    yticklabels=["Did not survive", "Survived"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Titanic Survival Confusion Matrix")
plt.tight_layout()
plt.show()

sex = input(f"Sex {sex_options}: ").strip().lower()
while sex not in sex_options:
    print("Choose one of the listed sex values.")
    sex = input(f"Sex {sex_options}: ").strip().lower()

embarked = input(f"Embarked {embarked_options}: ").strip().upper()
while embarked not in embarked_options:
    print("Choose one of the listed embarkation values.")
    embarked = input(f"Embarked {embarked_options}: ").strip().upper()

new_passenger = pd.DataFrame([{
    "Pclass": int(input("Passenger class (1-3): ")),
    "Age": float(input("Age: ")),
    "SibSp": int(input("Number of siblings/spouses aboard: ")),
    "Parch": int(input("Number of parents/children aboard: ")),
    "Fare": float(input("Fare: ")),
    "Sex": sex,
    "Embarked": embarked,
}])
new_passenger = pd.get_dummies(
    new_passenger, columns=["Sex", "Embarked"], dtype=int
).reindex(columns=X.columns, fill_value=0)
prediction = model.predict(new_passenger)[0]
print("Survival Prediction:", "Survived" if prediction == 1 else "Did not survive")
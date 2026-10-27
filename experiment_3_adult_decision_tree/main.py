from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def read_category(column):
    options = sorted(df[column].astype(str).unique())
    value = input(f"{column} {options}: ").strip()
    while value not in options:
        print("Choose one of the listed values.")
        value = input(f"{column} {options}: ").strip()
    return value


df = pd.read_csv(Path(__file__).with_name("adult.csv"))
raw_features = df.drop("income", axis=1)
X = pd.get_dummies(raw_features)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42, max_depth=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Actual values:", list(y_test))
print("Predicted values:", list(y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=model.classes_))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred, zero_division=0),
)

plt.figure(figsize=(15, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
)
plt.tight_layout()
plt.show()

new_adult = pd.DataFrame([{
    "age": int(input("Age: ")),
    "workclass": read_category("workclass"),
    "education": read_category("education"),
    "marital-status": read_category("marital-status"),
    "occupation": read_category("occupation"),
    "relationship": read_category("relationship"),
    "race": read_category("race"),
    "hours-per-week": int(input("Hours worked per week: ")),
}])
new_adult = pd.get_dummies(new_adult).reindex(columns=X.columns, fill_value=0)
print("Predicted income:", model.predict(new_adult)[0])
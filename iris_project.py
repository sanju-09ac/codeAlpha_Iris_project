import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Iris.csv")

# Features and target
X = data[['SepalLengthCm',
          'SepalWidthCm',
          'PetalLengthCm',
          'PetalWidthCm']]

y = data['Species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nPredictions:")
print(predictions)
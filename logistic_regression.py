import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load balanced training data
with open('data/processed_data/X_train_balanced.pickle', 'rb') as file:
    X_train = pickle.load(file)
with open('data/processed_data/y_train_balanced.pickle', 'rb') as file:
    y_train = pickle.load(file)

# Load test data
with open('data/processed_data/X_test.pickle', 'rb') as file:
    X_test = pickle.load(file)
test_data = pd.read_csv('data/processed_data/test_data_preprocessed.csv')
y_test = test_data['sentiment']

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model
with open('models/logistic_regression_model.pickle', 'wb') as file:
    pickle.dump(model, file)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
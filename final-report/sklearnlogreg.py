import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

data = pd.read_csv('final.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5033)

encoder = OneHotEncoder()
categorical_cols = ['key', 'mode'] # replace with the names of your categorical columns
X_train_encoded = pd.concat([X_train.drop(categorical_cols, axis=1), pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]).toarray())], axis=1)
X_test_encoded = pd.concat([X_test.drop(categorical_cols, axis=1), pd.DataFrame(encoder.transform(X_test[categorical_cols]).toarray())], axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion matrix:', confusion_matrix(y_test, y_pred))
print('Classification report:', classification_report(y_test, y_pred))


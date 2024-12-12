# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb
import joblib

# Load the dataset
print("Loading dataset...")
cropdf = pd.read_csv("Crop_recommendation.csv")
print("Dataset loaded successfully!")

# Basic exploration
print("\nBasic Dataset Information:")
print(cropdf.head())
print("Shape of the dataset:", cropdf.shape)
print("Columns:", cropdf.columns)
print("Are there missing values?", cropdf.isnull().any())
print("Number of various crops:", len(cropdf['label'].unique()))
print("List of crops:", cropdf['label'].unique())

# Visualization - Nitrogen (N) requirement
print("\nGenerating Nitrogen (N) requirement visualization...")
crop_summary = pd.pivot_table(cropdf, index=['label'], aggfunc='mean')
crop_summary_N = crop_summary.sort_values(by='N', ascending=False)

fig = make_subplots(rows=1, cols=2)

top = {'y': crop_summary_N['N'][0:10].sort_values().index, 'x': crop_summary_N['N'][0:10].sort_values()}
last = {'y': crop_summary_N['N'][-10:].index, 'x': crop_summary_N['N'][-10:]}

fig.add_trace(go.Bar(top, name="Most nitrogen required", marker_color='blue', orientation='h', text=top['x']), row=1, col=1)
fig.add_trace(go.Bar(last, name="Least nitrogen required", marker_color='red', orientation='h', text=last['x']), row=1, col=2)

fig.update_traces(texttemplate='%{text}', textposition='inside')
fig.update_layout(title_text="Nitrogen (N)", plot_bgcolor='white', font_size=12, font_color='black', height=500)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Correlation Heatmap
print("\nGenerating correlation heatmap...")
numeric_cropdf = cropdf.select_dtypes(include=['number'])
fig, ax = plt.subplots(1, 1, figsize=(15, 9))
sns.heatmap(numeric_cropdf.corr(), annot=True, cmap='Wistia')
ax.set(xlabel='features', ylabel='features')
plt.title('Correlation between different features', fontsize=15, c='black')
plt.show()

# Splitting the dataset
print("\nSplitting the dataset into training and testing sets...")
X = cropdf.drop('label', axis=1)
y = cropdf['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)
print("Data split completed!")

# Train the LightGBM model
print("\nTraining the LightGBM model...")
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
print("Model training completed!")

# Predict the results
print("\nMaking predictions on the test set...")
y_pred = model.predict(X_test)

# Evaluate the model
print("\nEvaluating the model...")
accuracy = accuracy_score(y_test, y_pred)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy))

# Training set accuracy
y_pred_train = model.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))

# Model scores
print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))

# Confusion Matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix - score: {:.4f}'.format(accuracy), size=15)
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
print("\nSaving the trained model...")
joblib.dump(model, 'model.pkl')
print("Model saved successfully as 'model.pkl'!")

# Example Prediction
print("\nExample prediction:")
example_input = [[90, 42, 43, 20.879744, 75, 5.5, 220]]
newdata = model.predict(example_input)
print("Prediction for new data:", newdata)

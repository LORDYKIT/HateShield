import tkinter as tk
from tkinter import ttk
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
data = pd.read_csv('aggressionCleaned.csv')

# Preprocessing: Splitting data into features (X) and target (y)
X = data['Text']  # Features (text content)
y = data['oh_label']  # Target (0 for unhateful, 1 for hateful)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction: Convert text data into numerical features using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Apply random oversampling to the training set
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_train_vectors, y_train)

# Initialize Linear SVM classifier
svm_classifier = LinearSVC()

# Train the classifier
svm_classifier.fit(X_resampled, y_resampled)



# Function to predict the label (hateful or not hateful) for a given text input
def predict(text):
    text_vector = vectorizer.transform([text])
    prediction = svm_classifier.predict(text_vector)
    return "Hateful" if prediction[0] == 1 else "Not Hateful"

# Function to handle button click event
def on_submit():
    text = text_entry.get()
    prediction = predict(text)
    result_label.config(text="Prediction: " + prediction)

# Create main window
root = tk.Tk()
root.title("SVM Hate Speech Detection")

# Create text entry widget
text_entry = ttk.Entry(root, width=40)
text_entry.grid(row=0, column=0, padx=10, pady=10)

# Create submit button
submit_button = ttk.Button(root, text="Submit", command=on_submit)
submit_button.grid(row=0, column=1, padx=10, pady=10)

# Create label to display result
result_label = ttk.Label(root, text="")
result_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Run the application
root.mainloop()

# Calculate performance metrics on the test set
y_pred = svm_classifier.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Calculate performance metrics on the training set
y_train_pred = svm_classifier.predict(X_resampled)
train_accuracy = accuracy_score(y_resampled, y_train_pred)


print("\nTraining Data:")
print("Accuracy:", train_accuracy)


metrics = ['Accuracy', 'F1-score', 'Precision', 'Recall']
accuracies = [accuracy, f1, precision, recall]

# Plot the accuracies for different metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics, accuracies, color='b', width=0.4, alpha=0.5)
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Performance Metrics on Test Set')
plt.show()

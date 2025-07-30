
"""# Task
Build an email spam detector using machine learning in Python. The detector should be trained to classify emails as either spam or non-spam (ham) using the dataset provided in "/content/spam.csv". The process should include data loading, preprocessing, splitting into training and testing sets, text vectorization using TF-IDF, model selection (Multinomial Naive Bayes), training, evaluation, and prediction on new data.

## Data loading

### Subtask:
Load the data from the `/content/spam.csv` file into a DataFrame and display the first 5 rows and the columns and their types to discern how to proceed.

**Reasoning**:
Load the data from the CSV file into a pandas DataFrame and display the head and info of the DataFrame to understand its structure and data types.
"""

import pandas as pd

df = pd.read_csv('/content/spam.csv', encoding='latin-1')
display(df.head())
display(df.info())

"""## Data preprocessing

### Subtask:
The dataset contains two columns, `v1` and `v2`. `v1` indicates whether the email is spam or ham, and `v2` contains the email message. Rename the columns to `label` and `message` respectively, and convert the `label` column into numerical values (0 for ham and 1 for spam).

**Reasoning**:
Rename the columns and convert the label column to numerical values as per the instructions. Then display the head of the modified dataframe.
"""

df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
display(df.head())

"""## Data splitting

### Subtask:
Split the data into training and testing sets to prepare for model training and evaluation.

**Reasoning**:
Split the data into training and testing sets using train_test_split.
"""

from sklearn.model_selection import train_test_split

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

"""## Text vectorization

### Subtask:
Convert the text data into numerical representations that can be used as input for the machine learning model. Use TF-IDF vectorization.

**Reasoning**:
Convert the text data into numerical representations using TF-IDF vectorization for both the training and testing sets and print their shapes.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f'Shape of X_train_tfidf: {X_train_tfidf.shape}')
print(f'Shape of X_test_tfidf: {X_test_tfidf.shape}')

"""## Model selection and training

### Subtask:
Choose a suitable classification model (e.g., Multinomial Naive Bayes) and train it on the training data.

**Reasoning**:
Import the Multinomial Naive Bayes model, instantiate it, and train it using the TF-IDF transformed training data and labels.
"""

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

"""## Model evaluation

### Subtask:
Evaluate the performance of the trained model on the testing data using appropriate metrics (e.g., accuracy, precision, recall, F1-score).

**Reasoning**:
Evaluate the trained model's performance on the test set using accuracy, precision, recall, and F1-score.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

"""## Model prediction

### Subtask:
Use the trained model to predict whether a new email is spam or not.

**Reasoning**:
Create example email messages, transform them using the fitted TF-IDF vectorizer, predict their labels using the trained model, and print the results.
"""

# 1. Create example email messages
new_emails = [
    "Congratulations! You've won a free iPhone! Click here to claim your prize.",
    "Hey, just checking in to see if you're free for coffee tomorrow.",
    "Urgent: Your account has been compromised. Verify your details immediately.",
    "Meeting reminder for 2 PM today."
]

# 2. Transform the new email messages using the fitted TF-IDF vectorizer
new_emails_tfidf = tfidf_vectorizer.transform(new_emails)

# 3. Use the trained model to predict the class label
predicted_labels = model.predict(new_emails_tfidf)

# 4. Print the original email messages and their corresponding predicted labels
label_map = {0: 'ham', 1: 'spam'}
for i, email in enumerate(new_emails):
    predicted_label = label_map[predicted_labels[i]]
    print(f"Email: '{email}'")
    print(f"Predicted Label: {predicted_label}")
    print("-" * 30)

"""## Summary:

### Data Analysis Key Findings

*   The dataset contains 5572 email entries with two main relevant columns: `v1` (label) and `v2` (message).
*   The data was split into training (4457 entries) and testing (1115 entries) sets.
*   TF-IDF vectorization transformed the text data into numerical matrices of shape (4457, 7735) for training and (1115, 7735) for testing, based on 7735 unique terms.
*   The trained Multinomial Naive Bayes model achieved an accuracy of 0.9623 on the test set.
*   The model demonstrated perfect precision (1.0000) on the test set, meaning no ham emails were misclassified as spam.
*   The recall on the test set was 0.7200, indicating that 72% of actual spam emails were correctly identified.
*   The F1-score, a harmonic mean of precision and recall, was 0.8372.
*   The model successfully predicted the labels for new emails, correctly identifying a promotional message as spam and conversational messages as ham.

### Insights or Next Steps

*   The model shows strong performance, particularly in avoiding false positives (classifying ham as spam). Further investigation into the 28% of missed spam emails (false negatives) could involve analyzing the characteristics of these specific messages.
*   Exploring other text preprocessing techniques (e.g., stemming, lemmatization, removing stop words) or different machine learning models (e.g., Support Vector Machines, Logistic Regression) could potentially improve recall while maintaining high precision.

"""

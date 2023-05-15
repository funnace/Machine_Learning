import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the SMS Spam dataset
df = pd.read_csv('C:\\Users\\System_Name\\Downloads\\sms_spam.csv', encoding='latin-1')

# Preprocessing
df.columns = ['label', 'message']
df['label'] = df.label.map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], random_state=1)
count_vector = CountVectorizer()
X_train_matrix = count_vector.fit_transform(X_train)
nb = MultinomialNB()
nb.fit(X_train_matrix, y_train)
X_test_matrix = count_vector.transform(X_test)
y_pred = nb.predict(X_test_matrix)

confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Confusion Matrix:\n', confusion_matrix)
print('Accuracy Score:', accuracy)
print('Precision Score:', precision)
print('Recall Score:', recall)
print('F1-Score:', f1)

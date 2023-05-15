import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#Data Shape
df = pd.read_csv('C:\\Users\\YASH\\Downloads\\Heart.csv')
print("Shape of Data: ",df.shape,"\n\n")

#Missing Values
missing_values = df.isnull().sum()
print('Missing values:\n', missing_values,"\n\n")

#Datatype of ecah column
data_types = df.dtypes
print('Datatypes of each column:\n', data_types,"\n\n")

# Randomly divide dataset in training (75%) and testing (25%)
train, test = train_test_split(df, test_size=0.25,random_state=36)
print('Training set:\n', train,"\n\n")
print('Testing set:\n', test,"\n\n")


# Creating the confusion matrix
y_true = [1]*50 + [0]*450
y_pred = [1]*100 + [0]*400
c_matrix = confusion_matrix(y_true, y_pred)
print('Confusion matrix:\n', c_matrix)

# Calculate accuracy, precision, recall, and F1 score
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print('Accuracy:', acc)
print('Precision:', prec)
print('Recall:', rec)
print('F1 score:', f1)
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import auc
from sklearn import metrics
import sklearn

# load the dataset
data = pd.read_csv('otu.csv')
data = data.T

# convert left/right to 0/1
le = LabelEncoder()
data[0] = le.fit_transform(data[0])
print(data.head(5))

# split dataset into feature target
X = data.iloc[:, 1:3033]
y = data.iloc[:, 0]

# training dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0)

# train Gaussian Nsive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# use cross validation
scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=5)
print('Cross-validated scores:', scores)

# accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print('Accuracy:', accuracy)

# Area under the ROC curve.
roc_auc = roc_auc_score(y_test, predictions)
print('Auc:', roc_auc)

# confusion matrix
confusion = confusion_matrix(y_test, predictions)
print('Confusion matrix: ')
print(confusion)

#sensivity and specificity
sensitivity = confusion[1, 1] / (confusion[1, 0] + confusion[1, 1])
specificity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
print('   Sensitivity:', sensitivity)
print('   Specificity:', specificity)

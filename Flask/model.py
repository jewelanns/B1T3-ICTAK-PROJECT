# Importing necessary libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Loading dataset
data = pd.read_excel('Satisfaction_Pro.xlsx')
### Splitting into target and features
X = data.drop(['satisfaction','Gender','Gate location','Departure Delay in Minutes'], axis=1)
y = data['satisfaction']
# split into train and test set

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# model training Random Forest
rf_clf = RandomForestClassifier()
rf_clf = rf_clf.fit(X_train,y_train)
y_pred_rf = rf_clf.predict(X_test)
# finding accuracy
rf_acc_ws = accuracy_score(y_test, y_pred_rf)
# Finding accuracy
print("Accuracy score", rf_acc_ws)
#Saving model using pickle
pickle.dump(rf_clf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load( open('model.pkl','rb'))

print(model.predict([[0,56,1,0,1,3,4,5,2,3,4,3,3,3,3,3,3,4,0]]))
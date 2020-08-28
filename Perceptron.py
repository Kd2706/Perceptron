import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Perceptron_model import MyPerceptron

data = pd.read_csv('student_records.csv')

x = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=1)

clf = MyPerceptron()

clf.fit(xtrain, ytrain)

y_pred = clf.predict(xtest)

print(accuracy_score(ytest, y_pred))

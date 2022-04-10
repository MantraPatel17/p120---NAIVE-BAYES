import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from sklearn import datasets

wine = datasets.load_wine()

# print the names of the 13 features
print("Features: ", wine.feature_names)

# print the label type of wine(class_0, class_1, class_2)
print("Labels: ", wine.target_names)


X = wine.data
Y = wine.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=109)

naiveBayes = GaussianNB()

naiveBayes.fit(X_train,Y_train)

y_pred = naiveBayes.predict(X_test)

print('Accruacy with naive_bayes', accuracy_score(Y_test,y_pred))


import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
import numpy as np

# Prepare the data for different classifiers
# Delete the column releadse_date, name, artists, id
csv_data = pd.read_csv("processed_data_full.csv",usecols=[0,2,3,4,5,7,8,9,10,11,13,15,16,17,18,19,20])

input_label = csv_data.iloc[:,:-2]
result_label = csv_data.iloc[:,-2:-1]

result_list = result_label.values.tolist()
result_dict = {}

# normalize the data

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

values = 0
for labels in result_list:
    if labels[0] not in result_dict.keys():
        result_dict[labels[0]] = values
        values += 1

label_list =[]
for x in result_list:
    label_list.append(result_dict[x[0]])

x = normalization(input_label)
y = DataFrame(label_list).values.ravel()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# KNN classfier

from sklearn.neighbors import KNeighborsClassifier
def KNN(X,y,x_test, y_test):
    model = KNeighborsClassifier(n_neighbors=10)#默认为5
    model.fit(X,y)
    predict = model.score(x_test,y_test)
    print("KNN Accuracy {:.2f}".format(predict))
    return predict


# SVM classifer
from sklearn.svm import SVC
def SVM(X,y,x_test, y_test):
    model = SVC(C=5.0)
    model.fit(X,y)
    predicted = model.score(x_test,y_test)
    print("SVM Accuracy {:.2f}".format(predicted))
    return predicted


# Linear Regression
from sklearn.linear_model import LogisticRegression
def LR(X,y,x_test, y_test):
    model = LogisticRegression(C=1000.0, random_state=0)
    model.fit(X,y)
    predicted = model.score(x_test, y_test)
    print("LR Accuracy {:.2f}".format(predicted))
    return predicted


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
def DecisionTree(X,y,x_test, y_test):
    model = DecisionTreeClassifier()
    model.fit(X,y)
    predicted = model.score(x_test, y_test)
    print("Decision Tree Accuracy {:.2f}".format(predicted))
    return predicted


# Random Forsest
from sklearn.ensemble import RandomForestClassifier
def RandomForest(X,y,x_test, y_test):
    model = RandomForestClassifier()
    model.fit(X,y)
    predicted = model.score(x_test, y_test)
    print("Random Forest Accuracy {:.2f}".format(predicted))
    return predicted

# bayes (Gaussion, Multinomial and BernouliNB)
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
def GNB(X,y,x_test, y_test):
    model =GaussianNB()
    model.fit(X,y)
    predicted = model.score(x_test, y_test)
    print("Gaussian bayes Accuracy {:.2f}".format(predicted))
    return predicted

# def MNB(X,y,x_test, y_test):
#     model = MultinomialNB()
#     model.fit(X,y)
#     predicted = model.score(x_test, y_test)
#     print("Multinomial bayes Accuracy {:.2f}".format(predicted))
#     return predicted

def BNB(X,y,x_test, y_test):
    model = BernoulliNB()
    model.fit(X,y)
    predicted = model.score(x_test, y_test)
    print("Bernoulli bayes Accuracy {:.2f}".format(predicted))
    return predicted


# see the result
KNN(x_train,y_train,x_test,y_test)
SVM(x_train,y_train,x_test,y_test)
#LR(x_train,y_train,x_test,y_test)
DecisionTree(x_train,y_train,x_test,y_test)
RandomForest(x_train,y_train,x_test,y_test)
GNB(x_train,y_train,x_test,y_test)
# MNB(x_train,y_train,x_test,y_test)  have negative value, can't use
BNB(x_train,y_train,x_test,y_test)

print()
print(result_dict)
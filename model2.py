import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
import pickle
from xgboost import XGBClassifier
from xgboost import plot_importance


def decisionTree(X_train, y_train):
    '''This is Decision Tree Method'''
    # Decision Tree Model
    tree = DecisionTreeClassifier(min_samples_leaf=17)
    t= tree.fit(X_train, y_train)
    #print("Decisioin Tree Model Score" , ":" , t.score(X_train, y_train) , "," , 
          #"Cross Validation Score" ,":" , t.score(X_test, y_test))
    file = open('pkl/dTree.pkl','wb')
    pickle.dump(t,file)
    file.close()

def randForest(X_train, y_train):
    '''This is Random Forest Method'''
    forest = RandomForestClassifier(n_estimators=36, min_samples_leaf=2)
    f = forest.fit(X_train, y_train)
    file = open('pkl/rForest.pkl','wb')
    pickle.dump(f,file)
    file.close()

def supportVectorClustering(X_train, y_train):
    '''This is Support Vector Clustering Method'''
    svc = SVC()
    s= svc.fit(X_train, y_train)
    file = open('pkl/svm.pkl','wb')
    pickle.dump(s,file)
    file.close()

def logisticRegression(X_train, y_train):
    lr = LogisticRegression(multi_class='multinomial', solver='newton-cg',fit_intercept=True)
    lr = lr.fit(X_train,y_train)
    file = open('pkl/logiReg.pkl','wb')
    pickle.dump(lr,file)
    file.close()    

def AdaBoost(X_train, y_train):
    ada = AdaBoostClassifier(n_estimators=2)
    af = ada.fit(X_train, y_train)
    file = open('pkl/Adaboost.pkl','wb')
    pickle.dump(af,file)
    file.close()

def xboostAlgo(X_train, y_train):
    model = XGBClassifier()
    model = XGBClassifier(learning_rate=0.1,n_estimators=100)
    mf = model.fit(X_train,y_train)
    file = open('pkl/xBoost.pkl','wb')
    pickle.dump(mf,file)
    file.close()

if __name__ == "__main__":
    data = pd.read_csv('data/students.csv')

    #Lable the Target Veriable
    data['FinalGrade'] = 'na'
    data.loc[(data.G3 >= 18) & (data.G3 <= 20), 'FinalGrade'] = 'Excellent'
    data.loc[(data.G3 >= 15) & (data.G3 <= 17), 'FinalGrade'] = 'Good' 
    data.loc[(data.G3 >= 11) & (data.G3 <= 14), 'FinalGrade'] = 'Satisfactory' 
    data.loc[(data.G3 >= 6) & (data.G3 <= 10), 'FinalGrade'] = 'Poor' 
    data.loc[(data.G3 >= 0) & (data.G3 <= 5), 'FinalGrade'] = 'Failure'

    # label encode final_grade
    le = preprocessing.LabelEncoder()
    data.FinalGrade = le.fit_transform(data.FinalGrade)

    data.sex = le.fit_transform(data.sex)
    data.school = le.fit_transform(data.school)
    data.address = le.fit_transform(data.address)
    data.famsize = le.fit_transform(data.famsize)
    data.Pstatus = le.fit_transform(data.Pstatus)
    data.Mjob = le.fit_transform(data.Mjob)

    data.Fjob = le.fit_transform(data.Fjob)
    data.reason = le.fit_transform(data.reason)
    data.guardian = le.fit_transform(data.guardian)
    data.schoolsup = le.fit_transform(data.schoolsup)
    data.famsup = le.fit_transform(data.famsup)

    data.paid = le.fit_transform(data.paid)
    data.activities = le.fit_transform(data.activities)
    data.nursery = le.fit_transform(data.nursery)
    data.higher = le.fit_transform(data.higher)
    data.internet = le.fit_transform(data.internet)

    data.romantic = le.fit_transform(data.romantic)
    data.subject = le.fit_transform(data.subject)
    
    #Label and Target veriable seperated
    X = data.drop(labels=['FinalGrade','G3'],axis=1)
    y = data.FinalGrade

    #data slipt in train and test cases
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

    #Function Calling
    decisionTree(X_train,y_train)
    randForest(X_train,y_train)
    supportVectorClustering(X_train,y_train)
    logisticRegression(X_train,y_train)
    AdaBoost(X_train,y_train)
    xboostAlgo(X_train,y_train)



    #reg = linear_model.LinearRegression()

    #reg.fit(x_train, y_train)

    #file = open('pkl/model1.pkl','wb')
    #pickle.dump(reg, file)
    #file.close()

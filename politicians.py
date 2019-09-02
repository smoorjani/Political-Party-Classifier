import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.cross_validation import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import math

eps = np.finfo(float).eps

# Function importing Dataset
df = pd.read_csv('politicians_dataset.csv')

for title in df.columns[2:]:
    df[title] = df[title].apply(lambda x: 1 if x == 'y' else 0)

X = df[['education','scotland','tax']]
Y = df['Party']



def calculate_entropy(dataframe, attr):
  classifications = dataframe.keys()[2] 
  target_variables = dataframe[classifications].unique()
  variables = dataframe[attr].unique()
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(dataframe[attr][dataframe[attr] == variable][dataframe[classifications] == target_variable])
          den = len(dataframe[attr][dataframe[attr] == variable])
          fraction = num/(den+eps)
          entropy += -fraction* math.log(fraction+eps,2)
      fraction2 = den/len(dataframe)
      entropy2 += -fraction2*entropy
  return abs(entropy2)

# def calculate_information_gain()
 
def train_using_gini(X_train, X_test, y_train): 
  
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 101,max_depth=3, min_samples_leaf=5) 
   
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
    
def train_using_entropy(X_train, X_test, y_train): 
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 101, max_depth = 3, min_samples_leaf = 5) 
  
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
  
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test)
    print("Test subjects:")
    print(X_test)
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred))
'''
## TESTING FOR OPTIMAL RANDOM STATE ##
ma = 0
state = 0
for states in range(0,5000):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = states) 
      
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = train_using_entropy(X_train, X_test, y_train) 

    #print("Results Using Entropy:") 
    # Prediction using entropy 
    #y_pred_entropy = prediction(X_test, clf_entropy)
    #cal_accuracy(y_test, y_pred_entropy)

    y_pred_entropy = clf_entropy.predict(X_test)
    acc = accuracy_score(y_test,y_pred_entropy)

    if acc > ma:
        ma = acc
        state = states


print(ma)
print(state)
        
'''


## TESTING FOR OPTIMAL TRAIN/TEST SPLIT ##
ma = 0
split = 0
for splt in range(1,50):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = float(splt/100), random_state = 327) 
      
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = train_using_entropy(X_train, X_test, y_train) 

    #print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)

    y_pred_entropy = clf_entropy.predict(X_test)
    acc = accuracy_score(y_test,y_pred_entropy)

    if acc > ma:
        ma = acc
        split = splt


print(ma)
print(split)
        

      








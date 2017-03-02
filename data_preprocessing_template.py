# Data Preprocessing

# Importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# You need to set the directory where the data.csv is:
        # You need to make sure that the .py file will be save on the same place
        # when this is settle you can press F5 to run the file. 

# We need to create a variable which contains the dataset
dataset = pd.read_csv('Data.csv')

# Define the indipendent variable X
X =  dataset.iloc[:, :-1].values

# Define the dependent variable Y
y = dataset.iloc[:, 3].values
                
# Taking care of missing values
# Import the class

# Library 'sklearn.preprocessing'
# Class 'Imputer'

from sklearn.preprocessing import Imputer
# ctrl+i to get info about an object
imputer = Imputer(missing_values="NaN", strategy="mean", axis= 0)

# Impute this imputer on the feature X
# Indexing in Python starts from 0, but the upper bound is excluded and take 
#the index as it was starting from 1
    
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
# create the first object of the LabelEncoder class
labelencoder_X = LabelEncoder()

# take a method of the LabelEncoder class
labelencoder_X.fit_transform(X[:, 0])

# we obtain the encoded value of these country
# Transfor the first column of the arry X, we need to incorporare the first method into a variable
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# country since it does not have order relashionship between the observations
# it needs to be encoded as a dummy variable

from sklearn.preprocessing import OneHotEncoder

# create OneHotEncoder object

onehotencoder = OneHotEncoder(categorical_features = [0])

# we defines a variables because we want to change X, then we use the object
# and we fit it into the column country (we specified before in the object definition the column 0)

X = onehotencoder.fit_transform(X).toarray()

# To encoding the dependent variable we only need to use LabelEncoder
labelencoder_y = LabelEncoder()
labelencoder_y.fit_transform(y)
y = labelencoder_y.fit_transform(y)

# Splitting the dataset in the Training set and the Test set
from sklearn.cross_validation import train_test_split
# Define the variables
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# Feature Scaling (ref. Euclidean Distance)
      #library                       #class
from sklearn.preprocessing import StandardScaler
# object
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Scaling the dummy variables you lose the interpretation of your model
# in this case since we do not need to interprete the model we scale also the dummies

# Classification problem with categorical dependent variable does not need to be scale

# Evan if decision trees the euclidean distance does not apply we scale so the 
# logarith get faster
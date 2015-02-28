#Check out: http://www.slideshare.net/SarahGuido/a-beginners-guide-to-machine-learning-with-scikitlearn
#And: http://nbviewer.ipython.org/gist/sarguido/7423289

import csv

import numpy
import scipy
#from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


#From: http://mldata.org/repository/data/viewslug/datasets-uci-breast-cancer/
#filename = "datasets-uci-breast-cancer.csv"
#From: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
filename = "wdbc.csv"


#Labels for the outputs
output_labels = ["B", "M"]

data = list(csv.reader(open(filename, 'rU' )))
data =  scipy.array(data)
X = data[0:,2:32]
y = data[0:, 1]


#Check all data has been successfully loaded (array shape):
print 'X dimensions: ', X.shape
print 'y dimensions: ', y.shape

#Show frequencies of y:
scipy.stats.itemfreq(y)

#Plot y frequencies
yFreq = scipy.stats.itemfreq(y)
yFreqs = [, yFreq[1][1].astype(int)]
plt.bar(0, yFreq[0][1].astype(int), label=yFreq[0][0])
plt.bar(1.5, yFreq[1][1].astype(int), label=yFreq[1][0])
plt.xticks(visible=False)
plt.legend()
plt.show()

#Convert y categories into binary features:
le = preprocessing.LabelEncoder()
#Fit numerical values to the categorical data.
le.fit(y)
#Transform the labels into numerical correspondents.
transformed_y = le.transform(y)


#Make sure X variables arre floats:
X = X.astype(numpy.float)

#Plot unnormalised data in the first column in X.
plt.hist(X[:, 1])
plt.show()

#Normalise and standardise the variables in X:

#Normalise data:
X_normalised = preprocessing.normalize(X, norm="l2")
#Plot normalised:
plt.hist(X_normalised[:, 1])
plt.show()
plt.hist(X_normalised[:, 2])
plt.show()
#L1 (l1) normalisation is least absolute deviations; L2 (l2, as above) normalisation is least squares.


#Standardise data:
X_standardised = preprocessing.scale(X)
#Plot standardised:
plt.hist(X_standardised[:, 1])
plt.show()
plt.hist(X_standardised[:, 2])
plt.show()

#Standardisation with scale(function) - mean 0 sd 1.
#Assumed by many algorithms so best practice to do it.


#Split into training and test data:
X_train, X_test, y_train, y_test = train_test_split(X, y)
#Check:
print 'X_train dimensions: ', X_train.shape
print 'y_train dimensions: ', y_train.shape
print 'X_test dimensions: ', X_test.shape
print 'y_test dimensions: ', y_test.shape


#Plot heatmap of correlations between different features:
correlation_matrix = numpy.corrcoef(X, rowvar=0)
fig, ax = plt.subplots()
heatmap = ax.pcolor(correlation_matrix, cmap=plt.cm.Blues)
plt.show()






############################################################

#Convert to array of floats (currently text):
#X = X.astype(numpy.float)

#Point out difference between scipy array and normal python 2D array.
#See http://pages.physics.cornell.edu/~myers/teaching/ComputationalMethods/python/arrays.html for details on scipy arrays

#Convert the labels into numerical values:
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y)
transformed_y = label_encoder.transform(y)

#Check label conversion:
print 'Converted: ', y[0]

#Normalise data:
X_normalised = preprocessing.normalize(X, norm="l2")

#Standardise data:
X_standardised = preprocessing.scale(X)


#Split the datset into training and validation sets:
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Check size of split:
print 'Training set size: ', len(X_train)
print 'Test set size: ', len(X_test)

#Plot heatmap of correlations between different features:
correlation_matrix = numpy.corrcoef(X);
fig, ax = plt.subplots()
heatmap = ax.pcolor(X, cmap=plt.cm.Blues)
plt.show()


######################################################################
##Create a DictVectorizer object.
#vectorizer = DictVectorizer()
##Transform the categorical data into zeros and ones and put it into an array (1 for when a case is true, 0 for when it is not)
#X = vectorizer.fit_transform(X).toarray()

##Check vectorisation:
#print 'Vectorised data: ', X[0];
#print 'Unvectorised data: ', vectorizer.inverse_transform(X[0]);



#Normalisation for Vector Space Model (clustering, text), norm=1, all values between 0 and 1.
#L1 (l1) normalisation is least absolute deviations; L2 (l2, as above) normalisation is least squares.
#See http://www.chioka.in/differences-between-the-l1-norm-and-the-l2-norm-least-absolute-deviations-and-least-squares/

#Plot normalised data:


#Standardise data
X_standardised = preprocessing.scale(X)


#Standardisation with scale(function) - mean 0 sd 1.
#Assumed by many algorithms so best practice to do it.

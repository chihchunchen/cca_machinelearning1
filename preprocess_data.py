# See http://nbviewer.ipython.org/gist/sarguido/7423289 

import csv

import numpy
import scipy
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


filename = "wdbc.data"


#Labels for the outputs
output_labels = ["M", "B"]

data = list(csv.reader(open(filename, 'rU' )))
data =  scipy.array(data)
X = data[0:,2:31]
#Convert to array of floats (currently text):
X = X.astype(numpy.float)

y = data[0:, 1]
#Point out difference between scipy array and normal python 2D array.
#See http://pages.physics.cornell.edu/~myers/teaching/ComputationalMethods/python/arrays.html for details on scipy arrays


#Check all data has been successfully loaded (array size):
print '# records in input: ', len(X)
print '# columns in input: ', len(X[0])
print 'input array size: ', X.size
print '# records in output: ', len(y)
print 'output array size: ', y.size

#Convert the labels into numerical values:
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y)
y = label_encoder.transform(y)

#Check label conversion:
print 'Converted: ', y[0]

#Split the datset into training and validation sets:
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Check size of split:
print 'Training set size: ', len(X_train)
print 'Test set size: ', len(X_test)

#Plot heatmap of correlations between different features:
correlation_matrix = numpy.corrcoef(X, rowvar=0);
fig, ax = plt.subplots()
heatmap = ax.pcolor(correlation_matrix, cmap=plt.cm.Blues)
plt.show()


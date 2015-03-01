#Check out: http://www.slideshare.net/SarahGuido/a-beginners-guide-to-machine-learning-with-scikitlearn
import csv
import numpy
import scipy
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

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
plt.bar(0.2, yFreq[0][1].astype(int), label=yFreq[0][0], color='red')
plt.bar(1.2, yFreq[1][1].astype(int), label=yFreq[1][0], color='blue')
plt.xticks(visible=False)
plt.legend()
plt.show()

#Convert y categories into binary features:
le = preprocessing.LabelEncoder()
#Fit numerical values to the categorical data.
le.fit(y)
#Transform the labels into numerical correspondents.
transformed_y = le.transform(y)

#Cast X variables into floats (they are parsed as strings):
X = X.astype(numpy.float)

#Plot unnormalised data in the first column in X.
plt.hist(X[:, 1])
plt.show()

#Normalise variables in X:
X_normalised = preprocessing.normalize(X, norm="l2")
#Plot normalised:
plt.hist(X_normalised[:, 1])
plt.show()
plt.hist(X_normalised[:, 2])
plt.show()
#L1 (l1) normalisation is least absolute deviations; L2 (l2, as above) normalisation is least squares.

#Standardise variables in X:
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


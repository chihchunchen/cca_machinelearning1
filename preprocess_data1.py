#Check out: http://www.slideshare.net/SarahGuido/a-beginners-guide-to-machine-learning-with-scikitlearn
import csv
import numpy
import scipy
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

#From: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
fileName = "wdbc.csv"
fileOpen = open(fileName, "rU")
csvData = csv.reader(fileOpen)
dataList = list(csvData)


#Labels for the outputs
output_labels = ["B", "M"]


dataArray =  scipy.array(dataList)
X = dataArray[0:,2:32]
y = dataArray[0:, 1]


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
yTransformed = le.transform(y)

#Cast X variables into floats (they are parsed as strings):
X = X.astype(numpy.float)

#Plot unnormalised data in the first column in X.
plt.hist(X[:, 1])
plt.show()

#Normalise variables in X:
XNormalised = preprocessing.normalize(X, axis = 0, norm="l2")
#Plot normalised:
plt.hist(XNormalised[:, 1])
plt.show()
plt.hist(XNormalised[:, 2])
plt.show()
#L1 (l1) normalisation is least absolute deviations; L2 (l2, as above) normalisation is least squares.

#Standardise variables in X:
XStandardised = preprocessing.scale(X, axis = 0)
#Plot standardised:
plt.hist(XStandardised[:, 1])
plt.show()
plt.hist(XStandardised[:, 2])
plt.show()
#Standardisation with scale(function) - mean 0 sd 1.
#Assumed by many algorithms so best practice to do it.

#Split into training and test data:
XTrain, XTest, yTrain, yTest = train_test_split(X, y)
#Check:
print 'X train dimensions: ', XTrain.shape
print 'y train dimensions: ', yTrain.shape
print 'X test dimensions: ', XTest.shape
print 'y test dimensions: ', yTest.shape


#Plot heatmap of correlations between different features:
correlationMatrix = numpy.corrcoef(X, rowvar=0)
fig, ax = plt.subplots()
heatmap = ax.pcolor(correlationMatrix, cmap=plt.cm.Blues)
plt.show()


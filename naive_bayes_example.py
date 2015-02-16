import csv
import numpy
import scipy
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation

filename = "wdbc.data"


#Labels for the outputs
output_labels = ["M", "B"]

data = list(csv.reader(open(filename, 'rU' )))
data =  scipy.array(data)
X = data[0:,2:31]
#Convert to array of floats (currently text):
X = X.astype(numpy.float)

y = data[0:, 1]

#Convert the labels into numerical values:
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y)
y = label_encoder.transform(y)

#Check label conversion:
print 'Converted: ', y[0]

#Split the datset into training and validation sets:
X_train, X_test, y_train, y_test = train_test_split(X, y)



################################
#Illustrate first with one case before going into cross-validation procedure.

#Gaussian since our X is continuous (if event-based/discrete, would use multinomial or Bernouilli)
nbmodel1 = GaussianNB().fit(X_train, y_train) 
y_pred = nbmodel1.predict(X_train)

#Validate classifier
print 'Proportion mislabeled: ', float((y_train != y_pred).sum())/len(y_train)

#Now see how performs with the test data:
y_pred_test = nbmodel1.predict(X_test)
print 'Proportion mislabeled: ', float((y_test != y_pred_test).sum())/len(y_test)

#Compare with model trained on randomised X data:
X_random = numpy.copy(X_train)
numpy.random.shuffle(X_random)
nbmodel_random = GaussianNB().fit(X_random, y_train)
y_pred_random = nbmodel_random.predict(X_random)

#Validate classifier
print 'Proportion mislabeled random model: ', float((y_train != y_pred_random).sum())/len(y_train)

#Now see how performs with the test data:
y_random_test = nbmodel_random.predict(X_test)
print 'Proportion mislabeled random model: ', float((y_test != y_random_test).sum())/len(y_test)


#################################

#k-fold cross-validation
#k is the number of samples the data is split into.
#When k=n, then same as leave one out. 
k_fold = cross_validation.KFold(n=len(X_train)/2, n_folds=10)

for train_indices, test_indices in k_fold:
	#Show the indices for training and test samples
	print('Train: %s, Test: %s' % (train_indices, test_indices))
	nbmodel = GaussianNB().fit(X_train[train_indices], y_train[train_indices])
	#Show proportion correct.
	print ('Proportion correct: %s' % (nbmodel.score(X_train[test_indices], y_train[test_indices])))
#More compact version
[GaussianNB().fit(X_train[train], y_train[train]).score(X_train[test], y_train[test]) for train, test in k_fold]


#Leave one out cross-validation (equivalent to when n_folds=len(X_train))
loo = cross_validation.LeaveOneOut(n=len(X_train));
[nbmodel.score(X_train[test], y_train[test]) for train, test in loo]

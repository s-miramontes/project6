"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""

from regression import (utils, logreg)
import numpy as np
from sklearn.preprocessing import StandardScaler
from random import gauss


def test_loss_gradient():
	"""
	Testing that the loss is smaller post trainig.
	Making use of loss_history list.
	"""

	Xtrain, Xval, ytrain, yval = utils.loadDataset(features=['Penicillin V Potassium 500 MG',
								 'Computed tomography of chest and abdomen','Plain chest X-ray (procedure)',
								 'Low Density Lipoprotein Cholesterol', 'Creatinine', 'AGE_DIAGNOSIS'],
								  split_percent=0.7, split_state=40)
	np.random.seed(40)
	model = logreg.LogisticRegression(6, max_iter = 10000,learning_rate = 0.000001, batch_size=15) # all other parameters stay the same
	model.train_model(Xtrain, ytrain, Xval, yval)

	# assert losses are not the same and last one (smaller)
	assert model.loss_history_train[-1] < model.loss_history_train[0]


def test_updates():
	"""
	Testing whether weights update after training.
	"""

	Xtrain, Xval, ytrain, yval = utils.loadDataset(features=['Penicillin V Potassium 500 MG',
								 'Computed tomography of chest and abdomen','Plain chest X-ray (procedure)',
								 'Low Density Lipoprotein Cholesterol', 'Creatinine', 'AGE_DIAGNOSIS'],
								  split_percent=0.7, split_state=40)
	np.random.seed(40)
	model = logreg.LogisticRegression(6, max_iter = 100000,learning_rate = 0.000001, batch_size=15)

	# make up some random weights to see if they change
	model.W = np.array([1, 2, 3, 4, 1, 1, 1])
	# save for comparison
	dummy_w = model.W.copy()

	# now train
	model.train_model(Xtrain, ytrain, Xval, yval)

	assert (not np.array_equal(dummy_w, model.W))

def test_predict():
	# Check that self.W is being updated as expected
	# and produces reasonable estimates for NSCLC classification

	# Check accuracy of model after training

	Xtrain, Xval, ytrain, yval = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                    'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol', 
                                    'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.7, split_state=42)
	np.random.seed(42)
    
	# using scaler for loaded data
	sc = StandardScaler()
	Xtrain = sc.fit_transform(Xtrain)
	Xval = sc.transform (Xval)
	#np.random.seed(42) 

	# instantiate model
	model = logreg.LogisticRegression(num_feats = 6, max_iter = 100, learning_rate = 0.04, batch_size = 15)

	Xval_bias = np.hstack([Xval, np.ones((Xval.shape[0], 1))])
	pred = model.make_prediction(Xval_bias)

	# identify those with prob greater than 0.5 and less than
	pred = np.where(pred > 0.5, 1, pred)
	pred = np.where(pred < 0.5, 0, pred)

	# add those 1's and 0's for accuracy 
	acc = np.sum(yval == pred) / len(yval)
	assert acc < 0.50 # this should be backwards





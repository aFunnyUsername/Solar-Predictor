#general ML utilities like reading data, splitting into X and Y, splitting into train/test
#normalizing, standardizing, getting statistics, etc.

import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing as pp
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt


#this function will create a datetime column out of the date and time columns
#currently present in the tmy dataset.  This is to make it consistent with
#pandas parsing of datetimes
def parse_index_time_column(fp, variables):
	tmy = pd.read_csv(fp, header=1, low_memory=False)
	time_list = []
	for i, time in enumerate(tmy['Time (HH:MM)']):
		if time == '24:00:00':
			time = '0:00'
			time_list.append(time)
		else:
			time_list.append(time)
	#This is done because tmy stores its time data from hour 1 to hour 24 but we
	#need it from hour 0 to hour 23.
	time_list.insert(0, time_list.pop(len(time_list) - 1))
	tmy['Time (HH:MM)'] = time_list

	datetime_list = []
	for i, date in enumerate(tmy['Date (MM/DD/YYYY)']):
		datetime = date + ' ' + time_list[i]
		datetime_list.append(datetime)
	
	tmy['DateTime'] = datetime_list
	tmy['DateTime'] = pd.to_datetime(tmy['DateTime'])
	
	tmy_reduced = pd.DataFrame()
	tmy_reduced['DateTime'] = [(str(tmy.loc[i, 'DateTime'].month) +
														  '/' + str(tmy.loc[i, 'DateTime'].day) +
															' ' + str(tmy.loc[i, 'DateTime'].hour)) 
															for i in range(len(tmy['DateTime']))]
	tmy_reduced.set_index('DateTime')
	
	for variable in variables:
		tmy_reduced[variable] = tmy[variable].values
	
	return tmy_reduced

def split_dataset_validation(df, validation_entries):
	split_point = df.shape[0] - validation_entries
	dataset_df, validation_df = df.loc[:split_point], df.loc[split_point:]	
	return dataset_df, validation_df

def split_dataset_train_test(df, values, split):
	X = df[values].values
	X = X.astype('float32')
	train_size = int(X.shape[0] * split)
	train, test = X[0:train_size], X[train_size:]
	return train, test

def naive_baseline(train_dataset, test_dataset):
	#NOTE, this might just be a walk-forward validation function later, if it's
	#similar enough when we actually train/make predictions
	history = [x for x in train_dataset]
	predictions = list()
	print(len(test_dataset))
	for i in range(len(test_dataset)):
		#prediction
		yhat = history[-1]
		#NOTE, for the first value, this will be the last value of the training
		#dataset.  As we add observations, it will become the observation at
		#t-1.  Pretty neat.
		predictions.append(yhat)
		#observation
		#NOTE, what actually happend
		obs = test_dataset[i] 
		history.append(obs)
		#print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
	rmse = sqrt(mean_squared_error(test_dataset, predictions))
	return rmse










